"""Feedback loop retraining script for the fraud detection pipeline.

Phase 4 — queries PostgreSQL for labeled transactions (via chargebacks),
trains a candidate XGBoost model, evaluates against the current production
model, and conditionally promotes if AUC-PR improves beyond threshold.

Usage:
    python -m scripts.retrain [--days 90] [--min-samples 100] [--threshold 0.005]

# crontab: 0 2 * * 0  python /app/scripts/retrain.py
"""
import argparse
import logging
import os

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.mlops.registry import (
    get_production_model,
    promote_to_production,
    register_model,
)

logger = logging.getLogger(__name__)


def get_repository():
    """Create a TransactionRepository connected to DATABASE_URL."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from src.database.repository import TransactionRepository

    database_url = os.environ.get(
        "DATABASE_URL", "postgresql://fraud_user:fraud_pass@localhost:5432/fraud_db"
    )
    engine = create_engine(database_url)
    session = Session(engine)
    return TransactionRepository(session)


def evaluate_model(model, X_val: pd.DataFrame, y_val: np.ndarray) -> float:
    """Evaluate a model and return AUC-PR score."""
    y_proba = model.predict_proba(X_val)[:, 1]
    return float(average_precision_score(y_val, y_proba))


def run_retrain(
    days: int = 90,
    min_samples: int = 100,
    threshold: float = 0.005,
) -> dict:
    """Execute the full retraining workflow.

    Returns a dict with 'status' key: 'insufficient_data', 'promoted', or 'no_improvement'.
    """
    repo = get_repository()

    # Step 1: Get labeled transactions
    labeled = repo.get_labeled_transactions(days=days)
    if len(labeled) < min_samples:
        logger.warning(
            "Only %d labeled transactions (need %d). Skipping retrain.",
            len(labeled), min_samples,
        )
        return {"status": "insufficient_data", "count": len(labeled)}

    # Step 2: Prepare features
    df = pd.DataFrame(labeled)
    feature_cols = ["amount", "composite_score"]
    X = df[feature_cols]
    y = df["is_fraud"].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Step 3: Train candidate model
    with mlflow.start_run() as run:
        candidate = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=float(len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)),
            random_state=42,
            eval_metric="aucpr",
        )
        candidate.fit(X_train, y_train)

        candidate_auc_pr = evaluate_model(candidate, X_val, y_val)
        mlflow.log_metric("candidate_auc_pr", candidate_auc_pr)
        mlflow.log_param("n_samples", len(labeled))
        mlflow.log_param("n_fraud", int(y.sum()))
        mlflow.xgboost.log_model(candidate, artifact_path="model")

        # Register candidate
        model_version = register_model(
            run_id=run.info.run_id,
            model_name="fraud-xgboost",
            artifact_path="model",
        )

        # Step 4: Compare with production
        try:
            prod_model, prod_info = get_production_model("fraud-xgboost")
            prod_auc_pr = evaluate_model(prod_model, X_val, y_val)
            mlflow.log_metric("production_auc_pr", prod_auc_pr)
        except (ValueError, Exception):
            # No production model — first training, always promote
            logger.info("No production model found. Promoting first model.")
            promote_to_production("fraud-xgboost", int(model_version.version))
            repo.save_promotion_event({
                "model_name": "fraud-xgboost",
                "from_version": None,
                "to_version": int(model_version.version),
                "old_auc_pr": None,
                "new_auc_pr": candidate_auc_pr,
                "improvement": 0.0,
                "promoted": True,
            })
            return {
                "status": "promoted",
                "version": model_version.version,
                "auc_pr": candidate_auc_pr,
            }

        improvement = candidate_auc_pr - prod_auc_pr
        mlflow.log_metric("improvement", improvement)

        promoted = improvement > threshold
        repo.save_promotion_event({
            "model_name": "fraud-xgboost",
            "from_version": int(prod_info["version"]),
            "to_version": int(model_version.version),
            "old_auc_pr": prod_auc_pr,
            "new_auc_pr": candidate_auc_pr,
            "improvement": improvement,
            "promoted": promoted,
        })

        if promoted:
            promote_to_production("fraud-xgboost", int(model_version.version))
            logger.info(
                "Promoted version %s (AUC-PR: %.4f → %.4f, +%.4f)",
                model_version.version, prod_auc_pr, candidate_auc_pr, improvement,
            )
            return {
                "status": "promoted",
                "version": model_version.version,
                "old_auc_pr": prod_auc_pr,
                "new_auc_pr": candidate_auc_pr,
                "improvement": improvement,
            }
        else:
            logger.info(
                "No promotion: improvement %.4f below threshold %.4f",
                improvement, threshold,
            )
            return {
                "status": "no_improvement",
                "version": model_version.version,
                "old_auc_pr": prod_auc_pr,
                "new_auc_pr": candidate_auc_pr,
                "improvement": improvement,
            }


def main():
    parser = argparse.ArgumentParser(description="Retrain fraud detection model")
    parser.add_argument("--days", type=int, default=90, help="Days of labeled data to use")
    parser.add_argument("--min-samples", type=int, default=100, help="Minimum labeled samples")
    parser.add_argument("--threshold", type=float, default=0.005, help="AUC-PR improvement threshold")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = run_retrain(days=args.days, min_samples=args.min_samples, threshold=args.threshold)
    print(f"Retrain result: {result}")


if __name__ == "__main__":
    main()
