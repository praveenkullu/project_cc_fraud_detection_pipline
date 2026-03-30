"""One-time script to register Phase 1 models in MLflow registry.

Loads the existing .pkl model files, logs them to MLflow with the
correct flavor, registers them, and sets the 'production' alias.

Usage:
    python -m scripts.register_models
"""
import logging
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from src.mlops.registry import promote_to_production

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"


def register_existing_models():
    """Register Phase 1 models in MLflow and set production alias."""
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
    mlflow.set_tracking_uri(mlflow_uri)

    # Register XGBoost model
    xgb_model = joblib.load(MODELS_DIR / "xgboost_smt.pkl")
    with mlflow.start_run(run_name="register-xgboost-phase1") as run:
        mlflow.xgboost.log_model(xgb_model, artifact_path="model")
        mlflow.log_param("source", "phase1_pkl")
        result = mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/model",
            name="fraud-xgboost",
        )
        promote_to_production("fraud-xgboost", int(result.version))
        logger.info("Registered fraud-xgboost v%s as production", result.version)

    # Register IsolationForest model
    iforest_model = joblib.load(MODELS_DIR / "iforest_model.pkl")
    with mlflow.start_run(run_name="register-iforest-phase1") as run:
        mlflow.sklearn.log_model(iforest_model, artifact_path="model")
        mlflow.log_param("source", "phase1_pkl")
        result = mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/model",
            name="fraud-iforest",
        )
        promote_to_production("fraud-iforest", int(result.version))
        logger.info("Registered fraud-iforest v%s as production", result.version)

    print("Done. Both models registered and set to 'production' alias.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    register_existing_models()
