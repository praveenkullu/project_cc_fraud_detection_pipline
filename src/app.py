"""FastAPI scoring application for the fraud detection pipeline.

Implements a tiered pipeline:
  Tier 1 — Hard Rules (BIN blocklist, new-account high-amount)
  Tier 2 — Velocity checks (sliding-window counters)
  Tier 3 — Feature enrichment (log_amount, hour_of_day, amount_zscore)
  Tier 4 — ML scoring (XGBoost + IsolationForest composite)
  Tier 5 — Decision routing (approve / manual_review / decline / step_up_auth)
"""
import math
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field

from src.features import log_amount, hour_of_day, amount_zscore
from src.hard_rules import check_hard_rules
from src.scoring import route_decision
from src.velocity_redis import get_velocity_checker


def get_production_model_fn(model_name: str):
    """Wrapper to load model from MLflow registry. Patchable for testing."""
    from src.mlops.registry import get_production_model
    return get_production_model(model_name)

MODELS_DIR = Path(__file__).parent.parent / "models"

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

PREDICTIONS_TOTAL = Counter(
    "fraud_predictions_total",
    "Total predictions made",
    ["decision"],
)
PREDICTION_LATENCY = Histogram(
    "fraud_prediction_latency_seconds",
    "Prediction latency in seconds",
)
COMPOSITE_SCORE = Histogram(
    "fraud_composite_score",
    "Distribution of composite fraud scores",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
MODEL_VERSION = Gauge(
    "fraud_model_version",
    "Current production model version (0 = local pkl)",
)

# ---------------------------------------------------------------------------
# Application lifespan — load models once at startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Always load config and feature columns from local files
    app.state.config = joblib.load(MODELS_DIR / "scoring_config.pkl")
    app.state.feature_cols = joblib.load(MODELS_DIR / "feature_cols.pkl")

    # Try MLflow registry first, fall back to local pkl files
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        try:
            xgb_model, xgb_info = get_production_model_fn("fraud-xgboost")
            iforest_model, iforest_info = get_production_model_fn("fraud-iforest")
            app.state.xgb = xgb_model
            app.state.iforest = iforest_model
            app.state.model_source = "mlflow_registry"
            app.state.model_version = {
                "xgboost": xgb_info,
                "iforest": iforest_info,
            }
            MODEL_VERSION.set(int(xgb_info["version"]))
        except Exception:
            app.state.xgb = joblib.load(MODELS_DIR / "xgboost_smt.pkl")
            app.state.iforest = joblib.load(MODELS_DIR / "iforest_model.pkl")
            app.state.model_source = "local_pkl"
            app.state.model_version = None
            MODEL_VERSION.set(0)
    else:
        app.state.xgb = joblib.load(MODELS_DIR / "xgboost_smt.pkl")
        app.state.iforest = joblib.load(MODELS_DIR / "iforest_model.pkl")
        app.state.model_source = "local_pkl"
        app.state.model_version = None
        MODEL_VERSION.set(0)

    velocity_backend = os.environ.get("VELOCITY_BACKEND", "memory")
    redis_client = None
    if velocity_backend == "redis":
        import redis
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
    app.state.velocity = get_velocity_checker(
        backend=velocity_backend, redis_client=redis_client
    )
    yield


app = FastAPI(title="Fraud Detection API", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class TransactionRequest(BaseModel):
    card_id: str
    amount: float = Field(..., gt=0)
    merchant_id: str
    timestamp: float
    ip_address: str
    account_age_hours: float = Field(..., ge=0)
    bin_number: str
    features: dict[str, float] = {}


class PredictResponse(BaseModel):
    transaction_id: str
    decision: str
    composite_score: float
    xgb_score: float
    iforest_score: float
    tier_results: list[dict]
    latency_ms: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    """Liveness check."""
    return {"status": "healthy"}


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model/info")
def model_info():
    """Return metadata about the currently loaded models."""
    cfg = app.state.config
    return {
        "model_source": app.state.model_source,
        "model_version": app.state.model_version,
        "scoring_config": {
            "best_alpha": float(cfg["best_alpha"]),
            "amount_mean": float(cfg["amount_mean"]),
            "amount_std": float(cfg["amount_std"]),
        },
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: TransactionRequest):
    """Run the full tiered fraud scoring pipeline."""
    t_start = time.perf_counter()
    transaction_id = str(uuid.uuid4())
    tier_results: list[dict] = []
    velocity = app.state.velocity

    # ------------------------------------------------------------------
    # Tier 1 — Hard Rules
    # ------------------------------------------------------------------
    declined, rules = check_hard_rules(req.bin_number, req.amount, req.account_age_hours)
    if declined:
        tier_results.append({"tier": "hard_rules", "triggered": rules})
        latency_ms = (time.perf_counter() - t_start) * 1000
        PREDICTIONS_TOTAL.labels(decision="decline").inc()
        PREDICTION_LATENCY.observe(time.perf_counter() - t_start)
        COMPOSITE_SCORE.observe(1.0)
        return PredictResponse(
            transaction_id=transaction_id,
            decision="decline",
            composite_score=1.0,
            xgb_score=1.0,
            iforest_score=1.0,
            tier_results=tier_results,
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    # Tier 2 — Velocity
    # ------------------------------------------------------------------
    flags = velocity.record_and_check(req.card_id, req.ip_address, req.amount, req.timestamp)
    if flags:
        tier_results.append({"tier": "velocity", "flags": flags})
        if any(f["action"] == "decline" for f in flags):
            latency_ms = (time.perf_counter() - t_start) * 1000
            PREDICTIONS_TOTAL.labels(decision="decline").inc()
            PREDICTION_LATENCY.observe(time.perf_counter() - t_start)
            COMPOSITE_SCORE.observe(1.0)
            return PredictResponse(
                transaction_id=transaction_id,
                decision="decline",
                composite_score=1.0,
                xgb_score=1.0,
                iforest_score=1.0,
                tier_results=tier_results,
                latency_ms=latency_ms,
            )

    # ------------------------------------------------------------------
    # Tier 3 — Feature Enrichment
    # ------------------------------------------------------------------
    feature_cols = app.state.feature_cols
    X = pd.DataFrame(
        np.zeros((1, len(feature_cols))),
        columns=feature_cols,
    )
    # Overlay any caller-provided raw features
    for col, val in req.features.items():
        if col in X.columns:
            X[col] = val

    # Overwrite engineered features with pipeline-computed values
    cfg = app.state.config
    X["log_amount"] = log_amount(req.amount)
    X["hour_of_day"] = hour_of_day(req.timestamp)
    X["amount_zscore"] = amount_zscore(
        req.amount,
        mean=float(cfg["amount_mean"]),
        std=float(cfg["amount_std"]),
    )

    # ------------------------------------------------------------------
    # Tier 4 — ML Scoring
    # ------------------------------------------------------------------
    xgb_score = float(app.state.xgb.predict_proba(X)[0, 1])
    raw_iforest = float(app.state.iforest.score_samples(X)[0])
    iforest_score = 1.0 / (1.0 + math.exp(raw_iforest))

    alpha = float(cfg["best_alpha"])
    composite = alpha * xgb_score + (1.0 - alpha) * iforest_score
    composite = max(0.0, min(1.0, composite))

    tier_results.append({
        "tier": "ml_scoring",
        "xgb_score": xgb_score,
        "iforest_score": iforest_score,
        "composite_score": composite,
    })

    # ------------------------------------------------------------------
    # Tier 5 — Decision Routing
    # ------------------------------------------------------------------
    amount_24h = velocity.get_amount_24h(req.card_id)
    decision = route_decision(composite, amount_24h).value

    latency_ms = (time.perf_counter() - t_start) * 1000
    PREDICTIONS_TOTAL.labels(decision=decision).inc()
    PREDICTION_LATENCY.observe(time.perf_counter() - t_start)
    COMPOSITE_SCORE.observe(composite)

    return PredictResponse(
        transaction_id=transaction_id,
        decision=decision,
        composite_score=composite,
        xgb_score=xgb_score,
        iforest_score=iforest_score,
        tier_results=tier_results,
        latency_ms=latency_ms,
    )
