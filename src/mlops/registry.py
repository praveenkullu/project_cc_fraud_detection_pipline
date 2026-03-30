"""MLflow Model Registry wrapper for the fraud detection pipeline.

Phase 4 — provides register, load, and promote operations against the
MLflow Model Registry using the 2.x alias-based API.
"""
import logging

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


def register_model(
    run_id: str,
    model_name: str,
    artifact_path: str = "model",
) -> "mlflow.entities.model_registry.ModelVersion":
    """Register a logged model artifact in the MLflow Model Registry."""
    model_uri = f"runs:/{run_id}/{artifact_path}"
    return mlflow.register_model(model_uri=model_uri, name=model_name)


def get_production_model(model_name: str) -> tuple:
    """Load the production model from the registry.

    Tries the 'production' alias first. Falls back to the latest version
    if no alias is set. Returns (model, info_dict).
    """
    client = MlflowClient()

    try:
        version_info = client.get_model_version_by_alias(
            name=model_name, alias="production"
        )
        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}@production"
        )
        return model, {
            "version": version_info.version,
            "run_id": version_info.run_id,
            "alias_used": True,
        }
    except Exception:
        logger.warning(
            "No 'production' alias for %s, falling back to latest version",
            model_name,
        )

    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(
            f"No registered versions found for model '{model_name}'"
        )

    latest = max(versions, key=lambda v: int(v.version))
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}/{latest.version}"
    )
    return model, {
        "version": latest.version,
        "run_id": latest.run_id,
        "alias_used": False,
    }


def promote_to_production(model_name: str, version: int) -> None:
    """Set the 'production' alias on the given model version."""
    client = MlflowClient()
    client.set_registered_model_alias(
        name=model_name, alias="production", version=str(version)
    )
