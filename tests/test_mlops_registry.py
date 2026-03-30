"""Tests for MLflow registry wrapper — Phase 4 Step 2.

TDD RED: Tests define the expected interface for the MLflow registry module.
All functions are tested with mocked MLflow client to avoid needing a server.
"""
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# register_model tests
# ---------------------------------------------------------------------------

class TestRegisterModel:
    """Verify register_model delegates to mlflow.register_model correctly."""

    @patch("src.mlops.registry.mlflow")
    def test_calls_mlflow_register_model(self, mock_mlflow):
        from src.mlops.registry import register_model

        mock_version = MagicMock()
        mock_version.version = "1"
        mock_version.name = "fraud-xgboost"
        mock_mlflow.register_model.return_value = mock_version

        result = register_model("run123", "fraud-xgboost")

        mock_mlflow.register_model.assert_called_once_with(
            model_uri="runs:/run123/model",
            name="fraud-xgboost",
        )
        assert result.version == "1"

    @patch("src.mlops.registry.mlflow")
    def test_custom_artifact_path(self, mock_mlflow):
        from src.mlops.registry import register_model

        mock_version = MagicMock()
        mock_mlflow.register_model.return_value = mock_version

        register_model("run456", "fraud-iforest", artifact_path="sklearn-model")

        mock_mlflow.register_model.assert_called_once_with(
            model_uri="runs:/run456/sklearn-model",
            name="fraud-iforest",
        )


# ---------------------------------------------------------------------------
# get_production_model tests
# ---------------------------------------------------------------------------

class TestGetProductionModel:
    """Verify get_production_model loads model by 'production' alias."""

    @patch("src.mlops.registry.mlflow")
    @patch("src.mlops.registry.MlflowClient")
    def test_loads_model_by_alias(self, MockClient, mock_mlflow):
        from src.mlops.registry import get_production_model

        client_instance = MockClient.return_value
        mock_version = MagicMock()
        mock_version.version = "3"
        mock_version.run_id = "run_abc"
        client_instance.get_model_version_by_alias.return_value = mock_version

        mock_model = MagicMock()
        mock_mlflow.pyfunc.load_model.return_value = mock_model

        model, info = get_production_model("fraud-xgboost")

        client_instance.get_model_version_by_alias.assert_called_once_with(
            name="fraud-xgboost", alias="production"
        )
        mock_mlflow.pyfunc.load_model.assert_called_once_with(
            model_uri="models:/fraud-xgboost@production"
        )
        assert model == mock_model
        assert info["version"] == "3"
        assert info["run_id"] == "run_abc"

    @patch("src.mlops.registry.mlflow")
    @patch("src.mlops.registry.MlflowClient")
    def test_fallback_to_latest_version(self, MockClient, mock_mlflow):
        from src.mlops.registry import get_production_model
        from mlflow.exceptions import MlflowException

        client_instance = MockClient.return_value
        client_instance.get_model_version_by_alias.side_effect = MlflowException(
            "No alias 'production'"
        )

        mock_version = MagicMock()
        mock_version.version = "2"
        mock_version.run_id = "run_fallback"
        client_instance.search_model_versions.return_value = [mock_version]

        mock_model = MagicMock()
        mock_mlflow.pyfunc.load_model.return_value = mock_model

        model, info = get_production_model("fraud-xgboost")

        client_instance.search_model_versions.assert_called_once()
        assert info["version"] == "2"
        assert info["run_id"] == "run_fallback"
        assert info["alias_used"] is False

    @patch("src.mlops.registry.mlflow")
    @patch("src.mlops.registry.MlflowClient")
    def test_no_versions_raises_error(self, MockClient, mock_mlflow):
        from src.mlops.registry import get_production_model
        from mlflow.exceptions import MlflowException

        client_instance = MockClient.return_value
        client_instance.get_model_version_by_alias.side_effect = MlflowException(
            "No alias"
        )
        client_instance.search_model_versions.return_value = []

        with pytest.raises(ValueError, match="No registered versions"):
            get_production_model("fraud-xgboost")


# ---------------------------------------------------------------------------
# promote_to_production tests
# ---------------------------------------------------------------------------

class TestPromoteToProduction:
    """Verify promote_to_production sets the 'production' alias."""

    @patch("src.mlops.registry.MlflowClient")
    def test_sets_production_alias(self, MockClient):
        from src.mlops.registry import promote_to_production

        client_instance = MockClient.return_value

        promote_to_production("fraud-xgboost", 5)

        client_instance.set_registered_model_alias.assert_called_once_with(
            name="fraud-xgboost", alias="production", version="5"
        )

    @patch("src.mlops.registry.MlflowClient")
    def test_version_passed_as_string(self, MockClient):
        from src.mlops.registry import promote_to_production

        client_instance = MockClient.return_value

        promote_to_production("fraud-iforest", 12)

        call_args = client_instance.set_registered_model_alias.call_args
        assert call_args.kwargs["version"] == "12"
