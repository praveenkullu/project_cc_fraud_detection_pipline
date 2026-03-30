"""Tests for src.app (FastAPI) — TDD RED phase.

Written BEFORE implementation. All tests will fail with ImportError
until src/app.py is created.
"""
import os
from unittest.mock import patch, MagicMock

import pytest


class TestHealthEndpoint:
    def test_health_endpoint_returns_200(self, client):
        """GET /health returns 200 with healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestPredictEndpoint:
    def test_predict_valid_transaction_returns_200(self, client, sample_transaction):
        """POST /predict with valid payload returns 200."""
        response = client.post("/predict", json=sample_transaction)
        assert response.status_code == 200

    def test_predict_response_has_required_fields(self, client, sample_transaction):
        """Response contains all required fields."""
        response = client.post("/predict", json=sample_transaction)
        body = response.json()
        assert "decision" in body
        assert "composite_score" in body
        assert "xgb_score" in body
        assert "iforest_score" in body
        assert "tier_results" in body
        assert "latency_ms" in body
        assert "transaction_id" in body

    def test_predict_decision_is_valid(self, client, sample_transaction):
        """decision field is one of the four valid values."""
        response = client.post("/predict", json=sample_transaction)
        assert response.json()["decision"] in {
            "approve", "manual_review", "decline", "step_up_auth"
        }

    def test_predict_composite_score_in_range(self, client, sample_transaction):
        """composite_score is always between 0.0 and 1.0 inclusive."""
        response = client.post("/predict", json=sample_transaction)
        score = response.json()["composite_score"]
        assert 0.0 <= score <= 1.0

    def test_predict_xgb_score_in_range(self, client, sample_transaction):
        """xgb_score is always between 0.0 and 1.0 inclusive."""
        response = client.post("/predict", json=sample_transaction)
        score = response.json()["xgb_score"]
        assert 0.0 <= score <= 1.0

    def test_predict_iforest_score_in_range(self, client, sample_transaction):
        """iforest_score is always between 0.0 and 1.0 inclusive."""
        response = client.post("/predict", json=sample_transaction)
        score = response.json()["iforest_score"]
        assert 0.0 <= score <= 1.0

    def test_predict_latency_ms_is_positive(self, client, sample_transaction):
        """latency_ms is a positive number."""
        response = client.post("/predict", json=sample_transaction)
        assert response.json()["latency_ms"] > 0

    def test_predict_tier_results_is_list(self, client, sample_transaction):
        """tier_results is a list."""
        response = client.post("/predict", json=sample_transaction)
        assert isinstance(response.json()["tier_results"], list)

    def test_predict_transaction_id_is_string(self, client, sample_transaction):
        """transaction_id is a non-empty string."""
        response = client.post("/predict", json=sample_transaction)
        tx_id = response.json()["transaction_id"]
        assert isinstance(tx_id, str)
        assert len(tx_id) > 0

    def test_predict_blocked_bin_declines_immediately(self, client, blocked_bin_transaction):
        """Blocked BIN triggers hard rules and returns decline decision."""
        response = client.post("/predict", json=blocked_bin_transaction)
        body = response.json()
        assert response.status_code == 200
        assert body["decision"] == "decline"
        # tier_results should contain hard_rules entry
        tier_names = [t.get("tier") for t in body["tier_results"]]
        assert "hard_rules" in tier_names

    def test_predict_blocked_bin_composite_score_is_one(self, client, blocked_bin_transaction):
        """Hard-rule decline short-circuits with composite_score=1.0."""
        response = client.post("/predict", json=blocked_bin_transaction)
        assert response.json()["composite_score"] == 1.0

    def test_predict_new_account_high_amount_declines(
        self, client, new_account_high_amount_transaction
    ):
        """New account + high amount triggers hard rules decline."""
        response = client.post("/predict", json=new_account_high_amount_transaction)
        body = response.json()
        assert response.status_code == 200
        assert body["decision"] == "decline"

    def test_predict_missing_amount_returns_422(self, client):
        """Request without 'amount' returns 422 Unprocessable Entity."""
        payload = {
            "card_id": "card_001",
            "merchant_id": "merch_001",
            "timestamp": 86400.0,
            "ip_address": "1.2.3.4",
            "account_age_hours": 720.0,
            "bin_number": "400000",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_negative_amount_returns_422(self, client):
        """Negative amount returns 422 (gt=0 constraint)."""
        payload = {
            "card_id": "card_001",
            "amount": -1.0,
            "merchant_id": "merch_001",
            "timestamp": 86400.0,
            "ip_address": "1.2.3.4",
            "account_age_hours": 720.0,
            "bin_number": "400000",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_zero_amount_returns_422(self, client):
        """Zero amount returns 422 (gt=0 requires strictly greater than 0)."""
        payload = {
            "card_id": "card_001",
            "amount": 0.0,
            "merchant_id": "merch_001",
            "timestamp": 86400.0,
            "ip_address": "1.2.3.4",
            "account_age_hours": 720.0,
            "bin_number": "400000",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_negative_account_age_returns_422(self, client):
        """Negative account_age_hours returns 422 (ge=0 constraint)."""
        payload = {
            "card_id": "card_001",
            "amount": 50.0,
            "merchant_id": "merch_001",
            "timestamp": 86400.0,
            "ip_address": "1.2.3.4",
            "account_age_hours": -1.0,
            "bin_number": "400000",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_velocity_decline_early_exit(self, client, sample_transaction):
        """11+ transactions on same card within 1h triggers velocity decline."""
        # Make 10 prior transactions to push velocity near threshold
        for i in range(10):
            tx = sample_transaction.copy()
            tx["timestamp"] = float(i)
            client.post("/predict", json=tx)

        # 11th should trigger decline from velocity
        tx = sample_transaction.copy()
        tx["timestamp"] = 10.0
        response = client.post("/predict", json=tx)
        body = response.json()
        assert response.status_code == 200
        assert body["decision"] == "decline"

    def test_predict_features_dict_accepted(self, client, sample_transaction):
        """Extra features in 'features' dict are accepted without error."""
        tx = sample_transaction.copy()
        tx["features"] = {"C1": 1.0, "C2": 2.0}
        response = client.post("/predict", json=tx)
        assert response.status_code == 200


class TestGetProductionModelFn:
    def test_delegates_to_registry(self):
        """get_production_model_fn delegates to src.mlops.registry."""
        mock_model = MagicMock()
        mock_info = {"version": "1", "run_id": "r1", "alias_used": True}
        with patch("src.mlops.registry.get_production_model", return_value=(mock_model, mock_info)):
            from src.app import get_production_model_fn
            model, info = get_production_model_fn("fraud-xgboost")
            assert model == mock_model
            assert info["version"] == "1"


class TestMLflowRegistryLoading:
    def test_mlflow_registry_success(self, sample_transaction):
        """When MLFLOW_TRACKING_URI is set and registry works, model_source is mlflow_registry."""
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = [[0.9, 0.1]]
        mock_model.score_samples.return_value = [-0.5]
        mock_info = {"version": "3", "run_id": "abc123", "alias_used": True}

        with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://fake:5000"}):
            with patch("src.app.get_production_model_fn") as mock_get:
                mock_get.side_effect = [
                    (mock_model, mock_info),
                    (mock_model, mock_info),
                ]
                from src.app import app as fresh_app
                from fastapi.testclient import TestClient
                with TestClient(fresh_app) as c:
                    response = c.get("/model/info")
                    assert response.json()["model_source"] == "mlflow_registry"

    def test_mlflow_registry_fallback_on_error(self, sample_transaction):
        """When MLFLOW_TRACKING_URI is set but registry fails, falls back to local pkl."""
        with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://fake:5000"}):
            with patch("src.app.get_production_model_fn") as mock_get:
                mock_get.side_effect = Exception("connection refused")
                from src.app import app as fresh_app
                from fastapi.testclient import TestClient
                with TestClient(fresh_app) as c:
                    response = c.get("/model/info")
                    assert response.json()["model_source"] == "local_pkl"


class TestVelocityBackendConfig:
    def test_redis_backend_lifespan(self, sample_transaction):
        """When VELOCITY_BACKEND=redis, app uses RedisVelocityChecker."""
        import sys
        import fakeredis

        mock_redis_mod = MagicMock()
        mock_redis_mod.Redis.from_url.return_value = fakeredis.FakeRedis(
            decode_responses=True
        )

        with patch.dict(os.environ, {"VELOCITY_BACKEND": "redis"}):
            with patch.dict(sys.modules, {"redis": mock_redis_mod}):
                from src.app import app as fresh_app
                from fastapi.testclient import TestClient
                with TestClient(fresh_app) as c:
                    response = c.post("/predict", json=sample_transaction)
                    assert response.status_code == 200


class TestMetricsEndpoint:
    def test_metrics_endpoint_returns_200(self, client):
        """GET /metrics returns 200."""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_endpoint_returns_prometheus_format(self, client):
        """GET /metrics response contains Prometheus # HELP and # TYPE lines."""
        response = client.get("/metrics")
        text = response.text
        assert "# HELP" in text
        assert "# TYPE" in text

    def test_metrics_counter_incremented_after_predict(self, client, sample_transaction):
        """After POST /predict, fraud_predictions_total counter appears in /metrics."""
        client.post("/predict", json=sample_transaction)
        response = client.get("/metrics")
        assert "fraud_predictions_total" in response.text

    def test_metrics_histogram_present(self, client, sample_transaction):
        """After POST /predict, latency histogram appears in /metrics."""
        client.post("/predict", json=sample_transaction)
        response = client.get("/metrics")
        assert "fraud_prediction_latency_seconds" in response.text

    def test_composite_score_histogram_present(self, client, sample_transaction):
        """After POST /predict, composite score histogram appears in /metrics."""
        client.post("/predict", json=sample_transaction)
        response = client.get("/metrics")
        assert "fraud_composite_score_bucket" in response.text

    def test_model_version_gauge_present(self, client, sample_transaction):
        """After startup, model version gauge appears in /metrics."""
        response = client.get("/metrics")
        assert "fraud_model_version" in response.text


class TestModelInfoEndpoint:
    def test_model_info_returns_200(self, client):
        """GET /model/info returns 200."""
        response = client.get("/model/info")
        assert response.status_code == 200

    def test_model_info_has_required_fields(self, client):
        """Response contains model_source and scoring_config."""
        response = client.get("/model/info")
        body = response.json()
        assert "model_source" in body
        assert "scoring_config" in body
        assert "best_alpha" in body["scoring_config"]

    def test_model_info_local_fallback(self, client):
        """Without MLflow, model_source should be 'local_pkl'."""
        response = client.get("/model/info")
        body = response.json()
        assert body["model_source"] == "local_pkl"
