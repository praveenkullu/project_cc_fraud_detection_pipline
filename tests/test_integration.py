"""Integration tests for the full Docker Compose stack — Phase 3 Step 19.

These tests require `docker compose up -d` to be running.
Skipped automatically if the API is not reachable.

Run with: pytest tests/test_integration.py -v --no-cov
"""
import time

import pytest
import httpx

API_URL = "http://localhost:8000"


def _api_reachable() -> bool:
    try:
        r = httpx.get(f"{API_URL}/health", timeout=2.0)
        return r.status_code == 200
    except (httpx.ConnectError, httpx.ReadTimeout):
        return False


pytestmark = pytest.mark.skipif(
    not _api_reachable(),
    reason="Docker Compose stack not running (API unreachable at localhost:8000)",
)


@pytest.fixture
def sample_transaction():
    return {
        "card_id": "integ_card_001",
        "amount": 99.99,
        "merchant_id": "merch_integ",
        "timestamp": 86400.0,
        "ip_address": "10.0.0.1",
        "account_age_hours": 720.0,
        "bin_number": "400000",
        "features": {},
    }


class TestEndToEndScoring:
    """POST /predict returns valid ScoringResult."""

    def test_score_transaction_returns_200(self, sample_transaction):
        r = httpx.post(f"{API_URL}/predict", json=sample_transaction, timeout=10.0)
        assert r.status_code == 200

    def test_score_transaction_schema(self, sample_transaction):
        r = httpx.post(f"{API_URL}/predict", json=sample_transaction, timeout=10.0)
        body = r.json()
        assert "transaction_id" in body
        assert "decision" in body
        assert "composite_score" in body
        assert body["decision"] in {"approve", "manual_review", "decline", "step_up_auth"}
        assert 0.0 <= body["composite_score"] <= 1.0


class TestHardRuleEarlyExit:
    """Hard rule transactions should not invoke ML tier."""

    def test_blocked_bin_decline(self):
        payload = {
            "card_id": "integ_blocked",
            "amount": 50.0,
            "merchant_id": "m1",
            "timestamp": 1000.0,
            "ip_address": "10.0.0.2",
            "account_age_hours": 100.0,
            "bin_number": "999999",
            "features": {},
        }
        r = httpx.post(f"{API_URL}/predict", json=payload, timeout=10.0)
        body = r.json()
        assert body["decision"] == "decline"
        tier_names = [t.get("tier") for t in body["tier_results"]]
        assert "hard_rules" in tier_names
        assert "ml_scoring" not in tier_names

    def test_new_account_high_amount_decline(self):
        payload = {
            "card_id": "integ_new_acct",
            "amount": 750.0,
            "merchant_id": "m1",
            "timestamp": 1000.0,
            "ip_address": "10.0.0.3",
            "account_age_hours": 12.0,
            "bin_number": "400000",
            "features": {},
        }
        r = httpx.post(f"{API_URL}/predict", json=payload, timeout=10.0)
        body = r.json()
        assert body["decision"] == "decline"


class TestVelocityDecline:
    """11+ transactions on same card triggers velocity decline."""

    def test_velocity_decline_after_11_transactions(self):
        card_id = f"integ_velocity_{int(time.time())}"
        for i in range(10):
            payload = {
                "card_id": card_id,
                "amount": 10.0,
                "merchant_id": "m1",
                "timestamp": float(i),
                "ip_address": "10.0.0.4",
                "account_age_hours": 100.0,
                "bin_number": "400000",
                "features": {},
            }
            httpx.post(f"{API_URL}/predict", json=payload, timeout=10.0)

        # 11th transaction should trigger velocity decline
        payload = {
            "card_id": card_id,
            "amount": 10.0,
            "merchant_id": "m1",
            "timestamp": 10.0,
            "ip_address": "10.0.0.4",
            "account_age_hours": 100.0,
            "bin_number": "400000",
            "features": {},
        }
        r = httpx.post(f"{API_URL}/predict", json=payload, timeout=10.0)
        body = r.json()
        assert body["decision"] == "decline"


class TestHealthAndMetrics:
    """Verify observability endpoints."""

    def test_health(self):
        r = httpx.get(f"{API_URL}/health", timeout=5.0)
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_metrics_prometheus_format(self):
        r = httpx.get(f"{API_URL}/metrics", timeout=5.0)
        assert r.status_code == 200
        assert "fraud_predictions_total" in r.text
