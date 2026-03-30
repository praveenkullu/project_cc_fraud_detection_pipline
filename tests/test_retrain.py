"""Tests for scripts/retrain.py — Phase 4 feedback loop retraining.

TDD RED: Tests define the expected behavior of the retrain workflow.
All external dependencies (DB, MLflow, sklearn) are mocked.
"""
from unittest.mock import MagicMock, patch, call

import pytest


@pytest.fixture
def mock_repo():
    """Mock TransactionRepository with labeled data."""
    repo = MagicMock()
    # 200 labeled transactions: 10 fraud, 190 legit
    labeled = []
    for i in range(190):
        labeled.append({
            "transaction_id": f"txn_{i}",
            "card_id": f"card_{i}",
            "amount": 50.0 + i,
            "composite_score": 0.1,
            "decision": "approve",
            "is_fraud": False,
        })
    for i in range(10):
        labeled.append({
            "transaction_id": f"fraud_{i}",
            "card_id": f"fcard_{i}",
            "amount": 500.0 + i,
            "composite_score": 0.8,
            "decision": "decline",
            "is_fraud": True,
        })
    repo.get_labeled_transactions.return_value = labeled
    repo.save_promotion_event.return_value = 1
    return repo


class TestRetrainInsufficientData:
    """Exit gracefully when not enough labeled data."""

    @patch("scripts.retrain.get_repository")
    def test_exits_on_insufficient_data(self, mock_get_repo):
        from scripts.retrain import run_retrain

        repo = MagicMock()
        repo.get_labeled_transactions.return_value = [
            {"transaction_id": "t1", "is_fraud": False, "amount": 10.0,
             "card_id": "c1", "composite_score": 0.1, "decision": "approve"}
        ] * 50  # only 50, below threshold
        mock_get_repo.return_value = repo

        result = run_retrain(min_samples=100)
        assert result["status"] == "insufficient_data"


class TestRetrainPromotion:
    """Promote model when AUC-PR improves beyond threshold."""

    @patch("scripts.retrain.promote_to_production")
    @patch("scripts.retrain.register_model")
    @patch("scripts.retrain.mlflow")
    @patch("scripts.retrain.get_repository")
    def test_promotes_when_improved(
        self, mock_get_repo, mock_mlflow, mock_register, mock_promote, mock_repo
    ):
        from scripts.retrain import run_retrain

        mock_get_repo.return_value = mock_repo

        # Mock MLflow run
        mock_run = MagicMock()
        mock_run.info.run_id = "new_run_123"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        # Mock model registration
        mock_version = MagicMock()
        mock_version.version = "5"
        mock_register.return_value = mock_version

        # Mock current production model with lower AUC-PR
        with patch("scripts.retrain.evaluate_model") as mock_eval:
            # candidate AUC-PR = 0.90, production AUC-PR = 0.85
            mock_eval.side_effect = [0.90, 0.85]
            with patch("scripts.retrain.get_production_model") as mock_get_prod:
                mock_prod_model = MagicMock()
                mock_get_prod.return_value = (mock_prod_model, {"version": "4", "run_id": "old_run"})
                result = run_retrain(threshold=0.005)

        assert result["status"] == "promoted"
        mock_promote.assert_called_once()
        mock_repo.save_promotion_event.assert_called_once()

    @patch("scripts.retrain.register_model")
    @patch("scripts.retrain.mlflow")
    @patch("scripts.retrain.get_repository")
    def test_no_promotion_below_threshold(
        self, mock_get_repo, mock_mlflow, mock_register, mock_repo
    ):
        from scripts.retrain import run_retrain

        mock_get_repo.return_value = mock_repo

        mock_run = MagicMock()
        mock_run.info.run_id = "new_run_456"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        mock_version = MagicMock()
        mock_version.version = "5"
        mock_register.return_value = mock_version

        with patch("scripts.retrain.evaluate_model") as mock_eval:
            # candidate 0.852, production 0.850 — improvement 0.002 < 0.005 threshold
            mock_eval.side_effect = [0.852, 0.850]
            with patch("scripts.retrain.get_production_model") as mock_get_prod:
                mock_prod_model = MagicMock()
                mock_get_prod.return_value = (mock_prod_model, {"version": "4", "run_id": "old_run"})
                result = run_retrain(threshold=0.005)

        assert result["status"] == "no_improvement"
        mock_repo.save_promotion_event.assert_called_once()
        event = mock_repo.save_promotion_event.call_args[0][0]
        assert event["promoted"] is False


class TestRetrainFirstModel:
    """When no production model exists, always promote."""

    @patch("scripts.retrain.promote_to_production")
    @patch("scripts.retrain.register_model")
    @patch("scripts.retrain.mlflow")
    @patch("scripts.retrain.get_repository")
    def test_promotes_first_model(
        self, mock_get_repo, mock_mlflow, mock_register, mock_promote, mock_repo
    ):
        from scripts.retrain import run_retrain

        mock_get_repo.return_value = mock_repo

        mock_run = MagicMock()
        mock_run.info.run_id = "first_run"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        mock_version = MagicMock()
        mock_version.version = "1"
        mock_register.return_value = mock_version

        with patch("scripts.retrain.evaluate_model", return_value=0.85):
            with patch("scripts.retrain.get_production_model") as mock_get_prod:
                mock_get_prod.side_effect = ValueError("No registered versions")
                result = run_retrain(threshold=0.005)

        assert result["status"] == "promoted"
        mock_promote.assert_called_once()
