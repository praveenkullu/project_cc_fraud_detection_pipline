"""Tests for database repository — Phase 3 Step 15.

TDD RED: Tests define the expected interface for TransactionRepository.
Uses SQLite in-memory for fast unit tests.
"""
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.database.models import Base, Transaction, Chargeback
from src.database.repository import TransactionRepository


@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def session(engine):
    with Session(engine) as s:
        yield s


@pytest.fixture
def repo(session):
    return TransactionRepository(session)


# ---------------------------------------------------------------------------
# Sample data helpers
# ---------------------------------------------------------------------------

def _make_scoring_result(decision="approve", composite=0.15, latency=12.5):
    """Create a dict mimicking PredictResponse fields."""
    return {
        "transaction_id": str(uuid.uuid4()),
        "card_id": "card_test",
        "amount": 99.99,
        "merchant_id": "merch_001",
        "merchant_category": "retail",
        "timestamp": 1700000000.0,
        "composite_score": composite,
        "decision": decision,
        "processing_time_ms": latency,
        "tier_results": [
            {"tier": "hard_rules", "triggered": False, "action": "pass", "latency_ms": 0.3},
            {"tier": "velocity", "triggered": False, "action": "pass", "latency_ms": 1.2},
            {"tier": "ml_scoring", "triggered": True, "action": "score", "latency_ms": 8.0},
        ],
    }


# ---------------------------------------------------------------------------
# save_transaction tests
# ---------------------------------------------------------------------------

class TestSaveTransaction:

    def test_saves_transaction_row(self, repo, session):
        data = _make_scoring_result()
        txn_id = repo.save_transaction(data)

        assert txn_id == data["transaction_id"]
        row = session.get(Transaction, txn_id)
        assert row is not None
        assert row.card_id == "card_test"
        assert row.amount == 99.99
        assert row.decision == "approve"

    def test_returns_transaction_id(self, repo):
        data = _make_scoring_result()
        result = repo.save_transaction(data)
        assert result == data["transaction_id"]

    def test_saves_merchant_category(self, repo, session):
        data = _make_scoring_result()
        data["merchant_category"] = "electronics"
        txn_id = repo.save_transaction(data)
        row = session.get(Transaction, txn_id)
        assert row.merchant_category == "electronics"

    def test_nullable_merchant_category(self, repo, session):
        data = _make_scoring_result()
        data.pop("merchant_category", None)
        txn_id = repo.save_transaction(data)
        row = session.get(Transaction, txn_id)
        assert row.merchant_category is None


# ---------------------------------------------------------------------------
# save_scoring_log tests
# ---------------------------------------------------------------------------

class TestSaveScoringLog:

    def test_saves_tier_results(self, repo, session):
        data = _make_scoring_result()
        txn_id = repo.save_transaction(data)
        repo.save_scoring_log(txn_id, data["tier_results"])

        from src.database.models import ScoringLog
        logs = session.query(ScoringLog).filter_by(transaction_id=txn_id).all()
        assert len(logs) == 3
        tier_names = {log.tier_name for log in logs}
        assert tier_names == {"hard_rules", "velocity", "ml_scoring"}

    def test_scoring_log_values(self, repo, session):
        data = _make_scoring_result()
        txn_id = repo.save_transaction(data)
        repo.save_scoring_log(txn_id, data["tier_results"])

        from src.database.models import ScoringLog
        hr_log = session.query(ScoringLog).filter_by(
            transaction_id=txn_id, tier_name="hard_rules"
        ).first()
        assert hr_log.triggered is False
        assert hr_log.action == "pass"
        assert hr_log.latency_ms == 0.3

    def test_empty_tier_results(self, repo, session):
        data = _make_scoring_result()
        txn_id = repo.save_transaction(data)
        repo.save_scoring_log(txn_id, [])

        from src.database.models import ScoringLog
        logs = session.query(ScoringLog).filter_by(transaction_id=txn_id).all()
        assert len(logs) == 0


# ---------------------------------------------------------------------------
# get_labeled_transactions tests
# ---------------------------------------------------------------------------

class TestGetLabeledTransactions:

    def _seed_transactions(self, session, count=5, days_ago=10):
        """Seed transactions with some chargebacks."""
        txns = []
        now = datetime.now(timezone.utc)
        for i in range(count):
            txn_id = str(uuid.uuid4())
            txn = Transaction(
                id=txn_id,
                card_id=f"card_{i}",
                amount=100.0 + i,
                merchant_id="m1",
                timestamp=1700000000.0 + i,
                composite_score=0.1 * i,
                decision="approve",
                processing_time_ms=5.0,
                created_at=now - timedelta(days=days_ago - i),
            )
            session.add(txn)
            txns.append(txn)

        session.flush()

        # Add chargebacks to first 2 transactions
        for txn in txns[:2]:
            cb = Chargeback(
                transaction_id=txn.id,
                reported_at=now - timedelta(days=5),
                is_fraud=True,
            )
            session.add(cb)

        session.commit()
        return txns

    def test_returns_transactions_within_days(self, repo, session):
        self._seed_transactions(session, count=5, days_ago=10)
        results = repo.get_labeled_transactions(days=30)
        assert len(results) == 5

    def test_filters_by_days(self, repo, session):
        self._seed_transactions(session, count=5, days_ago=10)
        # Only transactions from last 3 days
        results = repo.get_labeled_transactions(days=3)
        # days_ago ranges from 10 to 6 (10-0, 10-1, ..., 10-4)
        # Only i=3 (7 days ago) and i=4 (6 days ago) are within 3 days? No.
        # created_at = now - timedelta(days=10-i), so i=0→10d, i=1→9d, i=2→8d, i=3→7d, i=4→6d
        # None are within 3 days
        assert len(results) == 0

    def test_returns_dicts_with_is_fraud_label(self, repo, session):
        self._seed_transactions(session, count=3, days_ago=5)
        results = repo.get_labeled_transactions(days=30)
        assert len(results) == 3
        # Each result should have is_fraud key
        for r in results:
            assert "is_fraud" in r
            assert "transaction_id" in r

    def test_chargeback_labeled_as_fraud(self, repo, session):
        self._seed_transactions(session, count=3, days_ago=5)
        results = repo.get_labeled_transactions(days=30)
        fraud_count = sum(1 for r in results if r["is_fraud"])
        assert fraud_count == 2  # first 2 have chargebacks

    def test_no_chargeback_labeled_as_legitimate(self, repo, session):
        self._seed_transactions(session, count=3, days_ago=5)
        results = repo.get_labeled_transactions(days=30)
        legit = [r for r in results if not r["is_fraud"]]
        assert len(legit) == 1  # 3rd transaction has no chargeback


# ---------------------------------------------------------------------------
# save_promotion_event tests
# ---------------------------------------------------------------------------

class TestSavePromotionEvent:

    def test_saves_promotion_event(self, repo, session):
        from src.database.models import ModelPromotion
        event = {
            "model_name": "fraud-xgboost",
            "from_version": 2,
            "to_version": 3,
            "old_auc_pr": 0.85,
            "new_auc_pr": 0.87,
            "improvement": 0.02,
            "promoted": True,
        }
        promo_id = repo.save_promotion_event(event)
        assert promo_id is not None

        result = session.get(ModelPromotion, promo_id)
        assert result.model_name == "fraud-xgboost"
        assert result.promoted is True

    def test_saves_non_promoted_event(self, repo, session):
        from src.database.models import ModelPromotion
        event = {
            "model_name": "fraud-xgboost",
            "from_version": 2,
            "to_version": 3,
            "old_auc_pr": 0.85,
            "new_auc_pr": 0.853,
            "improvement": 0.003,
            "promoted": False,
        }
        promo_id = repo.save_promotion_event(event)
        result = session.get(ModelPromotion, promo_id)
        assert result.promoted is False
        assert result.improvement == 0.003

    def test_nullable_fields(self, repo, session):
        from src.database.models import ModelPromotion
        event = {
            "model_name": "fraud-iforest",
            "from_version": None,
            "to_version": 1,
            "old_auc_pr": None,
            "new_auc_pr": 0.80,
            "improvement": 0.0,
            "promoted": True,
        }
        promo_id = repo.save_promotion_event(event)
        result = session.get(ModelPromotion, promo_id)
        assert result.from_version is None
        assert result.old_auc_pr is None
