"""Tests for database models — Phase 3 Step 14.

TDD RED: These tests define the expected schema for Transaction, ScoringLog,
and Chargeback models. Tests will FAIL until models.py is implemented.
"""
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session

from src.database.models import Base, Transaction, ScoringLog, Chargeback


# ---------------------------------------------------------------------------
# Use an in-memory SQLite database for fast unit tests
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Schema structure tests
# ---------------------------------------------------------------------------

class TestTransactionModel:
    """Verify Transaction table schema and basic CRUD."""

    def test_table_name(self):
        assert Transaction.__tablename__ == "transactions"

    def test_has_required_columns(self):
        cols = {c.name for c in Transaction.__table__.columns}
        expected = {
            "id", "card_id", "amount", "merchant_id", "merchant_category",
            "timestamp", "composite_score", "decision", "processing_time_ms",
            "created_at",
        }
        assert expected.issubset(cols)

    def test_primary_key_is_id(self):
        pk_cols = [c.name for c in Transaction.__table__.primary_key.columns]
        assert pk_cols == ["id"]

    def test_indexes_exist(self):
        index_cols = set()
        for idx in Transaction.__table__.indexes:
            for col in idx.columns:
                index_cols.add(col.name)
        assert "card_id" in index_cols
        assert "timestamp" in index_cols
        assert "decision" in index_cols

    def test_insert_and_read(self, session):
        txn_id = str(uuid.uuid4())
        txn = Transaction(
            id=txn_id,
            card_id="card_001",
            amount=125.50,
            merchant_id="merch_abc",
            merchant_category="retail",
            timestamp=1700000000.0,
            composite_score=0.42,
            decision="approve",
            processing_time_ms=12.5,
        )
        session.add(txn)
        session.commit()

        result = session.get(Transaction, txn_id)
        assert result is not None
        assert result.card_id == "card_001"
        assert result.amount == 125.50
        assert result.decision == "approve"
        assert result.composite_score == 0.42
        assert result.processing_time_ms == 12.5

    def test_created_at_has_default(self, session):
        txn_id = str(uuid.uuid4())
        txn = Transaction(
            id=txn_id,
            card_id="card_002",
            amount=10.0,
            merchant_id="merch_xyz",
            timestamp=1700000000.0,
            composite_score=0.1,
            decision="approve",
            processing_time_ms=5.0,
        )
        session.add(txn)
        session.commit()
        result = session.get(Transaction, txn_id)
        assert result.created_at is not None


class TestScoringLogModel:
    """Verify ScoringLog table schema and FK relationship."""

    def test_table_name(self):
        assert ScoringLog.__tablename__ == "scoring_logs"

    def test_has_required_columns(self):
        cols = {c.name for c in ScoringLog.__table__.columns}
        expected = {
            "id", "transaction_id", "tier_name", "triggered", "action",
            "latency_ms",
        }
        assert expected.issubset(cols)

    def test_transaction_id_is_foreign_key(self):
        col = ScoringLog.__table__.c.transaction_id
        fk_targets = [fk.target_fullname for fk in col.foreign_keys]
        assert "transactions.id" in fk_targets

    def test_insert_scoring_log(self, session):
        txn_id = str(uuid.uuid4())
        txn = Transaction(
            id=txn_id, card_id="card_003", amount=50.0,
            merchant_id="m1", timestamp=1700000000.0,
            composite_score=0.2, decision="approve", processing_time_ms=8.0,
        )
        session.add(txn)
        session.flush()

        log = ScoringLog(
            transaction_id=txn_id,
            tier_name="hard_rules",
            triggered=False,
            action="pass",
            latency_ms=0.5,
        )
        session.add(log)
        session.commit()

        result = session.query(ScoringLog).filter_by(transaction_id=txn_id).first()
        assert result.tier_name == "hard_rules"
        assert result.triggered is False
        assert result.action == "pass"
        assert result.latency_ms == 0.5


class TestChargebackModel:
    """Verify Chargeback table schema and FK relationship."""

    def test_table_name(self):
        assert Chargeback.__tablename__ == "chargebacks"

    def test_has_required_columns(self):
        cols = {c.name for c in Chargeback.__table__.columns}
        expected = {
            "id", "transaction_id", "reported_at", "confirmed_at", "is_fraud",
        }
        assert expected.issubset(cols)

    def test_transaction_id_is_foreign_key(self):
        col = Chargeback.__table__.c.transaction_id
        fk_targets = [fk.target_fullname for fk in col.foreign_keys]
        assert "transactions.id" in fk_targets

    def test_insert_chargeback(self, session):
        txn_id = str(uuid.uuid4())
        txn = Transaction(
            id=txn_id, card_id="card_004", amount=999.0,
            merchant_id="m2", timestamp=1700000000.0,
            composite_score=0.85, decision="decline", processing_time_ms=3.0,
        )
        session.add(txn)
        session.flush()

        now = datetime.now(timezone.utc)
        cb = Chargeback(
            transaction_id=txn_id,
            reported_at=now,
            is_fraud=True,
        )
        session.add(cb)
        session.commit()

        result = session.query(Chargeback).filter_by(transaction_id=txn_id).first()
        assert result.is_fraud is True
        # SQLite strips tzinfo; compare without timezone
        assert result.reported_at.replace(tzinfo=None) == now.replace(tzinfo=None)
        assert result.confirmed_at is None  # nullable


class TestModelPromotionModel:
    """Verify ModelPromotion table schema and basic CRUD."""

    def test_table_name(self):
        from src.database.models import ModelPromotion
        assert ModelPromotion.__tablename__ == "model_promotions"

    def test_has_required_columns(self):
        from src.database.models import ModelPromotion
        cols = {c.name for c in ModelPromotion.__table__.columns}
        expected = {
            "id", "model_name", "from_version", "to_version",
            "old_auc_pr", "new_auc_pr", "improvement", "promoted",
            "promoted_at",
        }
        assert expected.issubset(cols)

    def test_insert_promotion(self, session):
        from src.database.models import ModelPromotion
        now = datetime.now(timezone.utc)
        promo = ModelPromotion(
            model_name="fraud-xgboost",
            from_version=2,
            to_version=3,
            old_auc_pr=0.85,
            new_auc_pr=0.87,
            improvement=0.02,
            promoted=True,
            promoted_at=now,
        )
        session.add(promo)
        session.commit()

        result = session.query(ModelPromotion).filter_by(model_name="fraud-xgboost").first()
        assert result is not None
        assert result.to_version == 3
        assert result.promoted is True
        assert result.improvement == 0.02

    def test_nullable_from_version(self, session):
        from src.database.models import ModelPromotion
        promo = ModelPromotion(
            model_name="fraud-iforest",
            from_version=None,
            to_version=1,
            old_auc_pr=None,
            new_auc_pr=0.80,
            improvement=0.0,
            promoted=True,
            promoted_at=datetime.now(timezone.utc),
        )
        session.add(promo)
        session.commit()
        result = session.query(ModelPromotion).filter_by(model_name="fraud-iforest").first()
        assert result.from_version is None
        assert result.old_auc_pr is None


class TestTableCreation:
    """Verify all tables are created from Base metadata."""

    def test_all_tables_exist(self, engine):
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        assert "transactions" in table_names
        assert "scoring_logs" in table_names
        assert "chargebacks" in table_names
        assert "model_promotions" in table_names
