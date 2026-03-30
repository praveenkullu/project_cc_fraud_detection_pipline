"""SQLAlchemy models for the fraud detection pipeline.

Phase 3 Step 14 — defines Transaction, ScoringLog, and Chargeback tables.
Compatible with both SQLite (testing) and PostgreSQL (production).
"""
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(String, primary_key=True)
    card_id = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    merchant_id = Column(String, nullable=False)
    merchant_category = Column(String, nullable=True)
    timestamp = Column(Float, nullable=False)
    composite_score = Column(Float, nullable=False)
    decision = Column(String, nullable=False)
    processing_time_ms = Column(Float, nullable=False)
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    scoring_logs = relationship("ScoringLog", back_populates="transaction")
    chargebacks = relationship("Chargeback", back_populates="transaction")

    __table_args__ = (
        Index("ix_transactions_card_id", "card_id"),
        Index("ix_transactions_timestamp", "timestamp"),
        Index("ix_transactions_decision", "decision"),
    )


class ScoringLog(Base):
    __tablename__ = "scoring_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(
        String, ForeignKey("transactions.id"), nullable=False
    )
    tier_name = Column(String, nullable=False)
    triggered = Column(Boolean, nullable=False)
    action = Column(String, nullable=False)
    latency_ms = Column(Float, nullable=False)

    transaction = relationship("Transaction", back_populates="scoring_logs")


class Chargeback(Base):
    __tablename__ = "chargebacks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(
        String, ForeignKey("transactions.id"), nullable=False
    )
    reported_at = Column(DateTime, nullable=False)
    confirmed_at = Column(DateTime, nullable=True)
    is_fraud = Column(Boolean, nullable=False)

    transaction = relationship("Transaction", back_populates="chargebacks")


class ModelPromotion(Base):
    __tablename__ = "model_promotions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String, nullable=False)
    from_version = Column(Integer, nullable=True)
    to_version = Column(Integer, nullable=False)
    old_auc_pr = Column(Float, nullable=True)
    new_auc_pr = Column(Float, nullable=False)
    improvement = Column(Float, nullable=False)
    promoted = Column(Boolean, nullable=False)
    promoted_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    __table_args__ = (
        Index("ix_model_promotions_model_name", "model_name"),
    )
