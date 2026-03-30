"""Database repository for the fraud detection pipeline.

Phase 3 Step 15 — synchronous SQLAlchemy repository for persisting
scoring results and querying labeled transactions.
"""
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.models import Chargeback, ModelPromotion, ScoringLog, Transaction


class TransactionRepository:
    """Persistence layer for fraud scoring results."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def save_transaction(self, data: dict[str, Any]) -> str:
        """Persist a scoring result and return the transaction ID."""
        txn = Transaction(
            id=data["transaction_id"],
            card_id=data["card_id"],
            amount=data["amount"],
            merchant_id=data["merchant_id"],
            merchant_category=data.get("merchant_category"),
            timestamp=data["timestamp"],
            composite_score=data["composite_score"],
            decision=data["decision"],
            processing_time_ms=data["processing_time_ms"],
        )
        self._session.add(txn)
        self._session.commit()
        return txn.id

    def save_scoring_log(
        self, transaction_id: str, tier_results: list[dict[str, Any]]
    ) -> None:
        """Persist per-tier scoring log entries."""
        for tier in tier_results:
            log = ScoringLog(
                transaction_id=transaction_id,
                tier_name=tier.get("tier", tier.get("tier_name", "")),
                triggered=tier.get("triggered", True),
                action=tier.get("action", ""),
                latency_ms=tier.get("latency_ms", 0.0),
            )
            self._session.add(log)
        self._session.commit()

    def get_labeled_transactions(self, days: int) -> list[dict[str, Any]]:
        """Return transactions from the last N days with fraud labels.

        Joins with chargebacks: transactions with a chargeback are labeled
        as fraud (is_fraud=True), others as legitimate (is_fraud=False).
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        txns = (
            self._session.query(Transaction)
            .filter(Transaction.created_at >= cutoff)
            .all()
        )

        chargeback_txn_ids = set(
            row[0]
            for row in self._session.query(Chargeback.transaction_id)
            .filter(
                Chargeback.transaction_id.in_([t.id for t in txns])
            )
            .all()
        )

        results = []
        for txn in txns:
            results.append({
                "transaction_id": txn.id,
                "card_id": txn.card_id,
                "amount": txn.amount,
                "composite_score": txn.composite_score,
                "decision": txn.decision,
                "is_fraud": txn.id in chargeback_txn_ids,
            })

        return results

    def save_promotion_event(self, data: dict[str, Any]) -> int:
        """Persist a model promotion audit event and return the row ID."""
        promo = ModelPromotion(
            model_name=data["model_name"],
            from_version=data.get("from_version"),
            to_version=data["to_version"],
            old_auc_pr=data.get("old_auc_pr"),
            new_auc_pr=data["new_auc_pr"],
            improvement=data["improvement"],
            promoted=data["promoted"],
        )
        self._session.add(promo)
        self._session.commit()
        return promo.id
