"""Seed synthetic transactions and chargebacks into PostgreSQL for demo.

Generates transactions and marks ~3.5% as fraud via chargeback entries,
enabling the retrain script to demonstrate the full feedback loop.

Usage:
    python -m scripts.seed_chargebacks [--count 500]
"""
import argparse
import logging
import os
import random
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from src.database.models import Base, Chargeback, Transaction

logger = logging.getLogger(__name__)


def seed(count: int = 500):
    """Seed `count` transactions with ~3.5% fraud chargebacks."""
    database_url = os.environ.get(
        "DATABASE_URL", "postgresql://fraud_user:fraud_pass@localhost:5432/fraud_db"
    )
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)

    now = datetime.now(timezone.utc)
    random.seed(42)

    with Session(engine) as session:
        for i in range(count):
            txn_id = str(uuid.uuid4())
            amount = round(random.uniform(5.0, 2000.0), 2)
            score = round(random.uniform(0.0, 1.0), 4)
            if score < 0.3:
                decision = "approve"
            elif score < 0.7:
                decision = "manual_review"
            else:
                decision = "decline"

            txn = Transaction(
                id=txn_id,
                card_id=f"card_{random.randint(1, 100)}",
                amount=amount,
                merchant_id=f"merch_{random.randint(1, 50)}",
                merchant_category=random.choice(["retail", "travel", "food", "electronics"]),
                timestamp=float((now - timedelta(days=random.randint(1, 60))).timestamp()),
                composite_score=score,
                decision=decision,
                processing_time_ms=round(random.uniform(5.0, 50.0), 1),
                created_at=now - timedelta(days=random.randint(1, 60)),
            )
            session.add(txn)

            # ~3.5% fraud rate
            if random.random() < 0.035:
                cb = Chargeback(
                    transaction_id=txn_id,
                    reported_at=now - timedelta(days=random.randint(0, 30)),
                    is_fraud=True,
                )
                session.add(cb)

        session.commit()
        logger.info("Seeded %d transactions", count)


def main():
    parser = argparse.ArgumentParser(description="Seed synthetic chargebacks")
    parser.add_argument("--count", type=int, default=500)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    seed(count=args.count)


if __name__ == "__main__":
    main()
