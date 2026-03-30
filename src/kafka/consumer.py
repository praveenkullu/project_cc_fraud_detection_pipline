"""Kafka consumers for the fraud detection pipeline.

Phase 3 Step 17 — TransactionConsumer scores inbound transactions,
DecisionPersistenceConsumer persists results to PostgreSQL.
"""
import json
import logging
from typing import Any, Callable

from src.kafka.topics import TRANSACTIONS_TOPIC, DECISIONS_TOPIC

logger = logging.getLogger(__name__)


class TransactionConsumer:
    """Reads transactions from Kafka, scores them, publishes decisions."""

    def __init__(self, consumer, pipeline_fn: Callable, decision_producer) -> None:
        self._consumer = consumer
        self._pipeline_fn = pipeline_fn
        self._decision_producer = decision_producer
        self._consumer.subscribe([TRANSACTIONS_TOPIC])

    def process_message(self, msg) -> None:
        """Process a single Kafka message."""
        if msg.error():
            logger.warning("Consumer error: %s", msg.error())
            return

        data = json.loads(msg.value())
        result = self._pipeline_fn(data)
        self._decision_producer.publish(result)

    def run(self, poll_timeout: float = 1.0, max_messages: int = 0) -> None:
        """Poll loop. max_messages=0 means run forever."""
        count = 0
        try:
            while max_messages == 0 or count < max_messages:
                msg = self._consumer.poll(poll_timeout)
                if msg is None:
                    continue
                self.process_message(msg)
                count += 1
        finally:
            self._consumer.close()


class DecisionPersistenceConsumer:
    """Reads decisions from Kafka and persists to PostgreSQL."""

    def __init__(self, consumer, repository) -> None:
        self._consumer = consumer
        self._repository = repository
        self._consumer.subscribe([DECISIONS_TOPIC])

    def process_message(self, msg) -> None:
        """Process a single Kafka message."""
        if msg.error():
            logger.warning("Consumer error: %s", msg.error())
            return

        data = json.loads(msg.value())
        self._repository.save_transaction(data)
        self._repository.save_scoring_log(
            data["transaction_id"],
            data.get("tier_results", []),
        )

    def run(self, poll_timeout: float = 1.0, max_messages: int = 0) -> None:
        """Poll loop. max_messages=0 means run forever."""
        count = 0
        try:
            while max_messages == 0 or count < max_messages:
                msg = self._consumer.poll(poll_timeout)
                if msg is None:
                    continue
                self.process_message(msg)
                count += 1
        finally:
            self._consumer.close()
