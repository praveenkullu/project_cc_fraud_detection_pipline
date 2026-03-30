"""Kafka decision producer for the fraud detection pipeline.

Phase 3 Step 17 — publishes ScoringResult to fraud.decisions topic.
"""
import json
import logging
from typing import Any

from src.kafka.topics import DECISIONS_TOPIC

logger = logging.getLogger(__name__)


def _delivery_callback(err, msg):
    if err is not None:
        logger.error("Kafka delivery failed: %s", err)


class DecisionProducer:
    """Publishes scoring results to Kafka decisions topic."""

    def __init__(self, producer) -> None:
        self._producer = producer

    def publish(self, scoring_result: dict[str, Any]) -> None:
        """Fire-and-forget publish of a scoring result."""
        self._producer.produce(
            topic=DECISIONS_TOPIC,
            key=scoring_result.get("transaction_id", ""),
            value=json.dumps(scoring_result),
            callback=_delivery_callback,
        )
        self._producer.poll(0)

    def flush(self, timeout: float = 5.0) -> None:
        """Flush pending messages."""
        self._producer.flush(timeout)
