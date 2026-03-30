"""Kafka topic constants and creation utility.

Phase 3 Step 17 — topic name constants and admin helper.
"""
from confluent_kafka.admin import NewTopic

TRANSACTIONS_TOPIC = "fraud.transactions"
DECISIONS_TOPIC = "fraud.decisions"


def create_topics(admin_client, num_partitions: int = 1, replication_factor: int = 1) -> None:
    """Create Kafka topics if they don't exist."""
    topics = [
        NewTopic(TRANSACTIONS_TOPIC, num_partitions=num_partitions, replication_factor=replication_factor),
        NewTopic(DECISIONS_TOPIC, num_partitions=num_partitions, replication_factor=replication_factor),
    ]
    admin_client.create_topics(topics)
