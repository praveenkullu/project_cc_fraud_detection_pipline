"""Tests for Kafka integration — Phase 3 Step 17.

TDD RED: Tests verify topics, producer, and consumer behavior.
Uses mocks for confluent_kafka since we can't run Kafka in unit tests.
"""
import json
from unittest.mock import MagicMock, patch, call

import pytest

from src.kafka.topics import TRANSACTIONS_TOPIC, DECISIONS_TOPIC, create_topics
from src.kafka.producer import DecisionProducer
from src.kafka.consumer import TransactionConsumer, DecisionPersistenceConsumer


# ---------------------------------------------------------------------------
# Topic constants
# ---------------------------------------------------------------------------

class TestTopics:

    def test_transactions_topic_name(self):
        assert TRANSACTIONS_TOPIC == "fraud.transactions"

    def test_decisions_topic_name(self):
        assert DECISIONS_TOPIC == "fraud.decisions"

    def test_create_topics_calls_admin(self):
        mock_admin = MagicMock()
        mock_admin.create_topics.return_value = {
            TRANSACTIONS_TOPIC: MagicMock(result=MagicMock(return_value=None)),
            DECISIONS_TOPIC: MagicMock(result=MagicMock(return_value=None)),
        }
        create_topics(mock_admin)
        mock_admin.create_topics.assert_called_once()
        # Should create exactly 2 topics
        args = mock_admin.create_topics.call_args[0][0]
        assert len(args) == 2


# ---------------------------------------------------------------------------
# Producer tests
# ---------------------------------------------------------------------------

class TestDeliveryCallback:

    def test_callback_no_error(self):
        from src.kafka.producer import _delivery_callback
        # Should not raise
        _delivery_callback(None, MagicMock())

    def test_callback_with_error(self):
        from src.kafka.producer import _delivery_callback
        # Should log error but not raise
        _delivery_callback("some error", MagicMock())


class TestDecisionProducer:

    @pytest.fixture
    def mock_producer(self):
        return MagicMock()

    @pytest.fixture
    def producer(self, mock_producer):
        return DecisionProducer(mock_producer)

    def test_publish_calls_produce(self, producer, mock_producer):
        result = {
            "transaction_id": "txn_001",
            "decision": "approve",
            "composite_score": 0.15,
        }
        producer.publish(result)
        mock_producer.produce.assert_called_once()

    def test_publish_to_decisions_topic(self, producer, mock_producer):
        result = {"transaction_id": "txn_002", "decision": "decline"}
        producer.publish(result)
        args, kwargs = mock_producer.produce.call_args
        assert kwargs.get("topic", args[0] if args else None) == DECISIONS_TOPIC

    def test_publish_serializes_to_json(self, producer, mock_producer):
        result = {"transaction_id": "txn_003", "decision": "approve", "composite_score": 0.2}
        producer.publish(result)
        _, kwargs = mock_producer.produce.call_args
        value = kwargs.get("value", None)
        assert value is not None
        parsed = json.loads(value)
        assert parsed["transaction_id"] == "txn_003"

    def test_publish_uses_transaction_id_as_key(self, producer, mock_producer):
        result = {"transaction_id": "txn_004", "decision": "approve"}
        producer.publish(result)
        _, kwargs = mock_producer.produce.call_args
        assert kwargs.get("key") == "txn_004"

    def test_publish_calls_poll(self, producer, mock_producer):
        result = {"transaction_id": "txn_005", "decision": "approve"}
        producer.publish(result)
        mock_producer.poll.assert_called_once_with(0)

    def test_flush(self, producer, mock_producer):
        producer.flush()
        mock_producer.flush.assert_called_once()


# ---------------------------------------------------------------------------
# TransactionConsumer tests
# ---------------------------------------------------------------------------

class TestTransactionConsumer:

    @pytest.fixture
    def mock_consumer(self):
        return MagicMock()

    @pytest.fixture
    def mock_pipeline_fn(self):
        return MagicMock(return_value={
            "transaction_id": "txn_100",
            "decision": "approve",
            "composite_score": 0.1,
        })

    @pytest.fixture
    def mock_decision_producer(self):
        return MagicMock()

    @pytest.fixture
    def consumer(self, mock_consumer, mock_pipeline_fn, mock_decision_producer):
        return TransactionConsumer(
            consumer=mock_consumer,
            pipeline_fn=mock_pipeline_fn,
            decision_producer=mock_decision_producer,
        )

    def test_subscribes_to_transactions_topic(self, consumer, mock_consumer):
        mock_consumer.subscribe.assert_called_once_with([TRANSACTIONS_TOPIC])

    def test_process_message_calls_pipeline(self, consumer, mock_pipeline_fn):
        msg = MagicMock()
        msg.error.return_value = None
        msg.value.return_value = json.dumps({
            "card_id": "card_1", "amount": 50.0,
            "merchant_id": "m1", "timestamp": 1000.0,
            "ip_address": "1.1.1.1", "account_age_hours": 100.0,
            "bin_number": "400000", "features": {},
        }).encode()
        consumer.process_message(msg)
        mock_pipeline_fn.assert_called_once()

    def test_process_message_publishes_result(self, consumer, mock_decision_producer):
        msg = MagicMock()
        msg.error.return_value = None
        msg.value.return_value = json.dumps({
            "card_id": "card_1", "amount": 50.0,
            "merchant_id": "m1", "timestamp": 1000.0,
            "ip_address": "1.1.1.1", "account_age_hours": 100.0,
            "bin_number": "400000", "features": {},
        }).encode()
        consumer.process_message(msg)
        mock_decision_producer.publish.assert_called_once()

    def test_process_message_skips_errors(self, consumer, mock_pipeline_fn):
        msg = MagicMock()
        msg.error.return_value = MagicMock()  # Has error
        consumer.process_message(msg)
        mock_pipeline_fn.assert_not_called()

    def test_run_processes_max_messages(self, consumer, mock_consumer, mock_pipeline_fn):
        msg = MagicMock()
        msg.error.return_value = None
        msg.value.return_value = json.dumps({
            "card_id": "c1", "amount": 10.0, "merchant_id": "m1",
            "timestamp": 1000.0, "ip_address": "1.1.1.1",
            "account_age_hours": 100.0, "bin_number": "400000", "features": {},
        }).encode()
        mock_consumer.poll.return_value = msg
        consumer.run(max_messages=2)
        assert mock_pipeline_fn.call_count == 2
        mock_consumer.close.assert_called_once()

    def test_run_skips_none_messages(self, consumer, mock_consumer, mock_pipeline_fn):
        msg = MagicMock()
        msg.error.return_value = None
        msg.value.return_value = json.dumps({
            "card_id": "c1", "amount": 10.0, "merchant_id": "m1",
            "timestamp": 1000.0, "ip_address": "1.1.1.1",
            "account_age_hours": 100.0, "bin_number": "400000", "features": {},
        }).encode()
        # Return None first, then a real message
        mock_consumer.poll.side_effect = [None, msg]
        consumer.run(max_messages=1)
        assert mock_pipeline_fn.call_count == 1


# ---------------------------------------------------------------------------
# DecisionPersistenceConsumer tests
# ---------------------------------------------------------------------------

class TestDecisionPersistenceConsumer:

    @pytest.fixture
    def mock_consumer(self):
        return MagicMock()

    @pytest.fixture
    def mock_repo(self):
        repo = MagicMock()
        repo.save_transaction.return_value = "txn_200"
        return repo

    @pytest.fixture
    def consumer(self, mock_consumer, mock_repo):
        return DecisionPersistenceConsumer(
            consumer=mock_consumer,
            repository=mock_repo,
        )

    def test_subscribes_to_decisions_topic(self, consumer, mock_consumer):
        mock_consumer.subscribe.assert_called_once_with([DECISIONS_TOPIC])

    def test_process_message_saves_transaction(self, consumer, mock_repo):
        data = {
            "transaction_id": "txn_200",
            "card_id": "card_1",
            "amount": 99.0,
            "merchant_id": "m1",
            "timestamp": 1000.0,
            "composite_score": 0.5,
            "decision": "manual_review",
            "processing_time_ms": 15.0,
            "tier_results": [],
        }
        msg = MagicMock()
        msg.error.return_value = None
        msg.value.return_value = json.dumps(data).encode()
        consumer.process_message(msg)
        mock_repo.save_transaction.assert_called_once()

    def test_process_message_saves_scoring_log(self, consumer, mock_repo):
        data = {
            "transaction_id": "txn_201",
            "card_id": "card_2",
            "amount": 50.0,
            "merchant_id": "m2",
            "timestamp": 2000.0,
            "composite_score": 0.1,
            "decision": "approve",
            "processing_time_ms": 5.0,
            "tier_results": [
                {"tier": "hard_rules", "triggered": False, "action": "pass", "latency_ms": 0.3},
            ],
        }
        msg = MagicMock()
        msg.error.return_value = None
        msg.value.return_value = json.dumps(data).encode()
        consumer.process_message(msg)
        mock_repo.save_scoring_log.assert_called_once()

    def test_process_message_skips_errors(self, consumer, mock_repo):
        msg = MagicMock()
        msg.error.return_value = MagicMock()
        consumer.process_message(msg)
        mock_repo.save_transaction.assert_not_called()

    def test_run_processes_max_messages(self, consumer, mock_consumer, mock_repo):
        data = {
            "transaction_id": "txn_300", "card_id": "c1", "amount": 50.0,
            "merchant_id": "m1", "timestamp": 1000.0, "composite_score": 0.1,
            "decision": "approve", "processing_time_ms": 5.0, "tier_results": [],
        }
        msg = MagicMock()
        msg.error.return_value = None
        msg.value.return_value = json.dumps(data).encode()
        mock_consumer.poll.return_value = msg
        consumer.run(max_messages=2)
        assert mock_repo.save_transaction.call_count == 2
        mock_consumer.close.assert_called_once()

    def test_run_skips_none_messages(self, consumer, mock_consumer, mock_repo):
        data = {
            "transaction_id": "txn_301", "card_id": "c1", "amount": 50.0,
            "merchant_id": "m1", "timestamp": 1000.0, "composite_score": 0.1,
            "decision": "approve", "processing_time_ms": 5.0, "tier_results": [],
        }
        msg = MagicMock()
        msg.error.return_value = None
        msg.value.return_value = json.dumps(data).encode()
        mock_consumer.poll.side_effect = [None, msg]
        consumer.run(max_messages=1)
        assert mock_repo.save_transaction.call_count == 1
