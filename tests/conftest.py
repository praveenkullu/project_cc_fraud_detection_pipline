"""Shared pytest fixtures for the fraud detection pipeline test suite."""
import pytest


@pytest.fixture
def sample_transaction():
    """Minimal valid transaction dict for API tests."""
    return {
        "card_id": "card_abc123",
        "amount": 99.99,
        "merchant_id": "merch_xyz",
        "timestamp": 86400.0,  # TransactionDT
        "ip_address": "1.2.3.4",
        "account_age_hours": 720.0,  # 30 days
        "bin_number": "400000",
        "features": {},  # empty — model will fill missing with 0
    }


@pytest.fixture
def blocked_bin_transaction(sample_transaction):
    tx = sample_transaction.copy()
    tx["bin_number"] = "999999"  # in BLOCKED_BINS
    return tx


@pytest.fixture
def new_account_high_amount_transaction(sample_transaction):
    tx = sample_transaction.copy()
    tx["amount"] = 750.0
    tx["account_age_hours"] = 12.0
    return tx


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from src.app import app

    with TestClient(app) as c:
        yield c
