-- Phase 3 Step 14: Initial schema for PostgreSQL 16
-- Run: psql -U fraud_user -d fraud_db -f migrations/001_initial.sql

CREATE TABLE IF NOT EXISTS transactions (
    id              VARCHAR PRIMARY KEY,
    card_id         VARCHAR NOT NULL,
    amount          DOUBLE PRECISION NOT NULL,
    merchant_id     VARCHAR NOT NULL,
    merchant_category VARCHAR,
    "timestamp"     DOUBLE PRECISION NOT NULL,
    composite_score DOUBLE PRECISION NOT NULL,
    decision        VARCHAR NOT NULL,
    processing_time_ms DOUBLE PRECISION NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_transactions_card_id ON transactions (card_id);
CREATE INDEX IF NOT EXISTS ix_transactions_timestamp ON transactions ("timestamp");
CREATE INDEX IF NOT EXISTS ix_transactions_decision ON transactions (decision);

CREATE TABLE IF NOT EXISTS scoring_logs (
    id              SERIAL PRIMARY KEY,
    transaction_id  VARCHAR NOT NULL REFERENCES transactions(id),
    tier_name       VARCHAR NOT NULL,
    triggered       BOOLEAN NOT NULL,
    action          VARCHAR NOT NULL,
    latency_ms      DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS chargebacks (
    id              SERIAL PRIMARY KEY,
    transaction_id  VARCHAR NOT NULL REFERENCES transactions(id),
    reported_at     TIMESTAMPTZ NOT NULL,
    confirmed_at    TIMESTAMPTZ,
    is_fraud        BOOLEAN NOT NULL
);
