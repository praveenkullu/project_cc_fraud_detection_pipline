-- Phase 4 Step 8: Model promotions audit trail
-- Run: psql -U fraud_user -d fraud_db -f migrations/002_model_promotions.sql

CREATE TABLE IF NOT EXISTS model_promotions (
    id              SERIAL PRIMARY KEY,
    model_name      VARCHAR NOT NULL,
    from_version    INTEGER,
    to_version      INTEGER NOT NULL,
    old_auc_pr      DOUBLE PRECISION,
    new_auc_pr      DOUBLE PRECISION NOT NULL,
    improvement     DOUBLE PRECISION NOT NULL,
    promoted        BOOLEAN NOT NULL,
    promoted_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_model_promotions_model_name ON model_promotions (model_name);
