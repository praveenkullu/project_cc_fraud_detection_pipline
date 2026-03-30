# Credit Card Fraud Detection Pipeline

**Course:** MCS5343 — Machine Learning Systems
**University:** LTU
**Dataset:** [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data)

## Architecture

```mermaid
graph TD
    A[Transaction] --> B{Hard Rules}
    B -->|Blocklisted BIN / new account limit| C[Decline]
    B -->|Pass| D{Velocity Checks}
    D -->|Redis sliding window exceeded| C
    D -->|Pass| E[Feature Enrichment]
    E --> F[ML Scoring]
    F --> G{Decision Router}
    G -->|score < 0.3| H[Approve]
    G -->|0.3 ≤ score < 0.7| I[Manual Review]
    G -->|score ≥ 0.7| C
    G -->|amt > $2000/24h| J[Step-Up Auth]
    F --> K[Kafka fraud.decisions]
    K --> L[Persistence Worker]
    L --> M[(PostgreSQL)]
```

**ML Scoring:** `composite = 0.9 × XGBoost_prob + 0.1 × IsolationForest_normalized`

## Quick Start

```bash
# 1. Install dependencies
make install

# 2. Download dataset → data/
#    From https://www.kaggle.com/competitions/ieee-fraud-detection/data
#    Place: train_transaction.csv, train_identity.csv in data/

# 3. Run EDA and training notebooks
make notebook

# 4. Run tests
make test

# 5. Start API (Phase 2+)
make serve

# 6. Start full stack (Phase 3+)
make docker-up
```

## Project Structure

```
├── notebooks/               # Phase 1: EDA, feature eng, training, evaluation
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
├── src/                     # Importable Python package
│   ├── app.py               # FastAPI entrypoint + 5-tier pipeline
│   ├── features.py          # Feature engineering utilities
│   ├── scoring.py           # Composite scoring + decision routing
│   ├── hard_rules.py        # Tier 1: BIN blocklist + new account limits
│   ├── velocity.py          # Tier 2: in-memory sliding window counters
│   ├── velocity_redis.py    # Tier 2: Redis-backed sliding window (Phase 3)
│   ├── database/            # Phase 3: PostgreSQL persistence
│   │   ├── models.py        # SQLAlchemy ORM (Transaction, ScoringLog, Chargeback, ModelPromotion)
│   │   └── repository.py    # CRUD operations + labeled transaction queries
│   ├── kafka/               # Phase 3: Event streaming
│   │   ├── topics.py        # Topic name constants + creation utility
│   │   ├── producer.py      # DecisionProducer → fraud.decisions
│   │   └── consumer.py      # TransactionConsumer + DecisionPersistenceConsumer
│   └── mlops/               # Phase 4: MLflow registry wrapper
│       └── registry.py      # register_model, get_production_model, promote_to_production
├── tests/                   # pytest test suite (211 tests, 100% coverage)
├── migrations/              # SQL schema migrations
│   ├── 001_initial.sql      # PostgreSQL 16 initial schema
│   └── 002_model_promotions.sql  # Model promotion audit trail
├── scripts/
│   ├── benchmark.py         # Latency benchmark (1000 txns, p50/p95/p99)
│   ├── retrain.py           # Feedback loop retraining + conditional promotion
│   ├── register_models.py   # One-time: register Phase 1 models in MLflow
│   └── seed_chargebacks.py  # Seed synthetic chargebacks for demo
├── docs/
│   └── report.md            # Final project report (10 sections)
├── docker-compose.yml       # 7-service stack
├── Dockerfile               # Multi-stage Python 3.11-slim
├── prometheus/
│   └── prometheus.yml       # Scrape config (api:8000/metrics every 15s)
├── grafana/
│   ├── provisioning/        # Auto-provisioned datasource + dashboard
│   └── dashboards/
│       └── fraud_pipeline.json
├── models/                  # Saved model artifacts (.pkl)
├── results/                 # Evaluation plots and benchmarks
└── data/                    # Dataset files (gitignored)
```

## Development Phases

| Phase | Status | Deliverable |
|---|---|---|
| 1 | ✓ Complete | Notebooks + trained models + MLflow experiments |
| 2 | ✓ Complete | FastAPI scoring API + 5-tier pipeline + unit tests |
| 3 | ✓ Complete | Docker stack (7 services), Redis velocity, Kafka, PostgreSQL, unit tests |
| 4 | ✓ Complete | MLflow registry, retraining script, Grafana dashboard (7 panels), 211 unit tests, final report |

## API Endpoints (Phase 2+)

The FastAPI scoring service implements three endpoints:

### POST /predict
Main fraud scoring endpoint. Runs the full 5-tier pipeline.

**Request:**
```json
{
  "card_id": "card_12345",
  "amount": 150.00,
  "merchant_id": "merch_999",
  "timestamp": 1711532400.0,
  "ip_address": "192.168.1.1",
  "account_age_hours": 48.0,
  "bin_number": "512345",
  "features": {}
}
```

**Response:**
```json
{
  "transaction_id": "550e8400-e29b-41d4-a716-446655440000",
  "decision": "approve",
  "composite_score": 0.25,
  "xgb_score": 0.20,
  "iforest_score": 0.35,
  "tier_results": [
    {"tier": "hard_rules"},
    {"tier": "velocity", "flags": []},
    {"tier": "ml_scoring", "xgb_score": 0.20, "iforest_score": 0.35, "composite_score": 0.25}
  ],
  "latency_ms": 8.5
}
```

**Decision outcomes:**
- `approve` — score < 0.3
- `manual_review` — 0.3 ≤ score < 0.7
- `decline` — score ≥ 0.7
- `step_up_auth` — 24-hour amount > $2,000

**Early exits:**
- Hard rules match (BIN blocklist, new account high amount) → decline immediately
- Velocity threshold exceeded → decline or flag for review

### GET /health
Liveness probe for container orchestration.

**Response:**
```json
{
  "status": "healthy"
}
```

### GET /metrics
Prometheus metrics endpoint (Content-Type: text/plain; version=0.0.4).

Tracked metrics:
- `fraud_predictions_total{decision=...}` — Counter of predictions by outcome
- `fraud_prediction_latency_seconds` — Histogram of end-to-end latency
- `fraud_composite_score` — Histogram of composite fraud scores
- `fraud_model_version` — Gauge: current production model version (0 = local pkl)

### GET /model/info
Model metadata endpoint (Phase 4).

**Response:**
```json
{
  "model_source": "local_pkl",
  "model_version": null,
  "scoring_config": {
    "best_alpha": 0.9,
    "amount_mean": 150.42,
    "amount_std": 235.67
  }
}
```

## Running the API

**Local development (in-memory velocity):**
```bash
make serve
# Starts uvicorn on http://localhost:8000
```

**Full Docker stack (7 services):**
```bash
make docker-up
# Starts: api, db (PostgreSQL 16), redis, kafka, mlflow, prometheus, grafana
```

| Service | Port | URL |
|---|---|---|
| API | 8000 | http://localhost:8000 |
| PostgreSQL | 5432 | `postgres://fraud_user:fraud_pass@localhost:5432/fraud_db` |
| Redis | 6379 | `redis://localhost:6379` |
| Kafka (external) | 29092 | `localhost:29092` |
| MLflow | 5000 | http://localhost:5000 |
| Prometheus | 9090 | http://localhost:9090 |
| Grafana | 3000 | http://localhost:3000 (admin/admin) |

## Docker Compose Services

The `docker-compose.yml` defines a 7-service stack with health checks and dependency ordering:

```
api → depends_on → db (healthy), redis (healthy), kafka (healthy)
prometheus → depends_on → api (healthy)
grafana → depends_on → prometheus
```

- **api** — FastAPI app (Python 3.11-slim, multi-stage build). Reads `VELOCITY_BACKEND`, `REDIS_URL`, `DATABASE_URL`, `KAFKA_BROKER_URL`.
- **db** — PostgreSQL 16-alpine. Schema auto-applied via `migrations/001_initial.sql` mounted into `/docker-entrypoint-initdb.d/`.
- **redis** — Redis 7-alpine. Used as velocity backend when `VELOCITY_BACKEND=redis`.
- **kafka** — Bitnami Kafka (KRaft mode, no ZooKeeper). Internal listener on `kafka:9092`, external on `localhost:29092`.
- **mlflow** — MLflow tracking server with SQLite backend and file-based artifact store.
- **prometheus** — Scrapes `api:8000/metrics` every 15 seconds.
- **grafana** — Auto-provisioned with Prometheus datasource and pre-built fraud pipeline dashboard (7 panels: decisions over time, fraud score distribution, latency percentiles, total predictions, model version, decision breakdown, API request rate).

## Database Schema

PostgreSQL tables created by `migrations/001_initial.sql` and `002_model_promotions.sql`:

| Table | Purpose | Key Columns |
|---|---|---|
| `transactions` | Scored transactions | `id` (UUID PK), `card_id`, `amount`, `composite_score`, `decision`, `processing_time_ms` |
| `scoring_logs` | Per-tier audit trail | `transaction_id` (FK), `tier_name`, `triggered`, `action`, `latency_ms` |
| `chargebacks` | Fraud labels for retraining | `transaction_id` (FK), `reported_at`, `is_fraud` |
| `model_promotions` | Model promotion audit log | `model_name`, `from_version`, `to_version`, `old_auc_pr`, `new_auc_pr`, `promoted` |

Indexes on `card_id`, `timestamp`, `decision`, and `model_name` for query performance.

## Kafka Topics

| Topic | Direction | Content |
|---|---|---|
| `fraud.transactions` | Inbound | JSON-serialized transaction requests |
| `fraud.decisions` | Outbound | JSON-serialized scoring results |

**Consumers** (run as separate processes):
- `TransactionConsumer` — reads `fraud.transactions`, scores via pipeline, publishes to `fraud.decisions`
- `DecisionPersistenceConsumer` — reads `fraud.decisions`, persists to PostgreSQL

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VELOCITY_BACKEND` | `memory` | Velocity implementation: `memory` or `redis` |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL (when `VELOCITY_BACKEND=redis`) |
| `DATABASE_URL` | — | PostgreSQL connection string |
| `KAFKA_BROKER_URL` | `kafka:9092` | Kafka bootstrap server |
| `MLFLOW_TRACKING_URI` | — | MLflow tracking server URL |

## Testing

```bash
# Unit tests (211 tests, 100% coverage)
make test

# Unit tests only (skip integration)
make test-unit

# Integration tests (requires docker compose up)
make test-integration
```

### Test Coverage by Module

| Module | Tests | Coverage |
|---|---|---|
| `src/app.py` | 32 | 100% |
| `src/velocity.py` | 17 | 100% |
| `src/velocity_redis.py` | 21 | 100% |
| `src/hard_rules.py` | 16 | 100% |
| `src/database/models.py` | 19 | 100% |
| `src/database/repository.py` | 15 | 100% |
| `src/kafka/` | 23 | 100% |
| `src/mlops/registry.py` | 7 | 100% |
| `src/features.py` + `src/scoring.py` | 50 | 100% |
| `scripts/retrain.py` | 4 | (mocked) |
| Integration tests | 7 | (skipped without Docker) |

## Benchmarking

```bash
# Requires running API (local or Docker)
make benchmark

# Custom options
python scripts/benchmark.py --url http://localhost:8000 --n 1000
```

Sends 1,000 transactions (40% normal, 30% velocity-triggering, 20% hard-rule, 10% ML-heavy) and generates:
- `results/latency_benchmark.png` — latency histogram with p50/p95/p99
- `results/tier_exit_rates.txt` — per-tier exit rates and latency statistics

## Available Make Targets

| Command | Description |
|---|---|
| `make install` | Install Python dependencies |
| `make serve` | Start API locally (uvicorn, port 8000) |
| `make test` | Run all unit tests with coverage |
| `make test-unit` | Run unit tests only (skip integration) |
| `make test-integration` | Run integration tests (requires Docker) |
| `make lint` | Run flake8 linter |
| `make docker-up` | Start full Docker Compose stack |
| `make docker-down` | Stop Docker Compose stack |
| `make docker-logs` | Tail Docker Compose logs |
| `make benchmark` | Run latency benchmark |
| `make retrain` | Run feedback loop retraining script |
| `make seed-chargebacks` | Seed synthetic chargebacks into PostgreSQL |
| `make register-models` | Register Phase 1 models in MLflow registry |
| `make migrate` | Apply new migrations to running Docker DB |
| `make notebook` | Start Jupyter notebook server |

## MLOps (Phase 4)

### Model Registry

Models are registered in MLflow with alias-based promotion:
- `fraud-xgboost` — XGBoost fraud probability model
- `fraud-iforest` — Isolation Forest anomaly detection model

The API loads from the registry when `MLFLOW_TRACKING_URI` is set, falling back to local `.pkl` files otherwise.

### Feedback Loop Retraining

```bash
# 1. Seed demo data
make seed-chargebacks

# 2. Register existing models
make register-models

# 3. Run retraining
make retrain
```

The retrain script:
1. Queries PostgreSQL for labeled transactions (chargeback = fraud)
2. Trains a candidate XGBoost model
3. Compares AUC-PR against current production model
4. Promotes if improvement exceeds 0.5% threshold
5. Logs all events to `model_promotions` table

### Model Promotion Workflow

```
Labeled Data → Train Candidate → Evaluate → Compare with Production
                                                    │
                                          ┌─────────┴──────────┐
                                    Improvement > 0.5%    No improvement
                                          │                     │
                                    Promote to prod       Log & skip
```

## Dataset

IEEE-CIS Fraud Detection (Kaggle):
- `train_transaction.csv` — 590,540 transactions, 394 features
- `train_identity.csv` — 144,233 rows, device/network identity
- Joined on `TransactionID` (left join)
- Fraud rate: ~3.5%

Place CSV files in `data/` (gitignored).
