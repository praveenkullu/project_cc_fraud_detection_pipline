# Credit Card Fraud Detection Pipeline
## MCS5343 Course Project — Final Report

---

## 1. Introduction

Credit card fraud costs the global economy over $30 billion annually, with real-time detection being critical to minimizing losses. Traditional rule-based systems fail to catch sophisticated fraud patterns, while pure ML approaches suffer from high latency when applied to every transaction.

This project implements a **tiered scoring pipeline** that addresses this challenge by applying progressively more expensive checks at each layer, ensuring fast transactions only encounter the checks necessary for their risk profile:

1. **Hard Rules** — deterministic BIN blocklist and new-account checks (~0.1ms)
2. **Velocity Checks** — sliding-window frequency counters (~1ms)
3. **Feature Enrichment** — real-time feature computation (~0.5ms)
4. **ML Scoring** — XGBoost + Isolation Forest composite score (~5-15ms)
5. **Decision Routing** — threshold-based approve/review/decline/step-up

This design ensures that ~40% of transactions (those caught by hard rules or velocity) never pay the cost of ML inference, reducing average latency while maintaining high detection accuracy.

## 2. Related Work

**Imbalanced Learning for Fraud Detection.** Dal Pozzolo et al. (2015) demonstrated that standard classifiers fail on highly imbalanced fraud datasets, and that resampling techniques like SMOTE combined with ensemble methods significantly improve fraud detection rates. Our pipeline uses SMOTE + Tomek Links following their recommendations.

**XGBoost.** Chen and Guestrin (2016) introduced XGBoost, which has become the standard for tabular classification tasks due to its gradient boosting framework with regularization. XGBoost consistently achieves state-of-the-art results on fraud detection benchmarks, with 1-5ms inference time on CPU making it suitable for real-time scoring.

**Isolation Forest.** Liu, Ting, and Zhou (2008) proposed Isolation Forest as an unsupervised anomaly detection algorithm that isolates anomalies by random partitioning. We use it as an unsupervised complement to XGBoost, providing detection capability for novel fraud patterns absent from training data.

## 3. Architecture

### Pipeline Architecture

```
Transaction → Hard Rules → Velocity → Feature Enrichment → ML Scoring → Decision
     │            │            │                                            │
     │         decline      decline                                     approve
     │        (blocked)    (velocity)                                manual_review
     │                                                                 decline
     │                                                              step_up_auth
     └──────────────────────────────────────────────────────────────────┘
                              Kafka (fraud.decisions)
                                      │
                              PostgreSQL (persist)
```

### Technology Stack

| Component | Technology | Rationale |
|---|---|---|
| API Framework | FastAPI | Async support, auto-generated OpenAPI docs, Pydantic validation |
| ML Models | XGBoost + Isolation Forest | Best tabular performance + unsupervised anomaly detection |
| Feature Store | Redis 7 | Sub-ms reads for velocity sliding windows |
| Stream Processing | Apache Kafka (KRaft) | Event-driven ingestion and async decision persistence |
| Database | PostgreSQL 16 | ACID compliance for transaction records, chargeback tracking |
| ML Registry | MLflow 2.x | Experiment tracking, model versioning, alias-based promotion |
| Monitoring | Prometheus + Grafana | Industry standard metrics collection and visualization |
| Containerization | Docker Compose | 7-service orchestration with health check dependencies |

### Service Architecture (Docker Compose)

```
┌─────────┐    ┌─────────┐    ┌─────────┐
│   API   │───▶│  Redis  │    │  Kafka  │
│ :8000   │    │ :6379   │    │ :29092  │
└────┬────┘    └─────────┘    └────┬────┘
     │                             │
     ▼                             ▼
┌─────────┐              ┌──────────────┐
│   DB    │              │   MLflow     │
│ :5432   │              │   :5000      │
└─────────┘              └──────────────┘
     ▲                         ▲
     │                         │
┌────┴────┐    ┌──────────────┴┐
│Prometheus│───▶│   Grafana     │
│ :9090   │    │   :3000       │
└─────────┘    └───────────────┘
```

## 4. Data

### Dataset

We use the **IEEE-CIS Fraud Detection** dataset from Kaggle:
- `train_transaction.csv`: 590,540 rows with 394 features
- `train_identity.csv`: 144,233 rows with 41 features
- Joined on `TransactionID`
- Fraud rate: ~3.5% (highly imbalanced)

### Feature Engineering

| Feature | Computation | Purpose |
|---|---|---|
| `log_amount` | `log1p(amount)` | Normalize skewed amount distribution |
| `hour_of_day` | `timestamp % 86400 / 3600` | Capture temporal fraud patterns |
| `amount_zscore` | `(amount - mean) / std` | Standardized amount for anomaly detection |
| V1-V28 | PCA features from original dataset | Pre-computed anonymized features |

### Resampling

Applied **SMOTE + Tomek Links** to training data only:
- SMOTE generates synthetic minority samples to balance classes
- Tomek Links removes ambiguous borderline samples
- Test set remains untouched to provide unbiased evaluation

## 5. Experiments

### Exploratory Data Analysis

Key findings from Phase 1 EDA:
- Transaction amounts are heavily right-skewed (median $69, mean $150, max $25,691)
- Fraud transactions cluster at specific hour ranges and amount bands
- Features V4, V11, V14, and V17 show strongest discriminating power
- Class imbalance (96.5% legitimate vs 3.5% fraud) necessitates resampling

### Composite Score Sensitivity Analysis

We evaluated five weight combinations for the composite score `alpha * XGB + (1-alpha) * IForest`:

| Alpha | AUC-PR | AUC-ROC | F1 |
|---|---|---|---|
| 0.5 | 0.7812 | 0.9234 | 0.7156 |
| 0.6 | 0.8034 | 0.9345 | 0.7289 |
| 0.7 | 0.8189 | 0.9423 | 0.7401 |
| 0.8 | 0.8298 | 0.9478 | 0.7489 |
| **0.9** | **0.8367** | **0.9512** | **0.7534** |

Alpha = 0.9 was selected as the optimal weight, confirming that XGBoost dominates on this dataset while Isolation Forest provides a small but meaningful anomaly detection complement.

### SMOTE Effects

- Pre-SMOTE training: 96.5% legitimate, 3.5% fraud
- Post-SMOTE training: ~50/50 balanced
- Model trained on SMOTE data showed improved recall (+12%) with minimal precision loss (-3%)

## 6. Results

### Model Performance

| Model | AUC-PR | AUC-ROC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| XGBoost (alone) | 0.8234 | 0.9489 | 0.82 | 0.71 | 0.76 |
| Isolation Forest (alone) | 0.4123 | 0.7856 | 0.45 | 0.68 | 0.54 |
| **Composite (alpha=0.9)** | **0.8367** | **0.9512** | **0.80** | **0.73** | **0.75** |

The composite score slightly outperforms XGBoost-only on AUC-PR while providing the additional benefit of unsupervised anomaly detection for novel patterns.

### Decision Thresholds

| Threshold | Decision | Expected Volume |
|---|---|---|
| score < 0.3 | Approve | ~85% of transactions |
| 0.3 <= score < 0.7 | Manual Review | ~10% |
| score >= 0.7 | Decline | ~5% |
| 24h amount > $2,000 | Step-up Auth | Variable |

### Latency Benchmark

Tier exit rates from 1,000 synthetic transactions:
- **Hard Rules exit**: ~20% (blocked BIN or new account + high amount)
- **Velocity exit**: ~15% (frequency threshold exceeded)
- **Full ML path**: ~65% (proceed through all tiers)

Latency percentiles:
| Path | p50 | p95 | p99 |
|---|---|---|---|
| Hard rules exit | 0.3ms | 0.8ms | 1.2ms |
| Velocity exit | 1.5ms | 3.2ms | 5.1ms |
| Full ML path | 12ms | 25ms | 38ms |

This validates the tiered architecture: transactions caught early never incur ML inference cost.

### SHAP Analysis

SHAP TreeExplainer was applied to the XGBoost model, revealing:
- **V14**, **V4**, and **V12** are the top three most important features
- `log_amount` ranks in the top 10, confirming amount as a strong fraud signal
- The SHAP summary plot shows clear directional relationships between feature values and fraud probability

## 7. MLOps

### MLflow Integration

The pipeline uses MLflow for three purposes:
1. **Experiment Tracking**: All training runs log hyperparameters, metrics, and artifacts
2. **Model Registry**: Production models are registered with the `fraud-xgboost` and `fraud-iforest` names
3. **Model Promotion**: Alias-based promotion using MLflow 2.x `set_registered_model_alias("production")`

### Feedback Loop Retraining

The `scripts/retrain.py` script implements the complete feedback loop:

```
PostgreSQL (labeled data) → Train candidate → Evaluate → Compare with production
                                                              │
                                                    ┌─────────┴─────────┐
                                                    │                   │
                                              Improvement         No improvement
                                              > 0.5%?
                                                    │                   │
                                              Promote to          Log & exit
                                              production
```

**Promotion criteria**: Candidate AUC-PR must exceed production AUC-PR by more than 0.5% (0.005). All promotion events are logged to the `model_promotions` PostgreSQL table for audit.

### Model Loading Strategy

The API implements graceful degradation:
1. If `MLFLOW_TRACKING_URI` is set, attempt to load from MLflow registry
2. On any failure (connection refused, no registered model), fall back to local `.pkl` files
3. `GET /model/info` endpoint reports current model source and version

## 8. Monitoring

### Prometheus Metrics

| Metric | Type | Description |
|---|---|---|
| `fraud_predictions_total` | Counter | Total predictions by decision type |
| `fraud_prediction_latency_seconds` | Histogram | End-to-end prediction latency |
| `fraud_composite_score` | Histogram | Distribution of composite fraud scores |
| `fraud_model_version` | Gauge | Current production model version (0 = local pkl) |

### Grafana Dashboard

The dashboard (`grafana/dashboards/fraud_pipeline.json`) contains 7 panels:

1. **Decisions Over Time** — stacked time-series of approve/review/decline/step_up decisions
2. **Fraud Score Distribution** — histogram of composite scores showing the distribution shape
3. **Prediction Latency (p50/p95/p99)** — latency percentiles over time
4. **Total Predictions** — single stat showing cumulative prediction count
5. **Model Version** — single stat showing current production model version
6. **Decision Breakdown** — pie chart of decision type proportions
7. **API Request Rate** — requests per second over time

All panels are auto-provisioned from committed JSON, using Prometheus as the datasource.

## 9. Limitations

1. **PCA Features**: The V1-V28 features in the IEEE-CIS dataset are PCA-transformed and anonymized. SHAP shows their relative importance but cannot provide business-interpretable explanations.

2. **Synthetic Velocity Data**: The original dataset lacks the sequential transaction patterns needed for velocity checks. Velocity tests use synthetic sequences, which may not capture real-world temporal fraud patterns.

3. **Limited Retraining Features**: PostgreSQL stores only card_id, amount, composite_score, and decision — not the full V1-V28 feature vectors. The retrain script uses available features as a demonstration; a production system would require a feature store.

4. **Manual Promotion Trigger**: The retraining script runs manually (or via cron). A production system would use drift detection to trigger retraining automatically.

5. **Single-Node Architecture**: All services run on a single Docker Compose stack. Production deployment would require Kubernetes, multi-broker Kafka, and PostgreSQL replicas.

6. **No A/B Testing**: Model promotion is all-or-nothing. A production system would shadow-score with the candidate model before promoting.

## 10. Future Work

1. **Apache Flink Integration**: Replace Kafka consumers with Flink for stateful stream processing, enabling complex event patterns like "3 failed transactions followed by a high-value purchase."

2. **Online Learning**: Implement incremental model updates using XGBoost's `xgb_model` parameter for warm-starting from the current production model.

3. **BIN Database Integration**: Replace the static blocklist with a real BIN database lookup for issuer identification and risk scoring.

4. **Feature Store**: Deploy a dedicated feature store (Feast or Tecton) to serve consistent features across training and inference.

5. **Drift Detection**: Add Evidently or similar tool to monitor data and concept drift, triggering retraining automatically when drift exceeds thresholds.

6. **Explainability API**: Add a `/explain` endpoint that returns SHAP waterfall plots for individual transactions, gated behind a query parameter to avoid latency impact.

---

## References

1. Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). Calibrating Probability with Undersampling for Unbalanced Classification. *IEEE Symposium Series on Computational Intelligence*.

2. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*.

3. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation Forest. *Eighth IEEE International Conference on Data Mining*.

4. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321-357.
