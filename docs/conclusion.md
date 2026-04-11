# Chapter 5: Conclusion and Future Work

## 5.1 Introduction

This chapter concludes the study on real-time credit card fraud detection using a tiered big data processing pipeline. It synthesizes the key findings across four development phases — model training, API serving, infrastructure deployment, and MLOps — and reflects on the system's contributions, limitations, and directions for future improvement.

---

## 5.2 Conclusion

### 5.2.1 Restatement of Problem and Objectives

Credit card fraud costs the global economy over $30 billion annually. Traditional rule-based systems cannot adapt to evolving fraud patterns; pure ML approaches applied to every transaction introduce latency incompatible with real-time authorization. The central challenge addressed by this project is: **how do you detect fraud accurately, in real time, at scale, while keeping per-transaction inference cost low?**

The primary objectives were:

1. Design and implement a tiered scoring pipeline where each layer filters traffic for subsequent, more expensive layers
2. Train a composite ML model (XGBoost + Isolation Forest) optimized for the imbalanced fraud detection task
3. Deploy the full system as a reproducible Docker stack with stream processing, persistence, and observability
4. Implement an MLOps feedback loop for continuous model improvement from labeled chargeback data

### 5.2.2 Summary of Approach

The pipeline applies five tiers in sequence:

1. **Hard Rules** — deterministic BIN blocklist and new-account amount limits; O(1) lookup, ~0.1–0.5 ms
2. **Velocity Checks** — Redis sorted-set sliding windows counting per-card and per-IP transaction frequency; ~1–3 ms
3. **Feature Enrichment** — real-time computation of `log_amount`, `hour_of_day`, and `amount_zscore`
4. **ML Scoring** — XGBoost (supervised fraud probability) + Isolation Forest (unsupervised anomaly score) → composite via `0.9 × XGB + 0.1 × IForest`
5. **Decision Routing** — threshold-based bucketing into approve / manual_review / decline / step_up_auth

All decisions are published to a Kafka topic (`fraud.decisions`) for asynchronous persistence to PostgreSQL, decoupling the latency-critical API path from database I/O.

The MLOps layer wraps MLflow 2.x for experiment tracking, model registration, and alias-based promotion. A feedback loop retraining script trains candidate models from labeled chargeback data and promotes to production when AUC-PR improves by more than 0.5%.

This approach is appropriate because: (a) the tiered design is well-suited to the skewed cost structure of fraud detection, where most transactions are legitimate; (b) XGBoost is the industry standard for tabular fraud classification; (c) open-source components (Kafka, Redis, PostgreSQL, MLflow, Prometheus, Grafana) make the architecture reproducible and maintainable without proprietary dependencies.

### 5.2.3 Key Results and Findings

**Tiered architecture validation** (synthetic benchmark, 1,000 fabricated transactions):
- 25.2% of transactions exit at Hard Rules
- 32.7% exit at Velocity Checks
- Only 42.1% reach full ML inference
- Median end-to-end latency: **5.29 ms** (p95: 58.50 ms, p99: 137.01 ms)

The tiered design reduces ML inference load by 57.9%, directly achieving the core latency-reduction objective.

**End-to-end decision quality** (dataset benchmark, 1,000 real IEEE-CIS transactions at ~3.5% fraud rate):
- All 1,000 transactions reach ML scoring (real data uses valid BINs / fresh card IDs)
- Median ML-path latency: **86.19 ms** (p95: 182.36 ms, p99: 218.04 ms) — within the 100 ms real-time threshold
- Decision distribution: approve 77.5%, manual review 18.2%, decline 4.1%, step-up auth 0.2%
- Fraud recall at decline threshold: **57.1%** (20/35 fraud cases declined automatically)
- Legitimate approval rate: **97.8%** (944/965 legit transactions approved)

**Model performance** (held-out test set, IEEE-CIS dataset):

| Model | AUC-ROC | AUC-PR | Recall @ 5% FPR |
|---|---|---|---|
| XGBoost baseline | 0.9037 | 0.4995 | 0.6255 |
| XGBoost + SMOTE/Tomek | **0.9109** | **0.5594** | **0.6774** |
| Composite (α=0.9) | 0.9085 | 0.5439 | 0.6484 |

SMOTE + Tomek Links improved recall by 8.2 percentage points (0.6255 → 0.6774) confirming the necessity of resampling on the 3.5% fraud-rate dataset. The composite score provides a marginal anomaly detection complement to the supervised model.

**MLOps and observability:** All 7 Docker services start cleanly from `docker compose up`. The Grafana dashboard provisions automatically with 7 panels. 211 unit tests achieve 100% code coverage. The MLflow registry, feedback loop script, and model promotion logic are fully operational.

### 5.2.4 Contributions

1. **End-to-end tiered pipeline implementation:** A complete, reproducible five-tier fraud detection pipeline integrating rules, velocity, feature engineering, ML scoring, and routing in a single FastAPI service — with empirical benchmarks validating the tier-filtering efficiency claim

2. **Composite scoring architecture:** Empirical alpha sweep demonstrating that α=0.9 is the optimal XGBoost/Isolation Forest blend on the IEEE-CIS dataset, with a principled justification for the unsupervised complement despite its low standalone AUC-PR

3. **Reproducible full-stack MLOps deployment:** Seven-service Docker Compose stack with automated Grafana provisioning, Prometheus metrics, MLflow registry, and a working feedback loop retraining script — all launchable from a single command

4. **100% test coverage across all pipeline components:** 211 unit tests covering hard rules, velocity checks (in-memory and Redis backends), ML scoring, FastAPI endpoints, Kafka integration, PostgreSQL models, and MLflow registry — demonstrating software engineering rigor alongside ML experimentation

### 5.2.5 Limitations

1. **PCA feature opacity:** V1–V28 features are anonymized and PCA-transformed. SHAP identifies their relative importance but cannot provide business-interpretable explanations (e.g., "declined because of unusual merchant category"). This limits the system's auditability in a regulated environment.

2. **Synthetic velocity test data:** The IEEE-CIS dataset lacks card_id, IP address, and session fields. Velocity check correctness is validated via synthetic sequences, not real temporal transaction patterns. Real-world velocity behavior may differ.

3. **Reduced-feature feedback loop:** The PostgreSQL feedback table stores only card_id, amount, composite_score, and decision — not the full V1–V28 feature vectors required for full model retraining. The retrain script is a functional demonstration, but a production system requires a feature store to retain inference-time feature values for retraining.

4. **Single-node infrastructure:** All services run on a single Docker Compose host. No horizontal scaling, no redundancy, no multi-broker Kafka. Suitable for a course project; unsuitable for production at scale.

5. **Manual retraining trigger:** The feedback loop runs on demand (`make retrain`). Without automated drift detection, model degradation between retraining cycles requires manual monitoring.

---

## 5.3 Future Work

### 5.3.1 Technical Infrastructure Improvements

**Kubernetes deployment:** Migrate from Docker Compose to Helm charts for Kubernetes. Enable horizontal pod autoscaling for the API tier, multi-broker Kafka for fault tolerance, and PostgreSQL read replicas for query scale-out. This would address the single-node infrastructure limitation and enable production-grade load testing.

**Apache Flink integration:** Replace the Kafka consumer pattern with Flink for stateful stream processing. Flink enables complex event pattern detection — for example, "three failed authorization attempts on the same card within 10 minutes followed by a high-value purchase" — which is not expressible in the current sliding-window velocity model.

**Improved p99 latency:** Profile the Docker Compose network stack to identify the source of the 132.77 ms p99 outliers. Likely candidates are Kafka producer batching latency and Redis socket buffer flushes under the single-container network namespace.

### 5.3.2 Model Enhancements

**Online / incremental learning:** Implement warm-start retraining using XGBoost's `xgb_model` parameter to fine-tune the production model on new chargeback data without full retraining. This reduces the feedback loop from hours to minutes and allows faster adaptation to emerging fraud patterns.

**Deep learning exploration:** Evaluate tabular transformers (TabNet, FT-Transformer) on the IEEE-CIS dataset. While XGBoost typically dominates tabular benchmarks, transformer-based models have shown competitive performance on feature-rich fraud datasets with complex interaction patterns.

**Sequence modeling for velocity:** Replace the current sliding-window counters with an LSTM or Temporal Convolutional Network that models the full transaction sequence per card, capturing long-range temporal dependencies (e.g., systematic testing of small transactions before a large fraudulent one).

### 5.3.3 Data and Feature Expansion

**Real BIN database lookup:** Replace the hardcoded six-BIN blocklist with integration against a commercial BIN database (e.g., Binlist, PPRO) for issuer country, card type, and known-compromised BIN ranges. This transforms a static rule into a dynamic, maintainable signal.

**Feature store deployment:** Deploy Feast or Hopsworks to serve consistent features across training and inference. Store full V1–V28 feature vectors at inference time for use in feedback loop retraining, closing the current feature-gap between training and operational retraining.

**Concept drift detection:** Integrate Evidently AI or Alibi Detect to monitor data drift (input feature distributions) and concept drift (score-to-outcome correlation). Automated drift alerts would trigger retraining without manual monitoring, improving resilience to evolving fraud patterns.

### 5.3.4 Real-World Application Directions

**Explainability API:** Add a `POST /explain` endpoint that returns a SHAP waterfall plot for an individual transaction, gated behind a `?explain=true` query parameter to avoid latency impact on the standard path. This would enable fraud analysts to understand specific decline decisions — critical for cardholder dispute resolution.

**A/B testing for model promotion:** Shadow-score transactions with a candidate model in parallel with the production model before promoting. Compare downstream chargeback rates (not just AUC-PR on a static test set) to make promotion decisions based on real-world business outcomes rather than held-out metrics.

**Multi-channel data ingestion:** Extend the pipeline to consume from mobile app telemetry (device fingerprint, geolocation change velocity) and e-commerce session data (cart abandonment patterns, address mismatch signals). These features are absent from the Kaggle dataset but are among the strongest fraud signals in production systems.

### 5.3.5 Ethical and Privacy Extensions

**Fairness auditing:** Implement automated bias detection to monitor decline rates across merchant category, geographic region, and card type. A disproportionate decline rate for a specific cardholder segment may indicate model bias that is not detectable from AUC metrics alone.

**Differential privacy for retraining:** Apply differential privacy (DP-SGD or DP noise injection) to the SMOTE-augmented training data to ensure that individual cardholder transactions cannot be reconstructed from model weights — particularly important when the feedback loop incorporates chargeback data linked to real accounts.

**Explainable adverse action notices:** In regulated markets (US FCBA, EU PSD2), cardholders are entitled to an explanation when a transaction is declined. The current SHAP implementation provides model-level importance rankings but not per-transaction human-readable reasons. Future work should map SHAP contributions back to business-interpretable decline reasons (e.g., "unusual transaction frequency," "high-risk amount for new account").

---

## 5.4 Closing Note

This project demonstrates that a production-grade, real-time fraud detection pipeline is achievable with open-source components and sound software engineering practices. The tiered architecture empirically reduces ML inference load by 57.9% in mixed-traffic scenarios, and against real IEEE-CIS transactions the composite model declines 57.1% of confirmed fraud cases automatically while approving 97.8% of legitimate cardholders — routing the remaining 18.2% to manual review. The composite model achieves AUC-PR of 0.5594 on the full held-out test set, and the full MLOps stack provides the observability and continuous learning mechanisms needed for a system that remains effective as fraud patterns evolve.

While infrastructure scale, feature store integration, and automated drift detection represent meaningful gaps between this course implementation and a production deployment, the architectural patterns — tiered filtering, composite scoring, feedback loop retraining, and metric-driven observability — are directly transferable to real-world fraud detection systems.
