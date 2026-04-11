"""Dataset benchmark for the fraud detection pipeline.

Samples real transactions from the IEEE-CIS training set (X_tr_raw.parquet +
y_tr.parquet), preprocesses them using the saved feature engineering artifacts,
and sends them to POST /predict.  Stratified sampling ensures the natural
~3.5% fraud rate is preserved (or oversampled via --fraud-ratio).

Outputs:
    results/dataset_tier_exit_rates.txt
    results/dataset_latency_benchmark.png

Usage:
    python scripts/benchmark_dataset.py [--url http://localhost:8001] [--n 1000]
    python scripts/benchmark_dataset.py --n 500 --fraud-ratio 0.5
"""
import argparse
import random
import time
from collections import Counter
from pathlib import Path

import httpx
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"


# ---------------------------------------------------------------------------
# Data loading + preprocessing (from scripts/sample.py)
# ---------------------------------------------------------------------------

def load_artifacts():
    feature_cols = joblib.load(MODELS_DIR / "feature_cols.pkl")
    encoders = joblib.load(MODELS_DIR / "label_encoders.pkl")
    imputer = joblib.load(MODELS_DIR / "imputer.pkl")
    return feature_cols, encoders, imputer


def preprocess_row(row: pd.Series, feature_cols, encoders, imputer) -> dict:
    """Apply the same pipeline as sample.py: encode → impute → dict."""
    df = row.to_frame().T.copy()
    df_features = df[feature_cols].copy()

    for col, le in encoders.items():
        if col in df_features.columns:
            val = str(df_features[col].iloc[0])
            try:
                df_features[col] = le.transform([val])[0]
            except (ValueError, KeyError):
                df_features[col] = 0

    imputed = imputer.transform(df_features)
    return dict(zip(feature_cols, imputed[0]))


def build_transactions(n: int, fraud_ratio: float | None) -> list[dict]:
    """Sample n rows from X_tr_raw, stratified by label, and build API payloads."""
    print("Loading training data ...")
    X = pd.read_parquet(DATA_DIR / "X_tr_raw.parquet")
    y = pd.read_parquet(DATA_DIR / "y_tr.parquet")["isFraud"]

    fraud_idx = y[y == 1].index.tolist()
    legit_idx = y[y == 0].index.tolist()

    if fraud_ratio is None:
        # Preserve natural rate (~3.5%)
        n_fraud = max(1, round(n * y.mean()))
    else:
        n_fraud = round(n * fraud_ratio)
    n_legit = n - n_fraud

    sampled_fraud = random.sample(fraud_idx, min(n_fraud, len(fraud_idx)))
    sampled_legit = random.sample(legit_idx, min(n_legit, len(legit_idx)))
    indices = sampled_fraud + sampled_legit
    random.shuffle(indices)

    print(f"Sampled {len(sampled_fraud)} fraud + {len(sampled_legit)} legit = {len(indices)} total")
    print("Preprocessing features ...")

    feature_cols, encoders, imputer = load_artifacts()
    transactions = []
    base_ts = float(int(time.time()))

    for pos, idx in enumerate(indices):
        row = X.loc[idx]
        label = int(y.loc[idx])
        raw_amount = float(row.get("TransactionAmt", 0.0))
        features = preprocess_row(row, feature_cols, encoders, imputer)

        txn = {
            "card_id": f"card_{idx}",
            "amount": raw_amount,
            "merchant_id": f"merch_{random.randint(100, 999)}",
            "timestamp": base_ts + pos,
            "ip_address": "192.168.1.1",
            "account_age_hours": float(random.randint(24, 8760)),
            "bin_number": str(random.randint(400000, 499999)),
            "features": {k: (v.item() if hasattr(v, "item") else v) for k, v in features.items()},
            # Carry ground-truth label for accuracy reporting (not sent to API)
            "_isFraud": label,
        }
        transactions.append(txn)

    return transactions


# ---------------------------------------------------------------------------
# Benchmark runner (mirrors benchmark.py)
# ---------------------------------------------------------------------------

def run_benchmark(api_url: str, transactions: list[dict]):
    latencies = []
    decisions = []
    tier_exits = {"hard_rules": 0, "velocity": 0, "ml_scoring": 0}
    ground_truth = []

    n = len(transactions)
    print(f"Sending {n} transactions to {api_url}/predict ...")
    client = httpx.Client(timeout=30.0)

    for i, txn in enumerate(transactions):
        label = txn.pop("_isFraud")  # strip before sending
        t0 = time.perf_counter()
        try:
            resp = client.post(f"{api_url}/predict", json=txn)
            t1 = time.perf_counter()

            if resp.status_code != 200:
                txn["_isFraud"] = label  # restore for retry safety
                continue

            body = resp.json()
            latency_ms = (t1 - t0) * 1000
            latencies.append(latency_ms)
            decisions.append(body["decision"])
            ground_truth.append(label)

            tier_names = [t.get("tier") for t in body["tier_results"]]
            if "hard_rules" in tier_names and "ml_scoring" not in tier_names:
                tier_exits["hard_rules"] += 1
            elif "velocity" in tier_names and "ml_scoring" not in tier_names:
                tier_exits["velocity"] += 1
            else:
                tier_exits["ml_scoring"] += 1

        except httpx.RequestError as e:
            print(f"  Request {i} failed: {e}")
            txn["_isFraud"] = label

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n} done")

    client.close()
    return latencies, decisions, tier_exits, ground_truth


# ---------------------------------------------------------------------------
# Results output
# ---------------------------------------------------------------------------

def save_results(latencies, decisions, tier_exits, ground_truth):
    RESULTS_DIR.mkdir(exist_ok=True)
    arr = np.array(latencies)

    # Decision accuracy vs ground truth
    # API "decline" maps to predicted fraud; anything else to not-fraud
    n_fraud_actual = sum(ground_truth)
    n_legit_actual = len(ground_truth) - n_fraud_actual

    fraud_declined = sum(
        1 for d, g in zip(decisions, ground_truth) if g == 1 and d == "decline"
    )
    legit_approved = sum(
        1 for d, g in zip(decisions, ground_truth) if g == 0 and d != "decline"
    )

    txt_path = RESULTS_DIR / "dataset_tier_exit_rates.txt"
    with open(txt_path, "w") as f:
        total = sum(tier_exits.values())

        f.write("Dataset Benchmark — Tier Exit Rate Analysis\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total scored transactions: {total}\n")
        f.write(f"  fraud (actual):          {n_fraud_actual}\n")
        f.write(f"  legitimate (actual):     {n_legit_actual}\n\n")

        for tier, count in tier_exits.items():
            pct = (count / total * 100) if total > 0 else 0
            f.write(f"  {tier:20s}: {count:5d} ({pct:5.1f}%)\n")

        f.write(f"\nLatency Statistics (ms)\n")
        f.write("=" * 40 + "\n")
        if len(arr) > 0:
            f.write(f"  p50:  {np.percentile(arr, 50):8.2f}\n")
            f.write(f"  p95:  {np.percentile(arr, 95):8.2f}\n")
            f.write(f"  p99:  {np.percentile(arr, 99):8.2f}\n")
            f.write(f"  mean: {np.mean(arr):8.2f}\n")
            f.write(f"  min:  {np.min(arr):8.2f}\n")
            f.write(f"  max:  {np.max(arr):8.2f}\n")

        f.write(f"\nDecision Distribution\n")
        f.write("=" * 40 + "\n")
        for decision, count in Counter(decisions).most_common():
            pct = count / len(decisions) * 100 if decisions else 0
            f.write(f"  {decision:20s}: {count:5d} ({pct:5.1f}%)\n")

        f.write(f"\nAccuracy vs Ground Truth\n")
        f.write("=" * 40 + "\n")
        if n_fraud_actual > 0:
            recall = fraud_declined / n_fraud_actual * 100
            f.write(f"  Fraud recall (declined):  {fraud_declined}/{n_fraud_actual} ({recall:.1f}%)\n")
        if n_legit_actual > 0:
            tnr = legit_approved / n_legit_actual * 100
            f.write(f"  Legit approval rate:      {legit_approved}/{n_legit_actual} ({tnr:.1f}%)\n")

    print(f"\nResults saved to {txt_path}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Latency histogram
        axes[0].hist(arr, bins=50, edgecolor="black", alpha=0.7)
        axes[0].axvline(np.percentile(arr, 50), color="green", linestyle="--", label="p50")
        axes[0].axvline(np.percentile(arr, 95), color="orange", linestyle="--", label="p95")
        axes[0].axvline(np.percentile(arr, 99), color="red", linestyle="--", label="p99")
        axes[0].set_xlabel("Latency (ms)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("End-to-End Prediction Latency")
        axes[0].legend()

        # Tier exit pie
        axes[1].pie(
            list(tier_exits.values()),
            labels=list(tier_exits.keys()),
            autopct="%1.1f%%",
            startangle=90,
        )
        axes[1].set_title("Tier Exit Rates")

        # Decision distribution bar (split by actual label)
        decision_labels = sorted(set(decisions))
        fraud_counts = [
            sum(1 for d, g in zip(decisions, ground_truth) if d == dl and g == 1)
            for dl in decision_labels
        ]
        legit_counts = [
            sum(1 for d, g in zip(decisions, ground_truth) if d == dl and g == 0)
            for dl in decision_labels
        ]
        x = range(len(decision_labels))
        axes[2].bar(x, legit_counts, label="Legit", alpha=0.8)
        axes[2].bar(x, fraud_counts, bottom=legit_counts, label="Fraud", alpha=0.8)
        axes[2].set_xticks(list(x))
        axes[2].set_xticklabels(decision_labels, rotation=15)
        axes[2].set_ylabel("Count")
        axes[2].set_title("Decision Distribution by Actual Label")
        axes[2].legend()

        plt.tight_layout()
        out_png = RESULTS_DIR / "dataset_latency_benchmark.png"
        plt.savefig(out_png, dpi=150)
        print(f"Plot saved to {out_png}")
    except ImportError:
        print("matplotlib not available — skipping plot")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Dataset-driven fraud pipeline benchmark")
    parser.add_argument("--url", default="http://localhost:8001", help="API base URL")
    parser.add_argument("--n", type=int, default=1000, help="Number of transactions to sample")
    parser.add_argument(
        "--fraud-ratio",
        type=float,
        default=None,
        help="Fraction of sampled rows that are fraud (default: natural ~3.5%%)",
    )
    args = parser.parse_args()

    transactions = build_transactions(args.n, args.fraud_ratio)
    latencies, decisions, tier_exits, ground_truth = run_benchmark(args.url, transactions)

    if not latencies:
        print("No successful responses. Is the API running?")
        return

    save_results(latencies, decisions, tier_exits, ground_truth)


if __name__ == "__main__":
    main()
