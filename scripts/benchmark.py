"""Latency benchmark for the fraud detection pipeline — Phase 3 Step 20.

Sends 1,000 transactions to POST /predict and analyzes per-tier latency.
Output: results/latency_benchmark.png + results/tier_exit_rates.txt

Usage:
    python scripts/benchmark.py [--url http://localhost:8000] [--n 1000]
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import httpx
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent / "results"


def generate_transactions(n: int) -> list[dict]:
    """Generate a mix of transaction types.

    40% normal, 30% velocity-triggering, 20% hard-rule-triggering, 10% ML-heavy.
    """
    transactions = []
    base_ts = 1_000_000.0

    for i in range(n):
        r = random.random()
        if r < 0.2:
            # Hard-rule triggering (20%): blocked BIN
            txn = {
                "card_id": f"bench_hard_{i}",
                "amount": 50.0,
                "merchant_id": "merch_bench",
                "timestamp": base_ts + i,
                "ip_address": f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}",
                "account_age_hours": 720.0,
                "bin_number": random.choice(["999999", "000000", "123456"]),
                "features": {},
            }
        elif r < 0.5:
            # Velocity-triggering (30%): same card to accumulate hits
            txn = {
                "card_id": "bench_velocity_card",
                "amount": 10.0,
                "merchant_id": "merch_bench",
                "timestamp": base_ts + i * 0.1,
                "ip_address": "10.0.0.1",
                "account_age_hours": 720.0,
                "bin_number": "400000",
                "features": {},
            }
        elif r < 0.9:
            # Normal (40%)
            txn = {
                "card_id": f"bench_normal_{i}",
                "amount": round(random.uniform(10.0, 500.0), 2),
                "merchant_id": f"merch_{random.randint(1, 100)}",
                "timestamp": base_ts + i,
                "ip_address": f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}",
                "account_age_hours": float(random.randint(24, 8760)),
                "bin_number": f"{random.randint(400000, 499999)}",
                "features": {},
            }
        else:
            # ML-heavy (10%): large amounts, borderline cases
            txn = {
                "card_id": f"bench_ml_{i}",
                "amount": round(random.uniform(500.0, 5000.0), 2),
                "merchant_id": f"merch_{random.randint(1, 100)}",
                "timestamp": base_ts + i,
                "ip_address": f"10.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}",
                "account_age_hours": float(random.randint(1, 48)),
                "bin_number": f"{random.randint(400000, 499999)}",
                "features": {},
            }
        transactions.append(txn)

    return transactions


def run_benchmark(api_url: str, n: int):
    """Send transactions and collect latency data."""
    transactions = generate_transactions(n)

    latencies = []
    decisions = []
    tier_exits = {"hard_rules": 0, "velocity": 0, "ml_scoring": 0}

    print(f"Sending {n} transactions to {api_url}/predict ...")
    client = httpx.Client(timeout=30.0)

    for i, txn in enumerate(transactions):
        t0 = time.perf_counter()
        try:
            resp = client.post(f"{api_url}/predict", json=txn)
            t1 = time.perf_counter()

            if resp.status_code != 200:
                continue

            body = resp.json()
            latency_ms = (t1 - t0) * 1000
            latencies.append(latency_ms)
            decisions.append(body["decision"])

            # Determine which tier caused the exit
            tier_names = [t.get("tier") for t in body["tier_results"]]
            if "hard_rules" in tier_names and body["decision"] == "decline":
                if "ml_scoring" not in tier_names:
                    tier_exits["hard_rules"] += 1
                    continue
            if "velocity" in tier_names:
                velocity_tiers = [t for t in body["tier_results"] if t.get("tier") == "velocity"]
                if velocity_tiers and body["decision"] == "decline":
                    tier_exits["velocity"] += 1
                    continue
            tier_exits["ml_scoring"] += 1

        except httpx.RequestError as e:
            print(f"  Request {i} failed: {e}")

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n} done")

    client.close()
    return latencies, decisions, tier_exits


def save_results(latencies, decisions, tier_exits, n):
    """Save benchmark results."""
    RESULTS_DIR.mkdir(exist_ok=True)

    latencies_arr = np.array(latencies)

    # Tier exit rates
    total = sum(tier_exits.values())
    with open(RESULTS_DIR / "tier_exit_rates.txt", "w") as f:
        f.write("Tier Exit Rate Analysis\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total scored transactions: {total}\n\n")
        for tier, count in tier_exits.items():
            pct = (count / total * 100) if total > 0 else 0
            f.write(f"  {tier:20s}: {count:5d} ({pct:5.1f}%)\n")

        f.write(f"\nLatency Statistics (ms)\n")
        f.write("=" * 40 + "\n")
        if len(latencies_arr) > 0:
            f.write(f"  p50:  {np.percentile(latencies_arr, 50):8.2f}\n")
            f.write(f"  p95:  {np.percentile(latencies_arr, 95):8.2f}\n")
            f.write(f"  p99:  {np.percentile(latencies_arr, 99):8.2f}\n")
            f.write(f"  mean: {np.mean(latencies_arr):8.2f}\n")
            f.write(f"  min:  {np.min(latencies_arr):8.2f}\n")
            f.write(f"  max:  {np.max(latencies_arr):8.2f}\n")

        f.write(f"\nDecision Distribution\n")
        f.write("=" * 40 + "\n")
        from collections import Counter
        for decision, count in Counter(decisions).most_common():
            f.write(f"  {decision:20s}: {count:5d}\n")

    print(f"\nResults saved to {RESULTS_DIR / 'tier_exit_rates.txt'}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Latency histogram
        axes[0].hist(latencies_arr, bins=50, edgecolor="black", alpha=0.7)
        axes[0].axvline(np.percentile(latencies_arr, 50), color="green", linestyle="--", label="p50")
        axes[0].axvline(np.percentile(latencies_arr, 95), color="orange", linestyle="--", label="p95")
        axes[0].axvline(np.percentile(latencies_arr, 99), color="red", linestyle="--", label="p99")
        axes[0].set_xlabel("Latency (ms)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("End-to-End Prediction Latency")
        axes[0].legend()

        # Tier exit pie chart
        labels = list(tier_exits.keys())
        sizes = list(tier_exits.values())
        axes[1].pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
        axes[1].set_title("Tier Exit Rates")

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "latency_benchmark.png", dpi=150)
        print(f"Plot saved to {RESULTS_DIR / 'latency_benchmark.png'}")
    except ImportError:
        print("matplotlib not available — skipping plot generation")


def main():
    parser = argparse.ArgumentParser(description="Fraud pipeline latency benchmark")
    parser.add_argument("--url", default="http://localhost:8001", help="API base URL")
    parser.add_argument("--n", type=int, default=1000, help="Number of transactions")
    args = parser.parse_args()

    latencies, decisions, tier_exits = run_benchmark(args.url, args.n)

    if not latencies:
        print("No successful responses. Is the API running?")
        return

    save_results(latencies, decisions, tier_exits, args.n)


if __name__ == "__main__":
    main()
