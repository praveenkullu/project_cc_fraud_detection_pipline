"""Feature engineering utilities for the fraud detection pipeline.

These functions are used in notebooks and the serving API.
All functions are pure (no side effects) to enable easy testing.
"""
import math
import numpy as np
import pandas as pd
from typing import Optional


def log_amount(amount: float) -> float:
    """Apply log1p transform to transaction amount."""
    return math.log1p(amount)


def hour_of_day(transaction_dt: float) -> int:
    """Extract hour-of-day (0–23) from TransactionDT (seconds since reference).

    TransactionDT is seconds elapsed since a reference epoch — not a real
    Unix timestamp. Hour-of-day captures intra-day spending patterns.
    """
    return int((transaction_dt % 86400) // 3600)


def amount_zscore(amount: float, mean: float, std: float) -> float:
    """Standardise amount using training-set statistics.

    std=0 returns 0.0 to avoid division by zero.
    """
    if std == 0.0:
        return 0.0
    return (amount - mean) / std


def normalize_iforest_scores(scores: np.ndarray) -> np.ndarray:
    """Map raw Isolation Forest decision_function scores to [0, 1].

    IsolationForest.decision_function returns negative values for anomalies
    and positive for inliers. We negate then min-max scale so that
    1 = most anomalous, 0 = most normal.

    Constant-score arrays (all identical values) return all zeros.
    """
    negated = -scores
    mn, mx = negated.min(), negated.max()
    if mx == mn:
        return np.zeros_like(scores, dtype=float)
    return (negated - mn) / (mx - mn)


def composite_score(
    xgb_prob: float,
    iforest_normalized: float,
    alpha: float = 0.7,
) -> float:
    """Combine XGBoost probability and normalised IsolationForest score.

    composite = alpha * xgb_prob + (1 - alpha) * iforest_normalized

    Both inputs must be in [0, 1]. alpha must be in [0, 1].
    Raises ValueError for out-of-range inputs.
    """
    if not (0.0 <= xgb_prob <= 1.0):
        raise ValueError(f"xgb_prob must be in [0, 1], got {xgb_prob}")
    if not (0.0 <= iforest_normalized <= 1.0):
        raise ValueError(f"iforest_normalized must be in [0, 1], got {iforest_normalized}")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    return alpha * xgb_prob + (1.0 - alpha) * iforest_normalized


def engineer_features(
    df: pd.DataFrame,
    amount_mean: Optional[float] = None,
    amount_std: Optional[float] = None,
) -> pd.DataFrame:
    """Add derived features to a transaction DataFrame.

    Adds columns: log_amount, hour_of_day, amount_zscore.

    If amount_mean / amount_std are None they are computed from df
    (training mode). Pass training-set stats at inference to avoid leakage.

    Returns a new DataFrame (does not mutate input).
    """
    out = df.copy()

    out["log_amount"] = np.log1p(out["TransactionAmt"].values)
    out["hour_of_day"] = ((out["TransactionDT"].values % 86400) // 3600).astype(int)

    if amount_mean is None:
        amount_mean = float(out["TransactionAmt"].mean())
    if amount_std is None:
        amount_std = float(out["TransactionAmt"].std())
        if np.isnan(amount_std):  # single-row DataFrame — ddof=1 yields NaN
            amount_std = 0.0

    if amount_std == 0.0 or np.isnan(amount_std):
        out["amount_zscore"] = 0.0
    else:
        out["amount_zscore"] = (out["TransactionAmt"] - amount_mean) / amount_std

    return out
