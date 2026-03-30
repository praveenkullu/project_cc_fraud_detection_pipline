"""Tests for src.features — written BEFORE implementation (TDD RED phase)."""
import math
import numpy as np
import pandas as pd
import pytest

from src.features import (
    log_amount,
    hour_of_day,
    amount_zscore,
    normalize_iforest_scores,
    composite_score,
    engineer_features,
)


# ---------------------------------------------------------------------------
# log_amount
# ---------------------------------------------------------------------------

class TestLogAmount:
    def test_zero_returns_zero(self):
        assert log_amount(0) == pytest.approx(0.0)

    def test_one_returns_log2(self):
        assert log_amount(1) == pytest.approx(math.log(2))

    def test_large_value(self):
        assert log_amount(999) == pytest.approx(math.log1p(999))

    def test_positive_amount(self):
        assert log_amount(100.0) > 0


# ---------------------------------------------------------------------------
# hour_of_day
# ---------------------------------------------------------------------------

class TestHourOfDay:
    def test_midnight(self):
        assert hour_of_day(0) == 0

    def test_one_hour(self):
        assert hour_of_day(3600) == 1

    def test_last_second_of_day(self):
        assert hour_of_day(86399) == 23

    def test_second_day_midnight(self):
        # 86400 seconds = start of day 2, wraps back to hour 0
        assert hour_of_day(86400) == 0

    def test_afternoon(self):
        # 54000 = 15 * 3600
        assert hour_of_day(54000) == 15

    def test_return_type_is_int(self):
        assert isinstance(hour_of_day(3600), int)


# ---------------------------------------------------------------------------
# amount_zscore
# ---------------------------------------------------------------------------

class TestAmountZscore:
    def test_mean_value_returns_zero(self):
        assert amount_zscore(100.0, mean=100.0, std=20.0) == pytest.approx(0.0)

    def test_one_std_above(self):
        assert amount_zscore(120.0, mean=100.0, std=20.0) == pytest.approx(1.0)

    def test_one_std_below(self):
        assert amount_zscore(80.0, mean=100.0, std=20.0) == pytest.approx(-1.0)

    def test_zero_std_returns_zero(self):
        # Guard against division by zero — constant feature has no signal
        assert amount_zscore(50.0, mean=50.0, std=0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# normalize_iforest_scores
# ---------------------------------------------------------------------------

class TestNormalizeIforestScores:
    def test_output_in_unit_interval(self):
        raw = np.array([-0.5, 0.0, 0.3, -0.1, 0.2])
        out = normalize_iforest_scores(raw)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_most_negative_maps_to_one(self):
        # IsoForest: most negative = most anomalous → should map to 1
        raw = np.array([-1.0, 0.0, 0.5])
        out = normalize_iforest_scores(raw)
        assert out[0] == pytest.approx(1.0)

    def test_most_positive_maps_to_zero(self):
        raw = np.array([-1.0, 0.0, 0.5])
        out = normalize_iforest_scores(raw)
        assert out[2] == pytest.approx(0.0)

    def test_constant_array_returns_zeros(self):
        raw = np.array([0.3, 0.3, 0.3])
        out = normalize_iforest_scores(raw)
        np.testing.assert_array_equal(out, np.zeros(3))

    def test_output_shape_preserved(self):
        raw = np.linspace(-0.5, 0.5, 100)
        out = normalize_iforest_scores(raw)
        assert out.shape == raw.shape


# ---------------------------------------------------------------------------
# composite_score
# ---------------------------------------------------------------------------

class TestCompositeScore:
    def test_both_one(self):
        assert composite_score(1.0, 1.0) == pytest.approx(1.0)

    def test_both_zero(self):
        assert composite_score(0.0, 0.0) == pytest.approx(0.0)

    def test_default_weights_70_30(self):
        # alpha=0.7 default: 0.7*0.8 + 0.3*0.4 = 0.56 + 0.12 = 0.68
        assert composite_score(0.8, 0.4) == pytest.approx(0.68)

    def test_equal_weights(self):
        assert composite_score(1.0, 0.0, alpha=0.5) == pytest.approx(0.5)

    def test_result_in_unit_interval(self):
        result = composite_score(0.6, 0.4)
        assert 0.0 <= result <= 1.0

    def test_invalid_xgb_prob_raises(self):
        with pytest.raises(ValueError):
            composite_score(1.5, 0.5)

    def test_invalid_iforest_score_raises(self):
        with pytest.raises(ValueError):
            composite_score(0.5, -0.1)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            composite_score(0.5, 0.5, alpha=1.5)


# ---------------------------------------------------------------------------
# engineer_features
# ---------------------------------------------------------------------------

class TestEngineerFeatures:
    def _make_df(self):
        return pd.DataFrame({
            "TransactionAmt": [10.0, 100.0, 1000.0],
            "TransactionDT": [0.0, 3600.0, 54000.0],
        })

    def test_adds_log_amount_column(self):
        df = self._make_df()
        out = engineer_features(df)
        assert "log_amount" in out.columns

    def test_adds_hour_of_day_column(self):
        df = self._make_df()
        out = engineer_features(df)
        assert "hour_of_day" in out.columns

    def test_adds_amount_zscore_column(self):
        df = self._make_df()
        out = engineer_features(df)
        assert "amount_zscore" in out.columns

    def test_does_not_mutate_input(self):
        df = self._make_df()
        original_cols = list(df.columns)
        engineer_features(df)
        assert list(df.columns) == original_cols

    def test_log_amount_values_correct(self):
        df = self._make_df()
        out = engineer_features(df)
        expected = np.log1p([10.0, 100.0, 1000.0])
        np.testing.assert_allclose(out["log_amount"].values, expected)

    def test_hour_of_day_values_correct(self):
        df = self._make_df()
        out = engineer_features(df)
        assert list(out["hour_of_day"]) == [0, 1, 15]

    def test_training_mode_computes_stats(self):
        df = self._make_df()
        out = engineer_features(df)
        # zscore of mean value should be ≈ 0
        mean_amt = df["TransactionAmt"].mean()
        out2 = engineer_features(
            pd.DataFrame({"TransactionAmt": [mean_amt], "TransactionDT": [0.0]})
        )
        assert out2["amount_zscore"].iloc[0] == pytest.approx(0.0, abs=1e-6)

    def test_inference_mode_uses_provided_stats(self):
        df = self._make_df()
        # If we pass mean=100, std=1, then amount=100 → zscore=0
        out = engineer_features(df, amount_mean=100.0, amount_std=1.0)
        assert out.loc[out["TransactionAmt"] == 100.0, "amount_zscore"].iloc[0] == pytest.approx(0.0)

    def test_zero_std_handled_gracefully(self):
        df = pd.DataFrame({
            "TransactionAmt": [50.0, 50.0],
            "TransactionDT": [0.0, 0.0],
        })
        out = engineer_features(df)
        # All zscores should be 0.0 when std=0
        assert (out["amount_zscore"] == 0.0).all()
