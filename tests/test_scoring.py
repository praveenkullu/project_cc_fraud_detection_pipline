"""Tests for src.scoring — written BEFORE implementation (TDD RED phase)."""
import pytest

from src.scoring import Decision, route_decision


class TestRouteDecision:
    def test_high_score_declines(self):
        assert route_decision(0.7) == Decision.DECLINE

    def test_above_high_threshold_declines(self):
        assert route_decision(0.9) == Decision.DECLINE

    def test_exact_high_threshold_declines(self):
        assert route_decision(0.7) == Decision.DECLINE

    def test_low_score_approves(self):
        assert route_decision(0.1) == Decision.APPROVE

    def test_below_low_threshold_approves(self):
        assert route_decision(0.0) == Decision.APPROVE

    def test_mid_range_is_manual_review(self):
        assert route_decision(0.5) == Decision.MANUAL_REVIEW

    def test_exact_low_threshold_is_manual_review(self):
        assert route_decision(0.3) == Decision.MANUAL_REVIEW

    def test_just_below_high_is_manual_review(self):
        assert route_decision(0.699) == Decision.MANUAL_REVIEW

    def test_step_up_auth_when_large_24h_spend(self):
        # score is low but 24h spend exceeds threshold
        assert route_decision(0.2, amount_24h=2500.0) == Decision.STEP_UP_AUTH

    def test_decline_takes_precedence_over_step_up(self):
        # High score should decline even if 24h spend is large
        assert route_decision(0.8, amount_24h=3000.0) == Decision.DECLINE

    def test_step_up_not_triggered_at_threshold(self):
        # Exactly at threshold → not triggered (must EXCEED)
        result = route_decision(0.2, amount_24h=2000.0)
        assert result == Decision.APPROVE

    def test_custom_thresholds(self):
        result = route_decision(0.6, low=0.5, high=0.8)
        assert result == Decision.MANUAL_REVIEW

    def test_invalid_composite_below_zero_raises(self):
        with pytest.raises(ValueError):
            route_decision(-0.1)

    def test_invalid_composite_above_one_raises(self):
        with pytest.raises(ValueError):
            route_decision(1.1)
