"""Tests for src.hard_rules — TDD RED phase.

Written BEFORE implementation. All tests will fail with ImportError
until src/hard_rules.py is created.
"""
import pytest

from src.hard_rules import check_hard_rules, BLOCKED_BINS


class TestCheckHardRules:
    def test_clean_transaction_no_rules_triggered(self):
        """Normal transaction with unknown BIN and low amount returns no flags."""
        declined, rules = check_hard_rules(
            bin_number="400000",
            amount=50.0,
            account_age_hours=720.0,
        )
        assert declined is False
        assert rules == []

    def test_blocked_bin_declines(self):
        """BIN in BLOCKED_BINS set returns decline with blocked_bin rule."""
        declined, rules = check_hard_rules(
            bin_number="999999",
            amount=10.0,
            account_age_hours=720.0,
        )
        assert declined is True
        assert len(rules) >= 1
        rule_names = [r["rule"] for r in rules]
        assert "blocked_bin" in rule_names
        blocked_rule = next(r for r in rules if r["rule"] == "blocked_bin")
        assert blocked_rule["action"] == "decline"

    def test_non_blocked_bin_passes(self):
        """Unknown BIN that is not in BLOCKED_BINS does not trigger blocklist."""
        declined, rules = check_hard_rules(
            bin_number="555555",
            amount=50.0,
            account_age_hours=720.0,
        )
        assert declined is False
        rule_names = [r["rule"] for r in rules]
        assert "blocked_bin" not in rule_names

    def test_new_account_high_amount_declines(self):
        """Account age < 24h with amount > 500 triggers new_account_high_amount."""
        declined, rules = check_hard_rules(
            bin_number="400000",
            amount=1000.0,
            account_age_hours=12.0,
        )
        assert declined is True
        rule_names = [r["rule"] for r in rules]
        assert "new_account_high_amount" in rule_names
        rule = next(r for r in rules if r["rule"] == "new_account_high_amount")
        assert rule["action"] == "decline"

    def test_new_account_low_amount_passes(self):
        """Account age < 24h but amount <= 500 does NOT trigger rule."""
        declined, rules = check_hard_rules(
            bin_number="400000",
            amount=100.0,
            account_age_hours=12.0,
        )
        assert declined is False
        rule_names = [r["rule"] for r in rules]
        assert "new_account_high_amount" not in rule_names

    def test_old_account_high_amount_passes(self):
        """Account age >= 24h with high amount does NOT trigger new_account rule."""
        declined, rules = check_hard_rules(
            bin_number="400000",
            amount=1000.0,
            account_age_hours=48.0,
        )
        assert declined is False
        rule_names = [r["rule"] for r in rules]
        assert "new_account_high_amount" not in rule_names

    def test_return_type_is_tuple(self):
        """Function always returns a tuple."""
        result = check_hard_rules("400000", 50.0, 720.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_return_first_element_is_bool(self):
        """First element of return tuple is always a bool."""
        declined, _ = check_hard_rules("400000", 50.0, 720.0)
        assert isinstance(declined, bool)

    def test_return_second_element_is_list(self):
        """Second element of return tuple is always a list."""
        _, rules = check_hard_rules("400000", 50.0, 720.0)
        assert isinstance(rules, list)

    def test_triggered_rule_has_required_keys(self):
        """Each triggered rule dict must contain 'rule' and 'action' keys."""
        _, rules = check_hard_rules("999999", 50.0, 720.0)
        assert len(rules) >= 1
        for rule in rules:
            assert "rule" in rule
            assert "action" in rule

    def test_both_rules_triggered_simultaneously(self):
        """Blocked BIN + new account high amount — both rules fire, declined=True."""
        declined, rules = check_hard_rules(
            bin_number="999999",
            amount=1000.0,
            account_age_hours=12.0,
        )
        assert declined is True
        rule_names = [r["rule"] for r in rules]
        assert "blocked_bin" in rule_names
        assert "new_account_high_amount" in rule_names

    def test_amount_at_boundary_does_not_decline(self):
        """Amount exactly at 500 does NOT trigger new_account_high_amount (must be > 500)."""
        declined, rules = check_hard_rules(
            bin_number="400000",
            amount=500.0,
            account_age_hours=12.0,
        )
        assert declined is False

    def test_account_age_at_boundary_does_not_decline(self):
        """Account age exactly at 24h does NOT trigger new_account rule (must be < 24)."""
        declined, rules = check_hard_rules(
            bin_number="400000",
            amount=1000.0,
            account_age_hours=24.0,
        )
        assert declined is False

    def test_blocked_bins_is_frozenset(self):
        """BLOCKED_BINS must be a frozenset."""
        assert isinstance(BLOCKED_BINS, frozenset)

    def test_blocked_bins_has_at_least_five_entries(self):
        """BLOCKED_BINS must contain at least 5 BINs."""
        assert len(BLOCKED_BINS) >= 5

    def test_all_blocked_bins_decline(self):
        """Every BIN in BLOCKED_BINS triggers a decline."""
        for bin_num in BLOCKED_BINS:
            declined, rules = check_hard_rules(bin_num, 10.0, 720.0)
            assert declined is True, f"Expected decline for BIN {bin_num}"
