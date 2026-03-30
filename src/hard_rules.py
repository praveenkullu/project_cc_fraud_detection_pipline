"""Hard rule checks for the fraud detection pipeline.

Tier 1 of the scoring pipeline — deterministic, fast checks that can
immediately decline a transaction without invoking ML models.
"""

BLOCKED_BINS: frozenset = frozenset({
    "999999",
    "000000",
    "123456",
    "111111",
    "666666",
})

_NEW_ACCOUNT_AMOUNT_THRESHOLD = 500.0
_NEW_ACCOUNT_AGE_THRESHOLD = 24.0


def check_hard_rules(
    bin_number: str,
    amount: float,
    account_age_hours: float,
) -> tuple[bool, list[dict]]:
    """Apply deterministic hard rules to a transaction.

    Rules evaluated in order:
    1. BIN blocklist — if bin_number is in BLOCKED_BINS, decline immediately.
    2. New account high amount — if amount > 500 and account_age_hours < 24, decline.

    Returns:
        (declined, triggered_rules) where declined is True if any rule fired
        and triggered_rules is a list of dicts with 'rule' and 'action' keys.
    """
    declined = False
    triggered_rules: list[dict] = []

    if bin_number in BLOCKED_BINS:
        triggered_rules.append({"rule": "blocked_bin", "action": "decline"})
        declined = True

    if amount > _NEW_ACCOUNT_AMOUNT_THRESHOLD and account_age_hours < _NEW_ACCOUNT_AGE_THRESHOLD:
        triggered_rules.append({"rule": "new_account_high_amount", "action": "decline"})
        declined = True

    return declined, triggered_rules
