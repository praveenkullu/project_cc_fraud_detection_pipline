"""Decision routing for the fraud scoring pipeline."""
from enum import Enum


class Decision(str, Enum):
    APPROVE = "approve"
    MANUAL_REVIEW = "manual_review"
    DECLINE = "decline"
    STEP_UP_AUTH = "step_up_auth"


def route_decision(
    composite: float,
    amount_24h: float = 0.0,
    step_up_threshold: float = 2000.0,
    low: float = 0.3,
    high: float = 0.7,
) -> Decision:
    """Map a composite fraud score + 24-hour spend to a Decision.

    Rules (evaluated in order):
    1. composite >= high  → DECLINE
    2. amount_24h > step_up_threshold → STEP_UP_AUTH
    3. composite >= low   → MANUAL_REVIEW
    4. else               → APPROVE

    composite must be in [0, 1]. Raises ValueError otherwise.
    """
    if not (0.0 <= composite <= 1.0):
        raise ValueError(f"composite must be in [0, 1], got {composite}")

    if composite >= high:
        return Decision.DECLINE
    if amount_24h > step_up_threshold:
        return Decision.STEP_UP_AUTH
    if composite >= low:
        return Decision.MANUAL_REVIEW
    return Decision.APPROVE
