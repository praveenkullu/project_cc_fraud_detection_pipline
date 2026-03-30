"""Redis-backed velocity checker for the fraud detection pipeline.

Phase 3 Step 16 — implements the same interface as VelocityChecker
using Redis sorted sets for sliding-window counters.
"""
import uuid

from src.velocity import (
    _WINDOW_1H,
    _WINDOW_24H,
    _CARD_COUNT_FLAG_THRESHOLD,
    _CARD_COUNT_DECLINE_THRESHOLD,
    _CARD_AMOUNT_STEP_UP_THRESHOLD,
    _IP_COUNT_FLAG_THRESHOLD,
    VelocityChecker,
)


class RedisVelocityChecker:
    """Redis-backed sliding-window velocity tracker.

    Uses sorted sets with timestamp as score for O(log N) window queries.
    Key schema: velocity:{entity_type}:{entity_id}
    Amount tracking: velocity:amount:card:{card_id} (score=timestamp, member=amount:uuid)
    """

    def __init__(self, redis_client) -> None:
        self._r = redis_client

    def record_and_check(
        self,
        card_id: str,
        ip_address: str,
        amount: float,
        timestamp: float,
    ) -> list[dict]:
        """Record a transaction and return any velocity flags.

        Uses Redis sorted sets with pipelined commands for efficiency.
        """
        card_key = f"velocity:card:{card_id}"
        ip_key = f"velocity:ip:{ip_address}"
        amount_key = f"velocity:amount:{card_id}"

        member_id = str(uuid.uuid4())

        pipe = self._r.pipeline()
        # Prune old entries
        pipe.zremrangebyscore(card_key, "-inf", timestamp - _WINDOW_24H)
        pipe.zremrangebyscore(ip_key, "-inf", timestamp - _WINDOW_1H)
        pipe.zremrangebyscore(amount_key, "-inf", timestamp - _WINDOW_24H)
        # Add new entries
        pipe.zadd(card_key, {f"{member_id}:card": timestamp})
        pipe.zadd(ip_key, {f"{member_id}:ip": timestamp})
        pipe.zadd(amount_key, {f"{amount}:{member_id}": timestamp})
        # Count within windows
        pipe.zcount(card_key, timestamp - _WINDOW_1H + 0.001, "+inf")
        pipe.zrangebyscore(amount_key, timestamp - _WINDOW_24H, "+inf")
        pipe.zcount(ip_key, timestamp - _WINDOW_1H + 0.001, "+inf")
        results = pipe.execute()

        card_1h_count = results[6]
        amount_members = results[7]
        ip_1h_count = results[8]

        # Sum amounts from member names (format: "amount:uuid")
        card_24h_amount = sum(
            float(m.split(":")[0]) for m in amount_members
        )

        flags: list[dict] = []

        if card_1h_count > _CARD_COUNT_DECLINE_THRESHOLD:
            flags.append({"flag": "high_card_velocity", "action": "decline"})
        elif card_1h_count > _CARD_COUNT_FLAG_THRESHOLD:
            flags.append({"flag": "high_card_velocity", "action": "flag"})

        if card_24h_amount > _CARD_AMOUNT_STEP_UP_THRESHOLD:
            flags.append({"flag": "high_24h_amount", "action": "step_up_auth"})

        if ip_1h_count > _IP_COUNT_FLAG_THRESHOLD:
            flags.append({"flag": "high_ip_velocity", "action": "flag"})

        return flags

    def get_amount_24h(self, card_id: str) -> float:
        """Return the total transaction amount for card_id in the last 24 hours."""
        amount_key = f"velocity:amount:{card_id}"

        # Get the latest timestamp to use as reference
        latest_entry = self._r.zrevrange(amount_key, 0, 0, withscores=True)
        if not latest_entry:
            return 0.0

        latest_ts = latest_entry[0][1]
        cutoff = latest_ts - _WINDOW_24H

        valid_members = self._r.zrangebyscore(amount_key, cutoff, "+inf")
        return sum(float(m.split(":")[0]) for m in valid_members)


def get_velocity_checker(
    backend: str = "memory", redis_client=None
) -> "VelocityChecker | RedisVelocityChecker":
    """Factory function to select velocity backend.

    Args:
        backend: "memory" or "redis"
        redis_client: Required when backend="redis"
    """
    if backend == "memory":
        return VelocityChecker()
    elif backend == "redis":
        if redis_client is None:
            import redis
            redis_client = redis.Redis(decode_responses=True)
        return RedisVelocityChecker(redis_client)
    else:
        raise ValueError(f"Unknown velocity backend: {backend}")
