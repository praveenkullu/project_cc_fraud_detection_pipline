"""In-memory velocity checker for the fraud detection pipeline.

Tier 2 of the scoring pipeline — sliding-window counters per card and IP.
All state mutations are protected by a threading.Lock for thread safety.
"""
import threading
from collections import defaultdict


_WINDOW_1H = 3600.0
_WINDOW_24H = 86400.0

_CARD_COUNT_FLAG_THRESHOLD = 5      # > 5 triggers flag
_CARD_COUNT_DECLINE_THRESHOLD = 10  # > 10 triggers decline
_CARD_AMOUNT_STEP_UP_THRESHOLD = 2000.0  # > 2000 triggers step_up_auth
_IP_COUNT_FLAG_THRESHOLD = 15       # > 15 triggers flag


class VelocityChecker:
    """Thread-safe in-memory sliding-window velocity tracker.

    Internal state: defaultdict keyed by (entity_type, entity_id).
    Values are lists of [timestamp, amount] pairs.
    """

    def __init__(self) -> None:
        self._store: defaultdict = defaultdict(list)
        self._lock = threading.Lock()

    def record_and_check(
        self,
        card_id: str,
        ip_address: str,
        amount: float,
        timestamp: float,
    ) -> list[dict]:
        """Record a transaction and return any velocity flags.

        Prunes stale entries, records the new transaction, then checks
        thresholds for card count (1h), card amount (24h), and IP count (1h).
        """
        with self._lock:
            self._prune(("card", card_id), timestamp, _WINDOW_24H)
            self._prune(("ip", ip_address), timestamp, _WINDOW_1H)
            self._store[("card", card_id)].append([timestamp, amount])
            self._store[("ip", ip_address)].append([timestamp, 0.0])

        flags: list[dict] = []
        with self._lock:
            card_entries = self._store[("card", card_id)]
            ip_entries = self._store[("ip", ip_address)]
            card_1h = sum(1 for ts, _ in card_entries if timestamp - ts < _WINDOW_1H)
            card_24h_amount = sum(amt for ts, amt in card_entries if timestamp - ts <= _WINDOW_24H)
            ip_1h = sum(1 for ts, _ in ip_entries if timestamp - ts < _WINDOW_1H)

        if card_1h > _CARD_COUNT_DECLINE_THRESHOLD:
            flags.append({"flag": "high_card_velocity", "action": "decline"})
        elif card_1h > _CARD_COUNT_FLAG_THRESHOLD:
            flags.append({"flag": "high_card_velocity", "action": "flag"})

        if card_24h_amount > _CARD_AMOUNT_STEP_UP_THRESHOLD:
            flags.append({"flag": "high_24h_amount", "action": "step_up_auth"})

        if ip_1h > _IP_COUNT_FLAG_THRESHOLD:
            flags.append({"flag": "high_ip_velocity", "action": "flag"})

        return flags

    def get_amount_24h(self, card_id: str) -> float:
        """Return the total transaction amount for card_id in the last 24 hours.

        Uses the timestamp of the most recent entry as the reference point.
        Returns 0.0 for unknown card_ids.
        """
        with self._lock:
            entries = self._store.get(("card", card_id), [])
            if not entries:
                return 0.0
            latest_ts = max(ts for ts, _ in entries)
            return float(sum(
                amt for ts, amt in entries if latest_ts - ts <= _WINDOW_24H
            ))

    def _prune(self, key: tuple, current_ts: float, window: float) -> None:
        """Remove entries older than window seconds. Caller must hold lock."""
        cutoff = current_ts - window
        self._store[key] = [
            entry for entry in self._store[key] if entry[0] > cutoff
        ]
