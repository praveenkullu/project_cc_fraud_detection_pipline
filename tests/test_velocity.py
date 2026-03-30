"""Tests for src.velocity — TDD RED phase.

Written BEFORE implementation. All tests will fail with ImportError
until src/velocity.py is created.
"""
import threading
import time
import pytest

from src.velocity import VelocityChecker


class TestVelocityChecker:
    def setup_method(self):
        """Fresh VelocityChecker for every test — no shared state."""
        self.vc = VelocityChecker()
        self.base_ts = 1_000_000.0  # arbitrary reference timestamp

    # ------------------------------------------------------------------
    # Basic behavior
    # ------------------------------------------------------------------

    def test_first_transaction_no_flags(self):
        """First transaction on any card returns empty list."""
        flags = self.vc.record_and_check(
            card_id="card_001",
            ip_address="1.2.3.4",
            amount=50.0,
            timestamp=self.base_ts,
        )
        assert flags == []

    def test_return_type_is_list(self):
        """record_and_check always returns a list."""
        result = self.vc.record_and_check("card_001", "1.2.3.4", 50.0, self.base_ts)
        assert isinstance(result, list)

    # ------------------------------------------------------------------
    # Card count velocity (1-hour window)
    # ------------------------------------------------------------------

    def test_card_count_flag_threshold(self):
        """6th transaction within 1 hour returns high_card_velocity flag action."""
        for i in range(5):
            self.vc.record_and_check(
                "card_001", "1.2.3.4", 10.0, self.base_ts + i
            )
        flags = self.vc.record_and_check(
            "card_001", "1.2.3.4", 10.0, self.base_ts + 5
        )
        flag_names = [f["flag"] for f in flags]
        assert "high_card_velocity" in flag_names
        hcv = next(f for f in flags if f["flag"] == "high_card_velocity")
        assert hcv["action"] == "flag"

    def test_card_count_decline_threshold(self):
        """11th transaction within 1 hour returns high_card_velocity with decline action."""
        for i in range(10):
            self.vc.record_and_check(
                "card_001", "1.2.3.4", 10.0, self.base_ts + i
            )
        flags = self.vc.record_and_check(
            "card_001", "1.2.3.4", 10.0, self.base_ts + 10
        )
        flag_names = [f["flag"] for f in flags]
        assert "high_card_velocity" in flag_names
        hcv = next(f for f in flags if f["flag"] == "high_card_velocity")
        assert hcv["action"] == "decline"

    def test_card_count_at_five_no_flag(self):
        """Exactly 5 transactions does NOT trigger a flag (threshold is > 5)."""
        for i in range(4):
            self.vc.record_and_check("card_001", "1.2.3.4", 10.0, self.base_ts + i)
        flags = self.vc.record_and_check("card_001", "1.2.3.4", 10.0, self.base_ts + 4)
        flag_names = [f["flag"] for f in flags]
        assert "high_card_velocity" not in flag_names

    # ------------------------------------------------------------------
    # 24-hour amount accumulation
    # ------------------------------------------------------------------

    def test_amount_24h_accumulates(self):
        """get_amount_24h reflects accumulated amounts correctly."""
        self.vc.record_and_check("card_001", "1.2.3.4", 100.0, self.base_ts)
        self.vc.record_and_check("card_001", "1.2.3.4", 200.0, self.base_ts + 60)
        total = self.vc.get_amount_24h("card_001")
        assert total == pytest.approx(300.0)

    def test_amount_24h_step_up_auth_flag(self):
        """Cumulative amount > 2000 in 24h returns step_up_auth flag."""
        for i in range(5):
            self.vc.record_and_check(
                "card_001", "1.2.3.4", 500.0, self.base_ts + i * 60
            )
        # 5 * 500 = 2500 > 2000
        flags = self.vc.record_and_check(
            "card_001", "1.2.3.4", 1.0, self.base_ts + 300
        )
        flag_names = [f["flag"] for f in flags]
        assert "high_24h_amount" in flag_names
        hav = next(f for f in flags if f["flag"] == "high_24h_amount")
        assert hav["action"] == "step_up_auth"

    def test_get_amount_24h_returns_zero_for_unknown(self):
        """Unknown card_id returns 0.0."""
        assert self.vc.get_amount_24h("nonexistent_card") == pytest.approx(0.0)

    def test_get_amount_24h_return_type(self):
        """get_amount_24h always returns a float."""
        result = self.vc.get_amount_24h("any_card")
        assert isinstance(result, float)

    # ------------------------------------------------------------------
    # IP count velocity (1-hour window)
    # ------------------------------------------------------------------

    def test_ip_count_flag_threshold(self):
        """16th transaction from same IP in 1 hour returns high_ip_velocity flag."""
        for i in range(15):
            self.vc.record_and_check(
                f"card_{i:03d}", "10.0.0.1", 10.0, self.base_ts + i
            )
        flags = self.vc.record_and_check(
            "card_999", "10.0.0.1", 10.0, self.base_ts + 15
        )
        flag_names = [f["flag"] for f in flags]
        assert "high_ip_velocity" in flag_names
        hiv = next(f for f in flags if f["flag"] == "high_ip_velocity")
        assert hiv["action"] == "flag"

    def test_ip_count_at_fifteen_no_flag(self):
        """Exactly 15 IP transactions does NOT trigger flag (threshold is > 15)."""
        for i in range(14):
            self.vc.record_and_check(
                f"card_{i:03d}", "10.0.0.1", 10.0, self.base_ts + i
            )
        flags = self.vc.record_and_check(
            "card_999", "10.0.0.1", 10.0, self.base_ts + 14
        )
        flag_names = [f["flag"] for f in flags]
        assert "high_ip_velocity" not in flag_names

    # ------------------------------------------------------------------
    # Sliding window expiry
    # ------------------------------------------------------------------

    def test_sliding_window_expiry(self):
        """Transactions older than the 1-hour window don't count toward velocity."""
        # Record 6 transactions well in the past (> 1 hour ago)
        old_ts = self.base_ts - 7200.0  # 2 hours ago
        for i in range(6):
            self.vc.record_and_check("card_001", "1.2.3.4", 10.0, old_ts + i)

        # A new transaction right now should see only 1 tx in the window
        flags = self.vc.record_and_check("card_001", "1.2.3.4", 10.0, self.base_ts)
        flag_names = [f["flag"] for f in flags]
        assert "high_card_velocity" not in flag_names

    def test_24h_window_expiry(self):
        """Amounts older than 24h do NOT count toward get_amount_24h."""
        old_ts = self.base_ts - 90000.0  # 25 hours ago
        self.vc.record_and_check("card_001", "1.2.3.4", 5000.0, old_ts)
        # Force a check at current time to trigger pruning
        self.vc.record_and_check("card_001", "1.2.3.4", 1.0, self.base_ts)
        total = self.vc.get_amount_24h("card_001")
        # Only the 1.0 from base_ts should count, not the 5000 from 25h ago
        assert total == pytest.approx(1.0)

    # ------------------------------------------------------------------
    # Independence between entities
    # ------------------------------------------------------------------

    def test_different_cards_independent(self):
        """Card A velocity does not affect Card B."""
        for i in range(10):
            self.vc.record_and_check("card_AAA", "1.2.3.4", 10.0, self.base_ts + i)

        # Card B has only 1 transaction — should get no flags
        flags = self.vc.record_and_check("card_BBB", "1.2.3.4", 10.0, self.base_ts)
        flag_names = [f["flag"] for f in flags]
        assert "high_card_velocity" not in flag_names

    def test_different_ips_independent(self):
        """IP A velocity does not affect IP B."""
        for i in range(15):
            self.vc.record_and_check(f"card_{i}", "192.168.1.1", 10.0, self.base_ts + i)

        flags = self.vc.record_and_check("card_new", "10.0.0.99", 10.0, self.base_ts)
        flag_names = [f["flag"] for f in flags]
        assert "high_ip_velocity" not in flag_names

    # ------------------------------------------------------------------
    # Thread safety
    # ------------------------------------------------------------------

    def test_thread_safety(self):
        """20 concurrent threads incrementing same card don't corrupt count."""
        results = []
        barrier = threading.Barrier(20)

        def worker(thread_id):
            barrier.wait()  # all threads start simultaneously
            flags = self.vc.record_and_check(
                "shared_card",
                f"ip_{thread_id}",
                10.0,
                self.base_ts + thread_id,
            )
            results.append(flags)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All 20 threads should have completed without exception
        assert len(results) == 20

        # After 20 concurrent inserts, the count in window should be exactly 20
        # (no data corruption). Verify by checking get_amount_24h consistency.
        total = self.vc.get_amount_24h("shared_card")
        assert total == pytest.approx(200.0)  # 20 threads * 10.0 each

    def test_flag_dict_has_required_keys(self):
        """Every flag dict must contain 'flag' and 'action' keys."""
        for i in range(6):
            self.vc.record_and_check("card_001", "1.2.3.4", 10.0, self.base_ts + i)
        flags = self.vc.record_and_check("card_001", "1.2.3.4", 10.0, self.base_ts + 6)
        for flag in flags:
            assert "flag" in flag
            assert "action" in flag
