"""Tests for per-entry TTL expiry.

Every test here drives a fake clock rather than calling ``time.sleep``. That
keeps the suite fast and, more importantly, deterministic: expiry is tested at
exact boundaries instead of hoping a sleep was long enough.
"""

from __future__ import annotations

import unittest

from lrucache import MISS, LRUCache


class FakeClock:
    """A manually advanced stand-in for :func:`time.monotonic`."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TtlConfigurationTests(unittest.TestCase):
    def test_default_ttl_defaults_to_none(self) -> None:
        self.assertIsNone(LRUCache[str, int](4).default_ttl)

    def test_default_ttl_is_exposed(self) -> None:
        self.assertEqual(LRUCache[str, int](4, default_ttl=30).default_ttl, 30)

    def test_non_positive_default_ttl_is_rejected(self) -> None:
        for bad in (0, -1):
            with self.subTest(default_ttl=bad):
                with self.assertRaises(ValueError):
                    LRUCache(4, default_ttl=bad)

    def test_non_positive_put_ttl_is_rejected(self) -> None:
        cache: LRUCache[str, int] = LRUCache(4)
        for bad in (0, -5):
            with self.subTest(ttl=bad):
                with self.assertRaises(ValueError):
                    cache.put("a", 1, ttl=bad)

    def test_a_rejected_put_stores_nothing(self) -> None:
        cache: LRUCache[str, int] = LRUCache(4)
        with self.assertRaises(ValueError):
            cache.put("a", 1, ttl=0)

        self.assertEqual(len(cache), 0)


class ExpiryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.cache: LRUCache[str, int] = LRUCache(4, clock=self.clock)

    def tearDown(self) -> None:
        self.cache._check_invariants()

    def test_entry_is_fresh_before_its_deadline(self) -> None:
        self.cache.put("a", 1, ttl=10)
        self.clock.advance(9.99)

        self.assertEqual(self.cache.get("a"), 1)

    def test_entry_expires_exactly_at_its_deadline(self) -> None:
        self.cache.put("a", 1, ttl=10)
        self.clock.advance(10)

        self.assertIs(self.cache.get("a", MISS), MISS)

    def test_entry_expires_after_its_deadline(self) -> None:
        self.cache.put("a", 1, ttl=10)
        self.clock.advance(11)

        self.assertIs(self.cache.get("a", MISS), MISS)

    def test_entries_without_a_ttl_never_expire(self) -> None:
        self.cache.put("a", 1)
        self.clock.advance(10_000_000)

        self.assertEqual(self.cache.get("a"), 1)

    def test_reading_an_expired_entry_frees_its_slot(self) -> None:
        self.cache.put("a", 1, ttl=10)
        self.clock.advance(10)

        self.cache.get("a")

        self.assertEqual(len(self.cache), 0)
        self.assertEqual(self.cache.keys(), [])

    def test_peek_also_reports_and_reaps_an_expired_entry(self) -> None:
        self.cache.put("a", 1, ttl=10)
        self.clock.advance(10)

        self.assertIs(self.cache.peek("a", MISS), MISS)
        self.assertEqual(len(self.cache), 0)

    def test_contains_reports_an_expired_entry_as_absent(self) -> None:
        self.cache.put("a", 1, ttl=10)
        self.clock.advance(10)

        self.assertNotIn("a", self.cache)

    def test_contains_does_not_reap(self) -> None:
        """``in`` is pure introspection: it must not mutate the cache at all."""
        self.cache.put("a", 1, ttl=10)
        self.clock.advance(10)

        self.assertNotIn("a", self.cache)
        self.assertEqual(len(self.cache), 1)  # the stale node is still occupying a slot

    def test_keys_and_items_hide_expired_entries(self) -> None:
        self.cache.put("stale", 1, ttl=5)
        self.cache.put("fresh", 2, ttl=100)
        self.clock.advance(10)

        self.assertEqual(self.cache.keys(), ["fresh"])
        self.assertEqual(self.cache.items(), [("fresh", 2)])

    def test_delete_reports_an_expired_entry_as_absent(self) -> None:
        self.cache.put("a", 1, ttl=10)
        self.clock.advance(10)

        self.assertFalse(self.cache.delete("a"))
        self.assertEqual(len(self.cache), 0)

    def test_only_expired_entries_are_dropped(self) -> None:
        self.cache.put("short", 1, ttl=5)
        self.cache.put("long", 2, ttl=100)
        self.clock.advance(10)

        self.assertIs(self.cache.get("short", MISS), MISS)
        self.assertEqual(self.cache.get("long"), 2)
        self.assertEqual(len(self.cache), 1)


class TtlRefreshTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.cache: LRUCache[str, int] = LRUCache(4, clock=self.clock)

    def tearDown(self) -> None:
        self.cache._check_invariants()

    def test_overwriting_restarts_the_ttl(self) -> None:
        self.cache.put("a", 1, ttl=10)
        self.clock.advance(9)

        self.cache.put("a", 2, ttl=10)
        self.clock.advance(9)  # 18s after the first put, 9s after the second

        self.assertEqual(self.cache.get("a"), 2)

    def test_a_get_does_not_extend_the_ttl(self) -> None:
        """Recency and freshness are independent: reading does not renew a lease."""
        self.cache.put("a", 1, ttl=10)
        self.clock.advance(9)
        self.assertEqual(self.cache.get("a"), 1)

        self.clock.advance(1)

        self.assertIs(self.cache.get("a", MISS), MISS)

    def test_overwriting_can_clear_a_ttl(self) -> None:
        self.cache.put("a", 1, ttl=10)
        self.cache.put("a", 2)  # no TTL, and no cache default
        self.clock.advance(1000)

        self.assertEqual(self.cache.get("a"), 2)


class DefaultTtlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.cache: LRUCache[str, int] = LRUCache(4, default_ttl=10, clock=self.clock)

    def tearDown(self) -> None:
        self.cache._check_invariants()

    def test_put_without_a_ttl_inherits_the_default(self) -> None:
        self.cache.put("a", 1)
        self.clock.advance(10)

        self.assertIs(self.cache.get("a", MISS), MISS)

    def test_an_explicit_ttl_overrides_the_default(self) -> None:
        self.cache.put("a", 1, ttl=100)
        self.clock.advance(50)

        self.assertEqual(self.cache.get("a"), 1)

    def test_an_infinite_ttl_opts_out_of_the_default(self) -> None:
        self.cache.put("a", 1, ttl=float("inf"))
        self.clock.advance(10_000_000)

        self.assertEqual(self.cache.get("a"), 1)


class TtlAndEvictionTests(unittest.TestCase):
    """TTL and capacity are separate limits; these tests pin down how they interact."""

    def setUp(self) -> None:
        self.clock = FakeClock()

    def test_an_expired_entry_still_occupies_a_slot_until_touched(self) -> None:
        """The documented cost of lazy expiry, asserted so it cannot change silently.

        ``live`` is older than ``stale``, so it is the eviction victim even though
        ``stale`` is dead and its slot is pure waste. Nothing reclaims that slot
        until somebody reads the key.
        """
        cache: LRUCache[str, int] = LRUCache(2, clock=self.clock)
        cache.put("live", 1)  # no TTL, and the least recently used entry
        cache.put("stale", 2, ttl=5)
        self.clock.advance(10)

        cache.put("c", 3)

        self.assertIn("c", cache)
        self.assertNotIn("live", cache)  # evicted, despite "stale" being dead
        self.assertEqual(len(cache), 2)  # "c" plus the stale node nobody has read
        cache._check_invariants()

    def test_an_expired_entry_is_a_normal_eviction_candidate(self) -> None:
        cache: LRUCache[str, int] = LRUCache(2, clock=self.clock)
        cache.put("stale", 1, ttl=5)
        cache.put("b", 2)
        self.clock.advance(10)

        cache.put("c", 3)
        cache.put("d", 4)

        self.assertNotIn("stale", cache)
        self.assertEqual(len(cache), 2)
        cache._check_invariants()

    def test_reaping_on_read_makes_room_without_evicting(self) -> None:
        cache: LRUCache[str, int] = LRUCache(2, clock=self.clock)
        cache.put("a", 1, ttl=5)
        cache.put("b", 2, ttl=5)
        self.clock.advance(10)

        cache.get("a")  # reaps "a"
        cache.put("c", 3)

        self.assertIn("c", cache)
        self.assertEqual(len(cache), 2)  # "c" plus the still-unread stale "b"
        cache._check_invariants()


if __name__ == "__main__":
    unittest.main()
