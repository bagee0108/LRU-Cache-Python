"""Tests for the hit/miss/eviction/expiration counters and the derived rates."""

from __future__ import annotations

import unittest

from lrucache import CacheStats, LRUCache

from .test_ttl import FakeClock


class CacheStatsArithmeticTests(unittest.TestCase):
    """The snapshot's derived properties, independent of any cache."""

    @staticmethod
    def make(hits: int, misses: int) -> CacheStats:
        return CacheStats(
            hits=hits, misses=misses, evictions=0, expirations=0, size=0, capacity=1
        )

    def test_lookups_is_hits_plus_misses(self) -> None:
        self.assertEqual(self.make(3, 2).lookups, 5)

    def test_hit_rate(self) -> None:
        self.assertEqual(self.make(3, 1).hit_rate, 0.75)

    def test_miss_rate_complements_hit_rate(self) -> None:
        self.assertEqual(self.make(3, 1).miss_rate, 0.25)

    def test_rates_are_zero_for_an_unused_cache(self) -> None:
        """No lookups means no division by zero, and no pretending it was 100%."""
        stats = self.make(0, 0)
        self.assertEqual(stats.lookups, 0)
        self.assertEqual(stats.hit_rate, 0.0)
        self.assertEqual(stats.miss_rate, 0.0)

    def test_all_hits_and_all_misses_are_the_rate_bounds(self) -> None:
        self.assertEqual(self.make(4, 0).hit_rate, 1.0)
        self.assertEqual(self.make(0, 4).hit_rate, 0.0)

    def test_snapshot_is_immutable(self) -> None:
        stats = self.make(1, 1)
        with self.assertRaises(Exception):
            stats.hits = 99  # type: ignore[misc]

    def test_as_dict_includes_counters_and_derived_rates(self) -> None:
        stats = CacheStats(
            hits=3, misses=1, evictions=2, expirations=1, size=4, capacity=8
        )

        self.assertEqual(
            stats.as_dict(),
            {
                "hits": 3,
                "misses": 1,
                "evictions": 2,
                "expirations": 1,
                "size": 4,
                "capacity": 8,
                "lookups": 4,
                "hit_rate": 0.75,
                "miss_rate": 0.25,
            },
        )


class CacheCounterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cache: LRUCache[str, int] = LRUCache(2)

    def test_a_new_cache_has_zeroed_counters(self) -> None:
        stats = self.cache.stats()
        self.assertEqual(
            (stats.hits, stats.misses, stats.evictions, stats.expirations), (0, 0, 0, 0)
        )
        self.assertEqual((stats.size, stats.capacity), (0, 2))

    def test_a_successful_get_counts_a_hit(self) -> None:
        self.cache.put("a", 1)
        self.cache.get("a")

        self.assertEqual(self.cache.stats().hits, 1)
        self.assertEqual(self.cache.stats().misses, 0)

    def test_a_failed_get_counts_a_miss(self) -> None:
        self.cache.get("absent")

        self.assertEqual(self.cache.stats().misses, 1)
        self.assertEqual(self.cache.stats().hits, 0)

    def test_put_alone_counts_no_lookups(self) -> None:
        self.cache.put("a", 1)
        self.cache.put("a", 2)

        self.assertEqual(self.cache.stats().lookups, 0)

    def test_eviction_is_counted(self) -> None:
        for key in ("a", "b", "c"):
            self.cache.put(key, 1)

        self.assertEqual(self.cache.stats().evictions, 1)

    def test_overwriting_counts_no_eviction(self) -> None:
        self.cache.put("a", 1)
        self.cache.put("b", 2)
        self.cache.put("a", 99)

        self.assertEqual(self.cache.stats().evictions, 0)

    def test_size_tracks_the_entry_count(self) -> None:
        self.cache.put("a", 1)
        self.assertEqual(self.cache.stats().size, 1)

    def test_hit_rate_reflects_a_mixed_workload(self) -> None:
        self.cache.put("a", 1)
        self.cache.get("a")
        self.cache.get("a")
        self.cache.get("a")
        self.cache.get("absent")

        stats = self.cache.stats()
        self.assertEqual((stats.hits, stats.misses), (3, 1))
        self.assertEqual(stats.hit_rate, 0.75)

    def test_peek_and_contains_leave_the_counters_alone(self) -> None:
        """Introspection must not distort the numbers it is there to report."""
        self.cache.put("a", 1)

        self.cache.peek("a")
        self.cache.peek("absent")
        "a" in self.cache  # noqa: B015 - exercising __contains__ for its side effects
        "absent" in self.cache  # noqa: B015

        self.assertEqual(self.cache.stats().lookups, 0)

    def test_delete_leaves_the_lookup_counters_alone(self) -> None:
        self.cache.put("a", 1)
        self.cache.delete("a")
        self.cache.delete("absent")

        self.assertEqual(self.cache.stats().lookups, 0)

    def test_stats_is_a_snapshot_not_a_live_view(self) -> None:
        self.cache.put("a", 1)
        before = self.cache.stats()

        self.cache.get("a")

        self.assertEqual(before.hits, 0)
        self.assertEqual(self.cache.stats().hits, 1)


class ExpirationCounterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.cache: LRUCache[str, int] = LRUCache(4, clock=self.clock)

    def test_reading_a_stale_entry_counts_an_expiration_and_a_miss(self) -> None:
        self.cache.put("a", 1, ttl=5)
        self.clock.advance(10)

        self.cache.get("a")

        stats = self.cache.stats()
        self.assertEqual(stats.expirations, 1)
        self.assertEqual(stats.misses, 1)
        self.assertEqual(stats.hits, 0)

    def test_expiration_is_not_counted_as_an_eviction(self) -> None:
        """Capacity pressure and staleness are different events and count separately."""
        self.cache.put("a", 1, ttl=5)
        self.clock.advance(10)

        self.cache.get("a")

        self.assertEqual(self.cache.stats().evictions, 0)

    def test_peek_counts_an_expiration_but_no_miss(self) -> None:
        self.cache.put("a", 1, ttl=5)
        self.clock.advance(10)

        self.cache.peek("a")

        stats = self.cache.stats()
        self.assertEqual(stats.expirations, 1)
        self.assertEqual(stats.lookups, 0)

    def test_a_stale_entry_is_only_counted_once(self) -> None:
        self.cache.put("a", 1, ttl=5)
        self.clock.advance(10)

        self.cache.get("a")
        self.cache.get("a")

        self.assertEqual(self.cache.stats().expirations, 1)
        self.assertEqual(self.cache.stats().misses, 2)


class ResetStatsTests(unittest.TestCase):
    def test_reset_zeroes_the_counters(self) -> None:
        cache: LRUCache[str, int] = LRUCache(1)
        cache.put("a", 1)
        cache.get("a")
        cache.get("absent")
        cache.put("b", 2)

        cache.reset_stats()

        stats = cache.stats()
        self.assertEqual(
            (stats.hits, stats.misses, stats.evictions, stats.expirations), (0, 0, 0, 0)
        )

    def test_reset_keeps_the_cached_entries(self) -> None:
        cache: LRUCache[str, int] = LRUCache(2)
        cache.put("a", 1)

        cache.reset_stats()

        self.assertEqual(cache.get("a"), 1)
        self.assertEqual(cache.stats().size, 1)

    def test_clear_does_not_reset_the_counters(self) -> None:
        cache: LRUCache[str, int] = LRUCache(2)
        cache.put("a", 1)
        cache.get("a")

        cache.clear()

        stats = cache.stats()
        self.assertEqual(stats.hits, 1)
        self.assertEqual(stats.size, 0)

    def test_clear_counts_no_evictions(self) -> None:
        cache: LRUCache[str, int] = LRUCache(2)
        cache.put("a", 1)
        cache.put("b", 2)

        cache.clear()

        self.assertEqual(cache.stats().evictions, 0)


if __name__ == "__main__":
    unittest.main()
