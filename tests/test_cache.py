"""Tests for the cache's public behaviour: get, put, delete and recency order."""

from __future__ import annotations

import unittest

from lrucache import MISS, LRUCache


class ConstructionTests(unittest.TestCase):
    def test_capacity_is_exposed(self) -> None:
        self.assertEqual(LRUCache[str, int](10).capacity, 10)

    def test_new_cache_is_empty(self) -> None:
        cache: LRUCache[str, int] = LRUCache(10)
        self.assertEqual(len(cache), 0)
        self.assertEqual(cache.keys(), [])

    def test_zero_capacity_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            LRUCache(0)

    def test_negative_capacity_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            LRUCache(-1)

    def test_non_integer_capacity_is_rejected(self) -> None:
        for bad in (1.5, "10", None, True):
            with self.subTest(capacity=bad):
                with self.assertRaises(TypeError):
                    LRUCache(bad)  # type: ignore[arg-type]

    def test_repr_reports_capacity_and_size(self) -> None:
        cache: LRUCache[str, int] = LRUCache(4)
        cache.put("a", 1)
        self.assertEqual(repr(cache), "LRUCache(capacity=4, size=1)")


class GetPutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cache: LRUCache[str, int] = LRUCache(4)

    def tearDown(self) -> None:
        self.cache._check_invariants()

    def test_put_then_get_returns_the_value(self) -> None:
        self.cache.put("a", 1)
        self.assertEqual(self.cache.get("a"), 1)

    def test_get_on_a_missing_key_returns_none_by_default(self) -> None:
        self.assertIsNone(self.cache.get("nope"))

    def test_get_on_a_missing_key_returns_the_supplied_default(self) -> None:
        self.assertEqual(self.cache.get("nope", -1), -1)

    def test_miss_sentinel_distinguishes_absent_from_stored_none(self) -> None:
        cache: LRUCache[str, int | None] = LRUCache(4)
        cache.put("present", None)

        self.assertIsNone(cache.get("present", MISS))
        self.assertIs(cache.get("absent", MISS), MISS)

    def test_overwriting_a_key_replaces_the_value_without_growing(self) -> None:
        self.cache.put("a", 1)
        self.cache.put("a", 2)

        self.assertEqual(self.cache.get("a"), 2)
        self.assertEqual(len(self.cache), 1)
        self.assertEqual(self.cache.keys(), ["a"])

    def test_len_tracks_distinct_keys(self) -> None:
        self.cache.put("a", 1)
        self.cache.put("b", 2)
        self.assertEqual(len(self.cache), 2)

    def test_contains_reports_membership(self) -> None:
        self.cache.put("a", 1)
        self.assertIn("a", self.cache)
        self.assertNotIn("b", self.cache)

    def test_keys_are_ordered_most_to_least_recently_used(self) -> None:
        for key in ("a", "b", "c"):
            self.cache.put(key, 1)

        self.assertEqual(self.cache.keys(), ["c", "b", "a"])

    def test_items_are_ordered_most_to_least_recently_used(self) -> None:
        self.cache.put("a", 1)
        self.cache.put("b", 2)

        self.assertEqual(self.cache.items(), [("b", 2), ("a", 1)])

    def test_values_of_any_type_round_trip(self) -> None:
        cache: LRUCache[str, object] = LRUCache(8)
        payloads: list[object] = [0, "", [], {}, None, False, {"nested": [1, 2]}]
        for index, payload in enumerate(payloads):
            cache.put(str(index), payload)

        for index, payload in enumerate(payloads):
            with self.subTest(payload=payload):
                self.assertEqual(cache.get(str(index), MISS), payload)


class RecencyTests(unittest.TestCase):
    """Recency order is the whole point, so it is asserted directly."""

    def setUp(self) -> None:
        self.cache: LRUCache[str, int] = LRUCache(4)

    def tearDown(self) -> None:
        self.cache._check_invariants()

    def test_get_promotes_the_entry(self) -> None:
        for key in ("a", "b", "c"):
            self.cache.put(key, 1)

        self.cache.get("a")

        self.assertEqual(self.cache.keys(), ["a", "c", "b"])

    def test_a_missed_get_does_not_reorder_anything(self) -> None:
        for key in ("a", "b"):
            self.cache.put(key, 1)

        self.cache.get("absent")

        self.assertEqual(self.cache.keys(), ["b", "a"])

    def test_overwriting_promotes_the_entry(self) -> None:
        for key in ("a", "b", "c"):
            self.cache.put(key, 1)

        self.cache.put("a", 99)

        self.assertEqual(self.cache.keys(), ["a", "c", "b"])

    def test_peek_returns_the_value_without_promoting(self) -> None:
        for key in ("a", "b", "c"):
            self.cache.put(key, 1)

        self.assertEqual(self.cache.peek("a"), 1)
        self.assertEqual(self.cache.keys(), ["c", "b", "a"])

    def test_peek_on_a_missing_key_returns_the_default(self) -> None:
        self.assertIs(self.cache.peek("absent", MISS), MISS)

    def test_contains_does_not_promote(self) -> None:
        for key in ("a", "b", "c"):
            self.cache.put(key, 1)

        self.assertIn("a", self.cache)
        self.assertEqual(self.cache.keys(), ["c", "b", "a"])

    def test_repeated_gets_on_the_front_entry_keep_it_in_front(self) -> None:
        self.cache.put("a", 1)
        self.cache.put("b", 2)

        for _ in range(3):
            self.cache.get("b")

        self.assertEqual(self.cache.keys(), ["b", "a"])
        self.assertEqual(len(self.cache), 2)


class EvictionTests(unittest.TestCase):
    """Capacity enforcement, and that the entry evicted is the right one."""

    def test_inserting_past_capacity_evicts_the_least_recently_used(self) -> None:
        cache: LRUCache[str, int] = LRUCache(3)
        for key in ("a", "b", "c"):
            cache.put(key, 1)

        cache.put("d", 1)

        self.assertNotIn("a", cache)
        self.assertEqual(cache.keys(), ["d", "c", "b"])
        self.assertEqual(len(cache), 3)
        cache._check_invariants()

    def test_a_get_protects_an_entry_from_the_next_eviction(self) -> None:
        cache: LRUCache[str, int] = LRUCache(3)
        for key in ("a", "b", "c"):
            cache.put(key, 1)

        cache.get("a")  # "a" is no longer the oldest; "b" is
        cache.put("d", 1)

        self.assertIn("a", cache)
        self.assertNotIn("b", cache)
        self.assertEqual(cache.keys(), ["d", "a", "c"])
        cache._check_invariants()

    def test_each_insert_past_capacity_evicts_exactly_one_entry(self) -> None:
        cache: LRUCache[int, int] = LRUCache(3)
        for key in range(10):
            cache.put(key, key)
            self.assertLessEqual(len(cache), 3)

        self.assertEqual(len(cache), 3)
        self.assertEqual(cache.keys(), [9, 8, 7])
        cache._check_invariants()

    def test_overwriting_at_capacity_never_evicts(self) -> None:
        cache: LRUCache[str, int] = LRUCache(2)
        cache.put("a", 1)
        cache.put("b", 2)

        cache.put("a", 99)

        self.assertEqual(len(cache), 2)
        self.assertIn("b", cache)
        self.assertEqual(cache.get("a"), 99)
        cache._check_invariants()

    def test_evicted_entry_is_reported_as_a_miss(self) -> None:
        cache: LRUCache[str, int] = LRUCache(1)
        cache.put("a", 1)
        cache.put("b", 2)

        self.assertIs(cache.get("a", MISS), MISS)

    def test_an_evicted_key_can_be_inserted_again(self) -> None:
        cache: LRUCache[str, int] = LRUCache(2)
        for key in ("a", "b", "c"):
            cache.put(key, 1)
        self.assertNotIn("a", cache)

        cache.put("a", 2)

        self.assertEqual(cache.get("a"), 2)
        self.assertEqual(len(cache), 2)
        cache._check_invariants()

    def test_peek_does_not_protect_an_entry_from_eviction(self) -> None:
        cache: LRUCache[str, int] = LRUCache(2)
        cache.put("a", 1)
        cache.put("b", 2)

        cache.peek("a")  # peek leaves recency alone, so "a" is still the victim
        cache.put("c", 3)

        self.assertNotIn("a", cache)
        cache._check_invariants()


class CapacityOneTests(unittest.TestCase):
    """Capacity 1 is the tightest edge case: every insert of a new key evicts."""

    def setUp(self) -> None:
        self.cache: LRUCache[str, int] = LRUCache(1)

    def tearDown(self) -> None:
        self.cache._check_invariants()

    def test_holds_a_single_entry(self) -> None:
        self.cache.put("a", 1)

        self.assertEqual(self.cache.get("a"), 1)
        self.assertEqual(len(self.cache), 1)

    def test_a_new_key_replaces_the_only_entry(self) -> None:
        self.cache.put("a", 1)
        self.cache.put("b", 2)

        self.assertIs(self.cache.get("a", MISS), MISS)
        self.assertEqual(self.cache.get("b"), 2)
        self.assertEqual(len(self.cache), 1)
        self.assertEqual(self.cache.keys(), ["b"])

    def test_overwriting_the_only_entry_keeps_it(self) -> None:
        self.cache.put("a", 1)
        self.cache.put("a", 2)

        self.assertEqual(self.cache.get("a"), 2)
        self.assertEqual(len(self.cache), 1)

    def test_repeated_churn_stays_consistent(self) -> None:
        for key in range(25):
            self.cache.put(str(key), key)
            self.assertEqual(len(self.cache), 1)
            self.cache._check_invariants()

        self.assertEqual(self.cache.keys(), ["24"])

    def test_get_then_replace_still_evicts_correctly(self) -> None:
        self.cache.put("a", 1)
        self.cache.get("a")
        self.cache.put("b", 2)

        self.assertEqual(self.cache.keys(), ["b"])


class DeleteAndClearTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cache: LRUCache[str, int] = LRUCache(4)

    def tearDown(self) -> None:
        self.cache._check_invariants()

    def test_delete_removes_the_entry_and_reports_true(self) -> None:
        self.cache.put("a", 1)

        self.assertTrue(self.cache.delete("a"))
        self.assertNotIn("a", self.cache)
        self.assertEqual(len(self.cache), 0)

    def test_delete_on_a_missing_key_reports_false(self) -> None:
        self.assertFalse(self.cache.delete("absent"))

    def test_delete_preserves_the_order_of_the_rest(self) -> None:
        for key in ("a", "b", "c"):
            self.cache.put(key, 1)

        self.cache.delete("b")

        self.assertEqual(self.cache.keys(), ["c", "a"])

    def test_a_deleted_key_can_be_put_back(self) -> None:
        self.cache.put("a", 1)
        self.cache.delete("a")
        self.cache.put("a", 2)

        self.assertEqual(self.cache.get("a"), 2)
        self.assertEqual(len(self.cache), 1)

    def test_clear_empties_the_cache_but_keeps_capacity(self) -> None:
        for key in ("a", "b"):
            self.cache.put(key, 1)

        self.cache.clear()

        self.assertEqual(len(self.cache), 0)
        self.assertEqual(self.cache.keys(), [])
        self.assertEqual(self.cache.capacity, 4)

    def test_cache_is_usable_after_clear(self) -> None:
        self.cache.put("a", 1)
        self.cache.clear()
        self.cache.put("b", 2)

        self.assertEqual(self.cache.get("b"), 2)
        self.assertEqual(self.cache.keys(), ["b"])


if __name__ == "__main__":
    unittest.main()
