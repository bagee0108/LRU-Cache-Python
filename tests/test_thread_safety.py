"""Concurrency tests.

A data structure built out of hand-maintained pointers is exactly the kind of
thing that corrupts silently under concurrent mutation: a torn splice leaves a
list that still *looks* fine from one direction. So rather than asserting on
timing-dependent values, these tests hammer the cache from many threads and then
assert the structural invariants -- the map and the list agreeing, both
traversals agreeing, the size being right -- which is what would break.

Note that this proves the cache is *correct* under concurrency, not that it is
fast under concurrency. A single lock serialises everything; see the README.
"""

from __future__ import annotations

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

from lrucache import LRUCache

THREADS = 8
OPS_PER_THREAD = 2_000


def run_concurrently(worker: object, threads: int = THREADS) -> list[object]:
    """Run ``worker(index)`` on ``threads`` threads and return the results.

    Exceptions raised inside a worker surface when the result is read, so a
    thread that blows up fails the test rather than vanishing.
    """
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [pool.submit(worker, index) for index in range(threads)]  # type: ignore[arg-type,operator]
        return [future.result() for future in futures]


class ConcurrentMutationTests(unittest.TestCase):
    def test_concurrent_puts_over_capacity_keep_the_structure_consistent(self) -> None:
        cache: LRUCache[str, int] = LRUCache(64)

        def worker(index: int) -> None:
            for op in range(OPS_PER_THREAD):
                cache.put(f"key-{op % 500}", index)

        run_concurrently(worker)

        cache._check_invariants()
        self.assertEqual(len(cache), 64)

    def test_concurrent_mixed_operations_keep_the_structure_consistent(self) -> None:
        """Reads promote and writes evict, so this is the worst case for the list."""
        cache: LRUCache[str, int] = LRUCache(32)

        def worker(index: int) -> None:
            for op in range(OPS_PER_THREAD):
                key = f"key-{op % 200}"
                if op % 4 == 0:
                    cache.put(key, index)
                elif op % 4 == 1:
                    cache.get(key)
                elif op % 4 == 2:
                    cache.delete(key)
                else:
                    cache.peek(key)

        run_concurrently(worker)

        cache._check_invariants()
        self.assertLessEqual(len(cache), 32)

    def test_capacity_is_never_exceeded_while_threads_are_running(self) -> None:
        """An observer thread checks the bound holds *during* the storm, not just after."""
        cache: LRUCache[str, int] = LRUCache(16)
        stop = threading.Event()
        overshoots: list[int] = []

        def writer(index: int) -> None:
            for op in range(OPS_PER_THREAD):
                cache.put(f"key-{index}-{op}", op)

        def observer() -> None:
            while not stop.is_set():
                size = len(cache)
                if size > 16:
                    overshoots.append(size)

        watcher = threading.Thread(target=observer, daemon=True)
        watcher.start()
        try:
            run_concurrently(writer)
        finally:
            stop.set()
            watcher.join(timeout=5)

        self.assertEqual(overshoots, [])
        cache._check_invariants()

    def test_concurrent_clear_and_put_do_not_corrupt_the_list(self) -> None:
        cache: LRUCache[str, int] = LRUCache(32)

        def worker(index: int) -> None:
            for op in range(OPS_PER_THREAD // 4):
                if index == 0 and op % 50 == 0:
                    cache.clear()
                else:
                    cache.put(f"key-{op % 100}", op)

        run_concurrently(worker)

        cache._check_invariants()


class ConcurrentCounterTests(unittest.TestCase):
    def test_no_lookup_is_lost_or_double_counted(self) -> None:
        """hits + misses must equal the number of get() calls made, exactly."""
        cache: LRUCache[int, int] = LRUCache(128)
        for key in range(128):
            cache.put(key, key)
        cache.reset_stats()

        def worker(index: int) -> None:
            for op in range(OPS_PER_THREAD):
                cache.get(op % 256)  # half the keyspace is cached, half is not

        run_concurrently(worker)

        stats = cache.stats()
        self.assertEqual(stats.lookups, THREADS * OPS_PER_THREAD)
        self.assertGreater(stats.hits, 0)
        self.assertGreater(stats.misses, 0)

    def test_every_eviction_is_counted(self) -> None:
        """Each insert of a new key past capacity evicts one entry, so the books balance."""
        capacity = 50
        cache: LRUCache[str, int] = LRUCache(capacity)
        distinct_keys = THREADS * 300

        def worker(index: int) -> None:
            for op in range(300):
                cache.put(f"key-{index}-{op}", op)

        run_concurrently(worker)

        stats = cache.stats()
        self.assertEqual(stats.evictions, distinct_keys - capacity)
        self.assertEqual(stats.size, capacity)
        cache._check_invariants()


class ConcurrentReaderTests(unittest.TestCase):
    def test_readers_always_see_a_complete_value(self) -> None:
        """A reader must never observe a key paired with another key's value."""
        cache: LRUCache[str, str] = LRUCache(16)
        for index in range(16):
            cache.put(f"key-{index}", f"value-{index}")

        mismatches: list[tuple[str, object]] = []

        def worker(index: int) -> None:
            for op in range(OPS_PER_THREAD):
                key = f"key-{op % 16}"
                value = cache.get(key)
                if value is not None and value != key.replace("key", "value"):
                    mismatches.append((key, value))

        run_concurrently(worker)

        self.assertEqual(mismatches, [])

    def test_stats_snapshots_are_internally_consistent(self) -> None:
        """hit_rate is computed from one snapshot, so it can never exceed 1.0."""
        cache: LRUCache[int, int] = LRUCache(64)
        stop = threading.Event()
        bad: list[float] = []

        def reader() -> None:
            while not stop.is_set():
                stats = cache.stats()
                if not 0.0 <= stats.hit_rate <= 1.0:
                    bad.append(stats.hit_rate)
                if stats.hits + stats.misses != stats.lookups:
                    bad.append(stats.hit_rate)

        def worker(index: int) -> None:
            for op in range(OPS_PER_THREAD):
                cache.put(op % 128, op)
                cache.get(op % 256)

        watcher = threading.Thread(target=reader, daemon=True)
        watcher.start()
        try:
            run_concurrently(worker)
        finally:
            stop.set()
            watcher.join(timeout=5)

        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
