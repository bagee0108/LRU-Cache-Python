"""The LRU cache itself: a hash map paired with a recency list.

Why this pair is O(1)
---------------------

Each structure covers the other's blind spot:

* The ``dict`` answers *"where does this key live?"* in O(1), but it knows
  nothing about recency.
* The :class:`~lrucache.node.DoublyLinkedList` maintains recency order, but
  answering *"where is this key?"* by walking it would be O(n).

Together they are constant time: the dict hands back the **node object itself**,
and because that node carries its own ``prev``/``next`` pointers, promoting it to
the most-recently-used end is a handful of pointer assignments. Nothing is
scanned and nothing is shifted. For contrast, a Python ``list`` would cost O(n)
to move an element, and a *singly* linked list would cost O(n) to find the
predecessor needed to unlink a node.

"O(1)" here means amortised constant time: dict lookups can occasionally trigger
a resize, and these are ordinary interpreted Python operations, not cheap ones.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Generic, TypeVar, overload

from .node import DoublyLinkedList, Node
from .stats import CacheStats

K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")


class _Miss:
    """The type of :data:`MISS`."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "MISS"

    def __bool__(self) -> bool:
        return False


#: Sentinel meaning "this key is not cached".
#:
#: ``None`` is a perfectly legitimate thing to cache, so code that has to tell
#: *absent* apart from *present and None* passes this as the default::
#:
#:     if cache.get(key, MISS) is MISS:
#:         ...  # genuinely not cached
MISS = _Miss()


class LRUCache(Generic[K, V]):
    """A capacity-bounded cache that evicts the least recently used entry.

    Both :meth:`get` and :meth:`put` are O(1), eviction included -- the victim is
    always the node at the tail of the recency list, so it is never searched for.
    Recency is updated on every successful :meth:`get` and on every :meth:`put`,
    including one that overwrites an existing key.

    Entries may optionally carry a TTL. Expiry is **lazy**: a stale entry is
    detected and dropped when something reads it, which keeps reads O(1) but
    means an expired entry still occupies a slot until it is read or evicted.
    There is no background reaper.

    The cache is thread-safe: every public operation holds a single reentrant
    lock for its whole duration, so the map, the list and the counters can never
    be seen mid-update. The honest trade-off is that one lock **serialises** all
    access -- this protects correctness, it does not buy parallelism, and
    throughput does not improve with more cores. See the benchmark's
    ``--threads`` mode.

    :meth:`keys` and :meth:`items` are O(n) and exist for debugging and tests
    rather than for the hot path.
    """

    __slots__ = (
        "_capacity",
        "_default_ttl",
        "_clock",
        "_lock",
        "_map",
        "_order",
        "_hits",
        "_misses",
        "_evictions",
        "_expirations",
    )

    def __init__(
        self,
        capacity: int,
        *,
        default_ttl: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create a cache holding at most ``capacity`` entries.

        Args:
            capacity: Maximum number of entries. Must be a positive integer.
            default_ttl: Seconds an entry stays fresh when :meth:`put` is not
                given an explicit ``ttl``. None means entries never expire.
            clock: Source of monotonically increasing seconds. Defaults to
                :func:`time.monotonic`, so that changes to the wall clock cannot
                make entries expire early or late. Tests inject a fake clock here
                to exercise expiry without sleeping.

        Raises:
            TypeError: If ``capacity`` is not an integer.
            ValueError: If ``capacity`` is less than 1, or ``default_ttl`` is not
                greater than zero.
        """
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError(f"capacity must be an int, got {type(capacity).__name__}")
        if capacity < 1:
            raise ValueError(f"capacity must be at least 1, got {capacity}")
        if default_ttl is not None and default_ttl <= 0:
            raise ValueError(f"default_ttl must be greater than 0, got {default_ttl}")

        self._capacity = capacity
        self._default_ttl = default_ttl
        self._clock = clock

        # Reentrant so that a composite operation built out of the public methods
        # cannot deadlock against itself. An RLock costs a little more than a
        # plain Lock; that is a deliberate trade of throughput for safety.
        self._lock = threading.RLock()

        self._map: dict[K, Node[K, V]] = {}
        self._order: DoublyLinkedList[K, V] = DoublyLinkedList()

        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

    @property
    def capacity(self) -> int:
        """The maximum number of entries this cache will hold."""
        return self._capacity

    @property
    def default_ttl(self) -> float | None:
        """Seconds entries stay fresh by default, or None if they never expire."""
        return self._default_ttl

    def _is_expired(self, node: Node[K, V]) -> bool:
        """Report whether ``node``'s deadline has passed. O(1), non-mutating."""
        return node.expires_at is not None and self._clock() >= node.expires_at

    def _drop(self, node: Node[K, V]) -> None:
        """Remove a stale ``node`` from both the map and the recency list. O(1)."""
        del self._map[node.key]
        self._order.unlink(node)
        self._expirations += 1

    def __len__(self) -> int:
        """Return the number of entries currently cached. O(1)."""
        with self._lock:
            return len(self._map)

    def __contains__(self, key: object) -> bool:
        """Report whether ``key`` is cached *and* fresh, without touching recency. O(1).

        Deliberately does not count as a hit or a miss, does not promote the
        entry, and does not reap an entry it finds expired, so that introspecting
        a cache cannot change its behaviour at all.
        """
        with self._lock:
            node = self._map.get(key)  # type: ignore[arg-type]
            return node is not None and not self._is_expired(node)

    @overload
    def get(self, key: K) -> V | None: ...

    @overload
    def get(self, key: K, default: T) -> V | T: ...

    def get(self, key: K, default: object = None) -> object:
        """Return the value for ``key``, promoting it to most recently used. O(1).

        An entry whose TTL has passed is treated as a miss and dropped on the
        spot -- this lazy reaping is the only thing that frees expired slots.

        Args:
            key: The key to look up.
            default: Returned when ``key`` is not cached. Pass :data:`MISS` to
                distinguish an absent key from a stored ``None``.

        Returns:
            The cached value, or ``default`` on a miss.
        """
        with self._lock:
            node = self._map.get(key)
            if node is None:
                self._misses += 1
                return default
            if self._is_expired(node):
                self._drop(node)
                self._misses += 1
                return default
            self._order.move_to_front(node)
            self._hits += 1
            return node.value

    def peek(self, key: K, default: object = None) -> object:
        """Return the value for ``key`` without reordering live entries. O(1).

        Useful for inspecting a cache -- in tests, or over an admin endpoint --
        when the read itself should not change what gets evicted next. Note that
        an expired entry is still reaped here: what ``peek`` leaves untouched is
        the recency *order* of fresh entries, not stale data.

        Does not count as a hit or a miss, so polling a cache cannot distort its
        hit rate. A stale entry it cleans up is still counted as an expiration,
        since that counter tracks entries actually dropped for staleness.
        """
        with self._lock:
            node = self._map.get(key)
            if node is None:
                return default
            if self._is_expired(node):
                self._drop(node)
                return default
            return node.value

    def put(self, key: K, value: V, ttl: float | None = None) -> None:
        """Insert or overwrite ``key``, making it most recently used. O(1).

        Overwriting an existing key updates its value in place, refreshes its
        TTL and promotes it; it does not change the number of entries and so can
        never evict.

        Inserting a *new* key when the cache is already full evicts exactly one
        entry: the least recently used one.

        Args:
            key: The key to store under.
            value: The value to store. ``None`` is a valid value.
            ttl: Seconds this entry stays fresh. None falls back to the cache's
                ``default_ttl``. To store an entry permanently in a cache that
                has a default TTL, pass ``float("inf")``.

        Raises:
            ValueError: If ``ttl`` is not greater than zero.
        """
        if ttl is not None and ttl <= 0:
            raise ValueError(f"ttl must be greater than 0, got {ttl}")

        with self._lock:
            effective_ttl = self._default_ttl if ttl is None else ttl
            expires_at = None if effective_ttl is None else self._clock() + effective_ttl

            node = self._map.get(key)
            if node is not None:
                node.value = value
                node.expires_at = expires_at
                self._order.move_to_front(node)
                return

            node = Node(key, value, expires_at)
            self._map[key] = node
            self._order.push_front(node)

            if len(self._map) > self._capacity:
                self._evict_lru()

    def _evict_lru(self) -> None:
        """Drop the least recently used entry. O(1).

        The victim is whatever sits at the tail end of the recency list, so
        eviction never searches for it.
        """
        victim = self._order.pop_back()
        if victim is None:  # pragma: no cover - capacity >= 1 keeps this unreachable
            return
        del self._map[victim.key]
        self._evictions += 1

    def delete(self, key: K) -> bool:
        """Remove ``key`` if present. O(1).

        Returns:
            True if a fresh entry was removed. False if the key was not cached at
            all, or was already expired -- an expired entry is logically absent,
            so callers are told the same thing :meth:`get` would tell them, even
            though the stale node is cleaned up either way.
        """
        with self._lock:
            node = self._map.pop(key, None)
            if node is None:
                return False
            was_fresh = not self._is_expired(node)
            self._order.unlink(node)
            return was_fresh

    def clear(self) -> None:
        """Drop every entry.

        Leaves the configured capacity and the counters alone: emptying a cache
        on purpose is neither an eviction nor an expiration, and erasing the
        history of how it performed would be misleading. Use :meth:`reset_stats`
        for that.
        """
        with self._lock:
            self._map.clear()
            self._order.clear()

    def stats(self) -> CacheStats:
        """Return an immutable snapshot of the counters. O(1).

        Taken under the lock, so the counters are mutually consistent even while
        other threads are working -- the hit rate computed from them is a rate
        that really occurred, not a mix of two different moments.

        ``size`` counts every entry held, including any that are expired but have
        not been read since, which is the honest figure for how much of the
        capacity is in use.
        """
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                expirations=self._expirations,
                size=len(self._map),
                capacity=self._capacity,
            )

    def reset_stats(self) -> None:
        """Zero the counters without touching the cached entries.

        Useful for measuring a warm cache: fill it, reset, then benchmark.
        """
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._expirations = 0

    def keys(self) -> list[K]:
        """Return fresh keys from most- to least-recently-used. O(n).

        Expired entries are filtered out but not reaped, keeping this read-only.
        A materialised list rather than a generator, so the whole traversal
        happens inside the lock and cannot observe a half-finished update.
        """
        with self._lock:
            return [node.key for node in self._order if not self._is_expired(node)]

    def items(self) -> list[tuple[K, V]]:
        """Return fresh ``(key, value)`` pairs from most- to least-recently-used. O(n)."""
        with self._lock:
            return [
                (node.key, node.value) for node in self._order if not self._is_expired(node)
            ]

    def __repr__(self) -> str:
        with self._lock:
            return f"{type(self).__name__}(capacity={self._capacity}, size={len(self._map)})"

    def _check_invariants(self) -> None:
        """Raise ``AssertionError`` unless the map and the list agree.

        Test-only helper. The two structures are redundant representations of the
        same entries, so any divergence between them is a bug.
        """
        with self._lock:
            self._order.check_invariants()

            if len(self._order) != len(self._map):
                raise AssertionError(
                    f"map holds {len(self._map)} entries but the list holds {len(self._order)}"
                )

            seen: set[K] = set()
            for node in self._order:
                if node.key in seen:
                    raise AssertionError(f"key {node.key!r} appears twice in the list")
                seen.add(node.key)
                mapped = self._map.get(node.key)
                if mapped is None:
                    raise AssertionError(f"list node {node!r} is missing from the map")
                if mapped is not node:
                    raise AssertionError(
                        f"map and list hold different nodes for key {node.key!r}"
                    )

            if len(self._map) > self._capacity:
                raise AssertionError(
                    f"cache holds {len(self._map)} entries, over capacity {self._capacity}"
                )
