"""The recency list: a doubly linked list with sentinel head and tail nodes.

This is the half of the LRU cache that maintains *order*. It deliberately knows
nothing about keys, hashing or capacity -- it only answers "what was used least
recently?" and "move this entry to the front", both in constant time.

Two design choices make every operation O(1) with no special cases:

* **Intrusive nodes.** A :class:`Node` carries its own ``prev``/``next``
  pointers, so a node that somebody already has a reference to (the cache's dict
  hands them out) can be unlinked in four pointer assignments. There is never a
  scan to find a neighbour.
* **Sentinels.** ``_head`` and ``_tail`` are permanent dummy nodes that are
  never removed and never hold data. Because the list always has them, inserting
  and unlinking never have to branch on "is the list empty?", "is this the first
  element?" or "is this the last element?" -- the neighbours always exist.

Order convention: the head end is the **most** recently used entry, the tail end
is the **least** recently used one. Eviction therefore always takes
``_tail.prev``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class Node(Generic[K, V]):
    """One cache entry, plus its position in the recency list.

    ``__slots__`` keeps nodes small and attribute access fast: at a few hundred
    thousand entries the per-object ``__dict__`` a normal class would carry is a
    meaningful share of the cache's memory.

    Args:
        key: The cache key this node holds.
        value: The cached value.
        expires_at: Deadline on the owning cache's clock after which this entry
            is stale, or None for an entry that never expires. Storing the
            absolute deadline rather than the original TTL means checking expiry
            is one comparison.
    """

    __slots__ = ("key", "value", "expires_at", "prev", "next")

    def __init__(self, key: K, value: V, expires_at: float | None = None) -> None:
        self.key = key
        self.value = value
        self.expires_at = expires_at
        self.prev: Node[K, V] | None = None
        self.next: Node[K, V] | None = None

    def __repr__(self) -> str:
        return f"Node(key={self.key!r}, value={self.value!r})"


class DoublyLinkedList(Generic[K, V]):
    """A sentinel-bounded doubly linked list ordered most- to least-recently-used.

    Every mutating method is O(1). Iteration is O(n) and is provided for tests,
    introspection and debugging rather than for the hot path.
    """

    __slots__ = ("_head", "_tail", "_size")

    def __init__(self) -> None:
        # The sentinels are structural only: their key and value slots are never
        # read, which is why it is safe to build them with None payloads even
        # though the list is otherwise typed on K and V.
        self._head: Node[K, V] = Node(None, None)  # type: ignore[arg-type]
        self._tail: Node[K, V] = Node(None, None)  # type: ignore[arg-type]
        self._head.next = self._tail
        self._tail.prev = self._head
        self._size = 0

    def __len__(self) -> int:
        """Return the number of real (non-sentinel) nodes. O(1)."""
        return self._size

    def __iter__(self) -> Iterator[Node[K, V]]:
        """Yield nodes from most- to least-recently-used. O(n)."""
        node = self._head.next
        while node is not self._tail:
            assert node is not None  # sentinels guarantee termination
            yield node
            node = node.next

    def push_front(self, node: Node[K, V]) -> None:
        """Splice ``node`` in as the most recently used entry. O(1)."""
        first = self._head.next
        assert first is not None
        node.prev = self._head
        node.next = first
        self._head.next = node
        first.prev = node
        self._size += 1

    def unlink(self, node: Node[K, V]) -> None:
        """Remove ``node`` from the list. O(1).

        The node's own pointers are cleared afterwards so a detached node cannot
        be used to walk a list it no longer belongs to, and so it does not keep
        its former neighbours alive.
        """
        prev, nxt = node.prev, node.next
        if prev is None or nxt is None:
            raise ValueError("node is not linked into a list")
        prev.next = nxt
        nxt.prev = prev
        node.prev = None
        node.next = None
        self._size -= 1

    def move_to_front(self, node: Node[K, V]) -> None:
        """Promote ``node`` to most recently used. O(1).

        A node that is already at the front is left untouched, which makes
        repeated hits on the same hot key close to free.
        """
        if self._head.next is node:
            return
        self.unlink(node)
        self.push_front(node)

    def peek_back(self) -> Node[K, V] | None:
        """Return the least recently used node without removing it. O(1)."""
        last = self._tail.prev
        if last is self._head:
            return None
        return last

    def pop_back(self) -> Node[K, V] | None:
        """Remove and return the least recently used node, or None if empty. O(1)."""
        last = self.peek_back()
        if last is None:
            return None
        self.unlink(last)
        return last

    def clear(self) -> None:
        """Drop every node. O(1) -- the detached chain is garbage collected."""
        self._head.next = self._tail
        self._tail.prev = self._head
        self._size = 0

    def check_invariants(self) -> None:
        """Raise ``AssertionError`` unless the list is internally consistent.

        Walks the chain forwards and backwards and cross-checks both traversals
        against the cached size. Used by the tests -- especially the concurrency
        ones -- to prove the pointer surgery never corrupts the structure.
        """
        if self._head.prev is not None:
            raise AssertionError("head sentinel must not have a predecessor")
        if self._tail.next is not None:
            raise AssertionError("tail sentinel must not have a successor")

        forward: list[Node[K, V]] = []
        node = self._head.next
        while node is not self._tail:
            if node is None:
                raise AssertionError("forward traversal hit None before the tail sentinel")
            if node.prev is None or node.prev.next is not node:
                raise AssertionError(f"broken back-pointer at {node!r}")
            forward.append(node)
            if len(forward) > self._size:
                raise AssertionError("forward traversal is longer than the recorded size (cycle?)")
            node = node.next

        backward: list[Node[K, V]] = []
        node = self._tail.prev
        while node is not self._head:
            if node is None:
                raise AssertionError("backward traversal hit None before the head sentinel")
            backward.append(node)
            if len(backward) > self._size:
                raise AssertionError("backward traversal is longer than the recorded size (cycle?)")
            node = node.prev

        if len(forward) != self._size:
            raise AssertionError(f"size is {self._size} but the list holds {len(forward)} nodes")
        if forward != list(reversed(backward)):
            raise AssertionError("forward and backward traversals disagree")
