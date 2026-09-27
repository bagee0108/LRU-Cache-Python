"""Tests for the recency list in isolation, with no cache around it.

The pointer surgery is the part of the cache that is easiest to get subtly wrong
and hardest to debug through the public API, so it gets its own suite.
"""

from __future__ import annotations

import unittest

from lrucache.node import DoublyLinkedList, Node


def keys_mru_to_lru(dll: DoublyLinkedList[str, int]) -> list[str]:
    """Return the list's keys in most- to least-recently-used order."""
    return [node.key for node in dll]


class DoublyLinkedListTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dll: DoublyLinkedList[str, int] = DoublyLinkedList()

    def tearDown(self) -> None:
        # Every test leaves the structure consistent, whatever it did to it.
        self.dll.check_invariants()

    def test_starts_empty(self) -> None:
        self.assertEqual(len(self.dll), 0)
        self.assertEqual(keys_mru_to_lru(self.dll), [])
        self.assertIsNone(self.dll.peek_back())
        self.assertIsNone(self.dll.pop_back())

    def test_push_front_orders_newest_first(self) -> None:
        for key in ("a", "b", "c"):
            self.dll.push_front(Node(key, 1))

        self.assertEqual(len(self.dll), 3)
        self.assertEqual(keys_mru_to_lru(self.dll), ["c", "b", "a"])

    def test_peek_back_returns_lru_without_removing_it(self) -> None:
        for key in ("a", "b"):
            self.dll.push_front(Node(key, 1))

        oldest = self.dll.peek_back()
        assert oldest is not None
        self.assertEqual(oldest.key, "a")
        self.assertEqual(len(self.dll), 2)

    def test_pop_back_removes_from_the_lru_end(self) -> None:
        for key in ("a", "b", "c"):
            self.dll.push_front(Node(key, 1))

        popped = self.dll.pop_back()
        assert popped is not None
        self.assertEqual(popped.key, "a")
        self.assertEqual(keys_mru_to_lru(self.dll), ["c", "b"])
        self.assertEqual(len(self.dll), 2)

    def test_unlink_middle_node(self) -> None:
        nodes = {key: Node(key, 1) for key in ("a", "b", "c")}
        for key in ("a", "b", "c"):
            self.dll.push_front(nodes[key])

        self.dll.unlink(nodes["b"])

        self.assertEqual(keys_mru_to_lru(self.dll), ["c", "a"])
        self.assertEqual(len(self.dll), 2)

    def test_unlink_clears_the_detached_nodes_pointers(self) -> None:
        node = Node("a", 1)
        self.dll.push_front(node)

        self.dll.unlink(node)

        self.assertIsNone(node.prev)
        self.assertIsNone(node.next)

    def test_unlink_rejects_a_detached_node(self) -> None:
        with self.assertRaises(ValueError):
            self.dll.unlink(Node("orphan", 1))

    def test_move_to_front_promotes_the_lru_node(self) -> None:
        nodes = {key: Node(key, 1) for key in ("a", "b", "c")}
        for key in ("a", "b", "c"):
            self.dll.push_front(nodes[key])

        self.dll.move_to_front(nodes["a"])

        self.assertEqual(keys_mru_to_lru(self.dll), ["a", "c", "b"])
        self.assertEqual(len(self.dll), 3)

    def test_move_to_front_is_a_noop_for_the_current_front(self) -> None:
        nodes = {key: Node(key, 1) for key in ("a", "b")}
        for key in ("a", "b"):
            self.dll.push_front(nodes[key])

        self.dll.move_to_front(nodes["b"])

        self.assertEqual(keys_mru_to_lru(self.dll), ["b", "a"])
        self.assertEqual(len(self.dll), 2)

    def test_move_to_front_preserves_size_on_a_single_node_list(self) -> None:
        node = Node("only", 1)
        self.dll.push_front(node)

        self.dll.move_to_front(node)

        self.assertEqual(len(self.dll), 1)
        self.assertEqual(keys_mru_to_lru(self.dll), ["only"])

    def test_clear_empties_the_list(self) -> None:
        for key in ("a", "b", "c"):
            self.dll.push_front(Node(key, 1))

        self.dll.clear()

        self.assertEqual(len(self.dll), 0)
        self.assertEqual(keys_mru_to_lru(self.dll), [])

    def test_drain_to_empty_then_refill(self) -> None:
        for key in ("a", "b"):
            self.dll.push_front(Node(key, 1))
        self.dll.pop_back()
        self.dll.pop_back()

        self.assertEqual(len(self.dll), 0)
        self.dll.check_invariants()

        self.dll.push_front(Node("c", 1))
        self.assertEqual(keys_mru_to_lru(self.dll), ["c"])

    def test_check_invariants_catches_a_corrupted_size(self) -> None:
        self.dll.push_front(Node("a", 1))
        self.dll._size = 99  # deliberately corrupt the structure

        with self.assertRaises(AssertionError):
            self.dll.check_invariants()

        self.dll._size = 1  # restore so tearDown's check passes


class NodeTests(unittest.TestCase):
    def test_new_node_is_detached(self) -> None:
        node = Node("a", 1)
        self.assertIsNone(node.prev)
        self.assertIsNone(node.next)

    def test_repr_shows_key_and_value(self) -> None:
        self.assertEqual(repr(Node("a", 1)), "Node(key='a', value=1)")

    def test_nodes_use_slots_and_reject_stray_attributes(self) -> None:
        node = Node("a", 1)
        with self.assertRaises(AttributeError):
            node.unexpected = "x"  # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main()
