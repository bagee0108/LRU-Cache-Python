"""An LRU cache built from scratch on a hash map plus a doubly linked list.

The public surface is assembled in :mod:`lrucache.cache`; this module re-exports
it so callers can simply ``from lrucache import LRUCache``.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
