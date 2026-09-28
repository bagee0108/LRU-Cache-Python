"""An LRU cache built from scratch on a hash map plus a doubly linked list.

The cache lives in :mod:`lrucache.cache` and the recency list it is built on
lives in :mod:`lrucache.node`; both are re-exported here so callers can simply
``from lrucache import LRUCache``.
"""

from __future__ import annotations

from .cache import MISS, LRUCache
from .stats import CacheStats

__version__ = "1.0.0"

__all__ = ["MISS", "CacheStats", "LRUCache", "__version__"]
