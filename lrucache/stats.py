"""An immutable snapshot of a cache's counters."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class CacheStats:
    """Point-in-time counters for a cache.

    Returned by :meth:`lrucache.LRUCache.stats`. It is a frozen copy rather than
    a live view, so a caller that reads several fields -- to compute a ratio, or
    to serialise them -- cannot see them shift underneath it while other threads
    are working.

    Attributes:
        hits: Lookups that found a fresh entry.
        misses: Lookups that found nothing, including ones that found a stale
            entry and discarded it.
        evictions: Entries dropped to stay within capacity.
        expirations: Entries dropped because their TTL had passed.
        size: Entries held at the moment of the snapshot, stale ones included.
        capacity: The cache's configured maximum.
    """

    hits: int
    misses: int
    evictions: int
    expirations: int
    size: int
    capacity: int

    @property
    def lookups(self) -> int:
        """Total number of lookups counted: ``hits + misses``."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups that hit, in ``[0.0, 1.0]``.

        Defined as 0.0 for a cache nobody has read yet, so that callers and
        dashboards never have to special-case a division by zero.
        """
        lookups = self.lookups
        return 0.0 if lookups == 0 else self.hits / lookups

    @property
    def miss_rate(self) -> float:
        """Fraction of lookups that missed, in ``[0.0, 1.0]``."""
        return 0.0 if self.lookups == 0 else 1.0 - self.hit_rate

    def as_dict(self) -> dict[str, float]:
        """Return the counters plus the derived rates as a JSON-ready dict."""
        data: dict[str, float] = dict(asdict(self))
        data["lookups"] = self.lookups
        data["hit_rate"] = self.hit_rate
        data["miss_rate"] = self.miss_rate
        return data
