"""Prefix store: chain_hash(original prefix S) -> compacted replacement.

An entry holds only (head_len, summary, cut); the substitution for a request
extending prefix S (|S| = cut) is msgs[:head_len] + [summary] + msgs[cut:].
Head bytes come from the CURRENT request, so volatile fields (cache_control)
are never replayed stale. The store is a cache: the compactor is
deterministic, so lost entries are recomputed identically on demand.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class Entry:
    head_len: int
    summary: dict
    cut: int  # length of the original prefix S this entry replaces
    size: int = field(default=0, compare=False)  # serialized chars; filled by put()


def entry_size(summary: dict) -> int:
    """Serialized size of a summary, in characters.

    A proxy for its memory: CPython stores an all-ASCII str at one byte per
    character, which is what transcripts of code are. CJK or emoji content
    costs 2-4x that, so the byte budget is an approximation, not a guarantee.
    """
    try:
        return len(json.dumps(summary, ensure_ascii=False))
    except (TypeError, ValueError):
        return 0


class PrefixStore:
    """LRU cache of compacted prefixes, bounded by total size and entry count.

    Size is the binding limit in practice: one entry runs ~0.25 chars per est
    token of the threshold, so entries get bigger exactly when a bound starts
    to matter. Eviction is cheap either way — the compactor is deterministic,
    so an evicted entry is recomputed identically on demand.
    """

    def __init__(self, max_entries: int = 4096, max_bytes: int = 64 * 1024 * 1024):
        self._max = max_entries
        self._max_bytes = max_bytes
        self._data: OrderedDict[str, Entry] = OrderedDict()
        self._bytes = 0
        self._lock = Lock()

    def get(self, chain_hash: str) -> Entry | None:
        with self._lock:
            entry = self._data.get(chain_hash)
            if entry is not None:
                self._data.move_to_end(chain_hash)
            return entry

    def put(self, chain_hash: str, entry: Entry) -> None:
        entry.size = entry_size(entry.summary)
        with self._lock:
            old = self._data.pop(chain_hash, None)
            if old is not None:
                self._bytes -= old.size
            self._data[chain_hash] = entry
            self._bytes += entry.size
            # Never evict down to empty: an entry over budget on its own is
            # still the one the next request will look for.
            while len(self._data) > 1 and (
                len(self._data) > self._max or self._bytes > self._max_bytes
            ):
                _, evicted = self._data.popitem(last=False)
                self._bytes -= evicted.size

    @property
    def nbytes(self) -> int:
        return self._bytes

    def __len__(self) -> int:
        return len(self._data)
