"""Prefix store: chain_hash(original prefix S) -> compacted replacement.

An entry holds only (head_len, summary, cut); the substitution for a request
extending prefix S (|S| = cut) is msgs[:head_len] + [summary] + msgs[cut:].
Head bytes come from the CURRENT request, so volatile fields (cache_control)
are never replayed stale. The store is a cache: the compactor is
deterministic, so lost entries are recomputed identically on demand.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock


@dataclass
class Entry:
    head_len: int
    summary: dict
    cut: int  # length of the original prefix S this entry replaces


class PrefixStore:
    def __init__(self, max_entries: int = 4096):
        self._max = max_entries
        self._data: OrderedDict[str, Entry] = OrderedDict()
        self._lock = Lock()

    def get(self, chain_hash: str) -> Entry | None:
        with self._lock:
            entry = self._data.get(chain_hash)
            if entry is not None:
                self._data.move_to_end(chain_hash)
            return entry

    def put(self, chain_hash: str, entry: Entry) -> None:
        with self._lock:
            self._data[chain_hash] = entry
            self._data.move_to_end(chain_hash)
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)
