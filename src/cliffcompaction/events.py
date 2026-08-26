"""Bounded in-memory event stream for `cliff watch`.

Emission is fire-and-forget: `emit()` never raises and never blocks, so a
watcher that stalls or dies cannot affect the request path. Events live only
in memory, capped by `maxlen`; nothing is written to disk.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import time
from collections import deque

logger = logging.getLogger("cliffcompaction")

# Per-subscriber queue depth. A watcher that falls this far behind starts
# losing events rather than growing memory without bound.
QUEUE_DEPTH = 512


class EventHub:
    def __init__(self, maxlen: int = 1024) -> None:
        self._buf: deque[dict] = deque(maxlen=maxlen)
        self._subs: set[asyncio.Queue] = set()
        self._seq = itertools.count(1)
        self.dropped = 0

    def emit(self, **fields) -> None:
        """Record an event. Never raises."""
        try:
            ev = {"seq": next(self._seq), "t": time.time(), **fields}
            self._buf.append(ev)
            for q in list(self._subs):
                try:
                    q.put_nowait(ev)
                except asyncio.QueueFull:
                    self.dropped += 1
        except Exception:  # pragma: no cover - defensive
            logger.debug("event emit failed (request unaffected)", exc_info=True)

    def backlog(self) -> list[dict]:
        return list(self._buf)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_DEPTH)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    @property
    def subscribers(self) -> int:
        return len(self._subs)


def sse(ev: dict) -> bytes:
    return b"data: " + json.dumps(ev, ensure_ascii=False).encode("utf-8") + b"\n\n"
