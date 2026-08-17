"""Request pipeline: match stored prefixes, substitute, compact over threshold.

Clients resend the ORIGINAL history each request, so the store is keyed by
original-prefix chain hashes; compaction runs on the substituted sequence and
results are stored under the longer original prefix.

Coordinate mapping when a stored entry matched:

    substituted = msgs[:base_head] + [summary] + msgs[base_cut:]

so index sub_cut in substituted coordinates maps back to the original list
as base_cut + (sub_cut - (base_head + 1)).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from .cliff import compact
from .config import Config
from .dialects.base import Dialect
from .hashing import chain_hashes
from .store import Entry, PrefixStore

logger = logging.getLogger("cliffcompaction")


def estimate_tokens(body: dict) -> int:
    """chars/4 estimate over the full serialized request body."""
    try:
        return len(json.dumps(body, ensure_ascii=False)) // 4
    except (TypeError, ValueError):
        return 0


@dataclass
class RequestCtx:
    body: dict
    dialect: Dialect
    msgs: list[dict]
    chain: list[str]
    # Provenance of `substituted`: 0/0 means no substitution (substituted is msgs).
    base_cut: int = 0
    base_head: int = 0
    substituted: list[dict] = field(default_factory=list)
    modified: bool = False  # substitution and/or compaction happened
    compacted: bool = False  # a compaction happened during prepare()
    est_tokens_in: int = 0
    est_tokens_out: int = 0

    def outgoing_body(self) -> dict:
        if not self.modified:
            return self.body
        out = dict(self.body)
        out["messages"] = self.substituted
        return out


class Engine:
    def __init__(self, cfg: Config, store: PrefixStore | None = None):
        self.cfg = cfg
        self.store = store or PrefixStore(cfg.store_max_entries)

    # ------------------------------------------------------------- pipeline

    def prepare(self, body: dict, dialect: Dialect) -> RequestCtx:
        msgs = body["messages"]
        digests = [dialect.digest_message(m) for m in msgs]
        chain = chain_hashes(digests)
        ctx = RequestCtx(body=body, dialect=dialect, msgs=msgs, chain=chain)
        ctx.substituted = msgs
        ctx.est_tokens_in = estimate_tokens(body)

        # Longest stored prefix wins.
        for i in range(len(msgs) - 1, -1, -1):
            entry = self.store.get(chain[i])
            if entry is None:
                continue
            if entry.cut != i + 1 or entry.head_len > entry.cut:
                logger.warning("store entry inconsistent at depth %d; ignoring", i + 1)
                break
            ctx.base_cut = entry.cut
            ctx.base_head = entry.head_len
            ctx.substituted = [
                *msgs[: entry.head_len],
                entry.summary,
                *msgs[entry.cut :],
            ]
            ctx.modified = True
            logger.info(
                "match: prefix depth %d/%d (head=%d)",
                entry.cut,
                len(msgs),
                entry.head_len,
            )
            break

        # Proactive trigger on the outgoing (post-substitution) size.
        ctx.est_tokens_out = estimate_tokens(ctx.outgoing_body())
        if ctx.est_tokens_out > self.cfg.threshold_tokens:
            self._compact_chain(ctx, reason="proactive")
        return ctx

    def reactive(self, ctx: RequestCtx) -> bool:
        """Called on an upstream context-length error. Compact regardless of
        threshold; returns True if the request should be replayed."""
        if ctx.compacted:
            # Already compacted this request and it still doesn't fit.
            return False
        return self._compact_chain(ctx, reason="reactive", force=True)

    # -------------------------------------------------------------- helpers

    @staticmethod
    def _msg_chars(msg: dict) -> int:
        try:
            return len(json.dumps(msg, ensure_ascii=False)) + 2
        except (TypeError, ValueError):
            return 0

    def _compact_chain(self, ctx: RequestCtx, reason: str, force: bool = False) -> bool:
        """Compact by replaying threshold crossings over the sequence.

        Feeds messages in order, compacting whenever the running estimate
        crosses the threshold; each step drops the prior step's summary, so a
        long history arriving at once (empty store, late attach) yields the
        same last-cycle-only summary an incrementally built chain would. With
        one crossing this is a single compaction.

        With `force`, a sequence that never crosses is compacted once at the
        end (reactive path: the provider rejected it regardless of the
        estimate).
        """
        cfg = self.cfg
        msgs = ctx.msgs
        try:
            fixed_chars = len(
                json.dumps({**ctx.body, "messages": []}, ensure_ascii=False)
            )
        except (TypeError, ValueError):
            fixed_chars = 0
        threshold_chars = cfg.threshold_tokens * 4

        # Working state: working == replacement + msgs[orig_cut:fed], where
        # replacement = msgs[:head_len] + [summary] once a compaction exists.
        if ctx.base_cut > 0:
            working = list(ctx.substituted[: ctx.base_head + 1])
            head_len = ctx.base_head
            orig_cut = ctx.base_cut
            have_summary = True
        else:
            working = []
            head_len = 0
            orig_cut = 0
            have_summary = False

        chars = fixed_chars + sum(self._msg_chars(m) for m in working)
        last_summary = None
        n_compactions = 0

        def apply(result) -> bool:
            """Map result.cut to original coordinates and adopt the result."""
            nonlocal working, chars, head_len, orig_cut, have_summary
            nonlocal last_summary, n_compactions
            if have_summary:
                if result.cut < head_len + 1:
                    logger.warning(
                        "compact (%s): cut %d inside replacement; fail-open",
                        reason,
                        result.cut,
                    )
                    return False
                new_cut = orig_cut + (result.cut - (head_len + 1))
            else:
                new_cut = result.cut
            if not (1 <= new_cut <= len(msgs)):
                logger.warning("compact (%s): cut maps out of range; fail-open", reason)
                return False
            working = result.messages
            chars = fixed_chars + sum(self._msg_chars(m) for m in working)
            head_len = result.head_len
            orig_cut = new_cut
            have_summary = True
            last_summary = result.summary
            n_compactions += 1
            return True

        for i in range(orig_cut, len(msgs)):
            working.append(msgs[i])
            chars += self._msg_chars(msgs[i])
            if chars > threshold_chars:
                result = compact(working, ctx.dialect, cfg)
                if result is None:
                    continue  # not enough turns yet; keep feeding
                if not apply(result):
                    return n_compactions > 0

        if n_compactions == 0 and force:
            result = compact(working, ctx.dialect, cfg)
            if result is not None:
                apply(result)

        if n_compactions == 0:
            logger.info("compact (%s): nothing to compact", reason)
            return False

        before = ctx.est_tokens_out
        ctx.substituted = working
        ctx.base_cut = orig_cut
        ctx.base_head = head_len
        ctx.modified = True
        ctx.compacted = True
        ctx.est_tokens_out = estimate_tokens(ctx.outgoing_body())
        self.store.put(
            ctx.chain[orig_cut - 1],
            Entry(head_len=head_len, summary=last_summary, cut=orig_cut),
        )
        logger.info(
            "compact (%s): ~%dk -> ~%dk est tokens, %d -> %d messages, "
            "%d chain step(s), stored@%d",
            reason,
            before // 1000,
            ctx.est_tokens_out // 1000,
            len(msgs),
            len(working),
            n_compactions,
            orig_cut,
        )
        return True
