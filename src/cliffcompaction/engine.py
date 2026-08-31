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

import dataclasses
import hashlib
import json
import logging
from dataclasses import dataclass, field

from .cliff import compact
from .config import Config
from .dialects.base import SUMMARY_HEADER, Dialect
from .hashing import chain_hashes
from .store import Entry, PrefixStore

logger = logging.getLogger("cliffcompaction")


def _summary_fingerprint(summary: dict) -> str:
    content = summary.get("content", "")
    text = content if isinstance(content, str) else json.dumps(content)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


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
    # Highest escalation rung applied to this request:
    # 0 = base config only, 1 = keep_recent=1, 2 = +thought/thinking caps,
    # 3 = summary truncated to fit (reactive only).
    rung: int = 0
    # Still over threshold once the ladder was exhausted. Always recorded;
    # only strict mode acts on it.
    over_budget: bool = False
    est_tokens_in: int = 0
    est_tokens_out: int = 0
    # Observability only (the watcher reads these; nothing branches on them).
    chain_steps: int = 0
    summary_fp: str = ""
    out_msgs: int = 0

    def outgoing_body(self) -> dict:
        if not self.modified:
            return self.body
        out = dict(self.body)
        out[self.dialect.messages_key] = self.substituted
        return out


class Engine:
    def __init__(self, cfg: Config, store: PrefixStore | None = None):
        self.cfg = cfg
        self.store = store or PrefixStore(cfg.store_max_entries)

    # ------------------------------------------------------------- pipeline

    def prepare(self, body: dict, dialect: Dialect) -> RequestCtx:
        msgs = body[dialect.messages_key]
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
                "match: prefix depth %d/%d (head=%d, summary#%s)",
                entry.cut,
                len(msgs),
                entry.head_len,
                _summary_fingerprint(entry.summary),
            )
            break

        # Proactive trigger on the outgoing (post-substitution) size.
        ctx.est_tokens_out = estimate_tokens(ctx.outgoing_body())
        if ctx.est_tokens_out > self.cfg.threshold_tokens:
            self._compact_chain(ctx, reason="proactive")
            # Escalate while still over budget: harsher knobs, cliff()
            # semantics untouched. Truncation (rung 3) is normally
            # reactive-only; proactively we stop at rung 2 and send over
            # budget (soft) — oversized content ages into the compacted
            # region next cycle. Strict mode has no next cycle worth
            # deferring to, so it walks the last rung too and then refuses.
            for rung in (1, 2, 3) if self.cfg.strict else (1, 2):
                if ctx.est_tokens_out <= self.cfg.threshold_tokens:
                    break
                if rung == 3:
                    if self._truncate_summary(ctx, reason="strict"):
                        ctx.rung = 3
                elif self._compact_chain(
                    ctx,
                    reason=f"escalated rung{rung}",
                    force=True,
                    compact_cfg=self._rung_cfg(rung),
                ):
                    ctx.rung = rung
            if ctx.est_tokens_out > self.cfg.threshold_tokens:
                ctx.over_budget = True
                if ctx.compacted:
                    logger.info(
                        "over budget after escalation (~%dk est > %dk); %s",
                        ctx.est_tokens_out // 1000,
                        self.cfg.threshold_tokens // 1000,
                        "refusing (strict)" if self.cfg.strict else "sending anyway",
                    )
        return ctx

    def reactive(self, ctx: RequestCtx) -> bool:
        """Called on an upstream context-length error. Walks the escalation
        ladder one rung per call; returns True if the request should be
        replayed, False when out of options."""
        if not ctx.compacted and ctx.rung == 0:
            if self._compact_chain(ctx, reason="reactive", force=True):
                return True
        while ctx.rung < 3:
            ctx.rung += 1
            if ctx.rung < 3:
                if self._compact_chain(
                    ctx,
                    reason=f"reactive rung{ctx.rung}",
                    force=True,
                    compact_cfg=self._rung_cfg(ctx.rung),
                ):
                    return True
            else:
                if self._truncate_summary(ctx):
                    return True
        return False

    # -------------------------------------------------------------- helpers

    def _rung_cfg(self, rung: int) -> Config:
        """Derived config for an escalation rung. Rung 1: minimal tail.
        Rung 2: additionally the lean summary knobs."""
        kw: dict = {"keep_recent": 1}
        if rung >= 2:
            cap = self.cfg.thought_max_chars
            kw["thought_max_chars"] = 300 if cap <= 0 else min(cap, 300)
            kw["keep_thinking"] = False
        return dataclasses.replace(self.cfg, **kw)

    @staticmethod
    def _summary_text(msg: dict) -> str:
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and isinstance(b.get("text"), str):
                    return b["text"]
        return ""

    def _truncate_summary(self, ctx: RequestCtx, reason: str = "reactive") -> bool:
        """Last-resort rung: shrink the existing summary to fit the threshold
        budget, keeping the NEWEST parts. Degenerates to a header-only summary
        when no budget remains. Never touches the head or the kept tail.

        Reactive path only, except in strict mode, where the proactive ladder
        walks this rung as well before refusing."""
        if not ctx.compacted or ctx.base_cut <= 0:
            return False
        idx = ctx.base_head
        old = ctx.substituted[idx]
        text = self._summary_text(old)
        if not text.startswith(SUMMARY_HEADER):
            return False
        others = [m for i, m in enumerate(ctx.substituted) if i != idx]
        try:
            fixed = len(
                json.dumps(
                    {**ctx.body, ctx.dialect.messages_key: others},
                    ensure_ascii=False,
                )
            )
        except (TypeError, ValueError):
            return False
        budget = self.cfg.threshold_tokens * 4 - fixed - len(SUMMARY_HEADER) - 64
        parts = text[len(SUMMARY_HEADER):].strip().split("\n\n---\n\n")
        kept: list[str] = []
        used = 0
        for part in reversed(parts):  # newest first
            if used + len(part) > max(budget, 0):
                break
            kept.append(part)
            used += len(part) + 9  # separator overhead
        kept.reverse()
        new_text = (
            SUMMARY_HEADER + "\n\n" + "\n\n---\n\n".join(kept)
            if kept
            else SUMMARY_HEADER
        )
        if len(new_text) >= len(text):
            return False  # nothing gained
        new_summary = ctx.dialect.user_message(new_text)
        ctx.substituted = [*ctx.substituted[:idx], new_summary, *ctx.substituted[idx + 1:]]
        ctx.modified = True
        before = ctx.est_tokens_out
        ctx.est_tokens_out = estimate_tokens(ctx.outgoing_body())
        self.store.put(
            ctx.chain[ctx.base_cut - 1],
            Entry(head_len=ctx.base_head, summary=new_summary, cut=ctx.base_cut),
        )
        logger.info(
            "%s rung3: summary truncated (%d -> %d parts, ~%dk -> ~%dk est tokens) summary#%s",
            reason,
            len(parts),
            len(kept),
            before // 1000,
            ctx.est_tokens_out // 1000,
            _summary_fingerprint(new_summary),
        )
        return True

    @staticmethod
    def _msg_chars(msg: dict) -> int:
        try:
            return len(json.dumps(msg, ensure_ascii=False)) + 2
        except (TypeError, ValueError):
            return 0

    def _compact_chain(
        self,
        ctx: RequestCtx,
        reason: str,
        force: bool = False,
        compact_cfg: Config | None = None,
    ) -> bool:
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
        knobs = compact_cfg or cfg
        msgs = ctx.msgs
        try:
            fixed_chars = len(
                json.dumps(
                    {**ctx.body, ctx.dialect.messages_key: []}, ensure_ascii=False
                )
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
                result = compact(working, ctx.dialect, knobs)
                if result is None:
                    continue  # not enough turns yet; keep feeding
                if not apply(result):
                    return n_compactions > 0

        if n_compactions == 0 and force:
            result = compact(working, ctx.dialect, knobs)
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
        ctx.chain_steps = n_compactions
        ctx.summary_fp = _summary_fingerprint(last_summary)
        ctx.out_msgs = len(working)
        self.store.put(
            ctx.chain[orig_cut - 1],
            Entry(head_len=head_len, summary=last_summary, cut=orig_cut),
        )
        logger.info(
            "compact (%s): ~%dk -> ~%dk est tokens, %d -> %d messages, "
            "%d chain step(s), stored@%d summary#%s",
            reason,
            before // 1000,
            ctx.est_tokens_out // 1000,
            len(msgs),
            len(working),
            n_compactions,
            orig_cut,
            _summary_fingerprint(last_summary),
        )
        return True
