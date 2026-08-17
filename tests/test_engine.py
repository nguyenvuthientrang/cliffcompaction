"""Engine pipeline: match -> substitute -> compact, across a growing session."""

from cliffcompaction.config import Config
from cliffcompaction.dialects.anthropic import DIALECT as ANTHROPIC
from cliffcompaction.dialects.base import SUMMARY_HEADER
from cliffcompaction.engine import Engine

from util import a_assistant, a_body, a_result, a_session


def grow(msgs: list[dict], start: int, n: int, result_chars: int = 2000) -> list[dict]:
    out = list(msgs)
    for i in range(start, start + n):
        tid = f"tu_{i}"
        out.append(a_assistant(f"Step {i}", (tid, "bash", {"command": f"cmd {i}"})))
        out.append(a_result(tid, "R" * result_chars))
    return out


def n_summaries(msgs: list[dict]) -> int:
    return sum(
        1
        for m in msgs
        if m.get("role") == "user"
        and isinstance(m.get("content"), str)
        and m["content"].startswith(SUMMARY_HEADER)
    )


def test_under_threshold_passthrough():
    engine = Engine(Config(threshold_tokens=1_000_000))
    ctx = engine.prepare(a_body(a_session(4)), ANTHROPIC)
    assert not ctx.modified
    assert ctx.outgoing_body()["messages"] == a_session(4)


def test_compact_store_and_substitute():
    engine = Engine(Config(threshold_tokens=2_000, keep_recent=1))
    msgs = a_session(10)

    # First over-threshold request: compacted + stored.
    ctx1 = engine.prepare(a_body(msgs), ANTHROPIC)
    assert ctx1.compacted and ctx1.modified
    assert n_summaries(ctx1.substituted) == 1
    assert len(engine.store) == 1
    assert ctx1.est_tokens_out < ctx1.est_tokens_in

    # Scaffold keeps growing the ORIGINAL history (it never saw C).
    msgs2 = grow(msgs, 10, 1, result_chars=10)
    ctx2 = engine.prepare(a_body(msgs2), ANTHROPIC)
    assert ctx2.modified
    out = ctx2.substituted
    # substituted = head + summary + original tail (same objects)
    assert out[0] is msgs2[0]
    assert out[1]["content"].startswith(SUMMARY_HEADER)
    assert out[-1] is msgs2[-1]
    # the tail includes everything after the stored cut
    assert out[2:] == msgs2[ctx2.base_cut :]


def test_recompaction_mapping_and_flatness():
    engine = Engine(Config(threshold_tokens=2_000, keep_recent=1))
    msgs = a_session(10)
    engine.prepare(a_body(msgs), ANTHROPIC)

    # Grow well past the threshold again -> second compaction on C + tail.
    msgs2 = grow(msgs, 10, 8)
    ctx = engine.prepare(a_body(msgs2), ANTHROPIC)
    assert ctx.compacted
    out = ctx.substituted
    assert n_summaries(out) == 1  # flat, never nested
    assert out[0] is msgs2[0]
    # kept tail messages are the scaffold's original objects
    assert out[-1] is msgs2[-1]
    assert len(engine.store) == 2  # entries at both cut depths

    # A third request extending the original matches the DEEPEST prefix.
    msgs3 = grow(msgs2, 18, 1, result_chars=10)
    ctx3 = engine.prepare(a_body(msgs3), ANTHROPIC)
    assert ctx3.base_cut > len(msgs)  # matched the second, deeper entry
    assert n_summaries(ctx3.substituted) == 1


def test_diverging_branches_share_prefix():
    engine = Engine(Config(threshold_tokens=2_000, keep_recent=1))
    msgs = a_session(10)
    engine.prepare(a_body(msgs), ANTHROPIC)

    branch_a = grow(msgs, 10, 1, result_chars=10)
    branch_b = list(msgs) + [a_assistant("different continuation", None)]
    ctx_a = engine.prepare(a_body(branch_a), ANTHROPIC)
    ctx_b = engine.prepare(a_body(branch_b), ANTHROPIC)
    assert ctx_a.modified and ctx_b.modified
    assert ctx_a.substituted[-1] is branch_a[-1]
    assert ctx_b.substituted[-1] is branch_b[-1]


def test_history_mutation_breaks_match_fail_open():
    engine = Engine(Config(threshold_tokens=2_000, keep_recent=1))
    msgs = a_session(10)
    engine.prepare(a_body(msgs), ANTHROPIC)

    mutated = [dict(m) for m in msgs]
    mutated[3] = {"role": "user", "content": "history rewritten by scaffold"}
    small = Engine(Config(threshold_tokens=1_000_000), store=engine.store)
    ctx = small.prepare(a_body(mutated), ANTHROPIC)
    assert not ctx.modified  # no match anywhere -> verbatim passthrough


def test_reactive_compacts_regardless_of_threshold():
    engine = Engine(Config(threshold_tokens=1_000_000, keep_recent=1))
    ctx = engine.prepare(a_body(a_session(10)), ANTHROPIC)
    assert not ctx.modified
    assert engine.reactive(ctx) is True
    assert ctx.compacted
    assert n_summaries(ctx.substituted) == 1


def test_store_loss_replays_chain_not_accumulate():
    """A whole day's history arriving at once (store lost overnight) must
    yield the same bounded, recent-only summary the live chain would have —
    never an accumulation of every cycle's content."""
    cfg = Config(threshold_tokens=2_000, keep_recent=1)

    # Live world: the proxy saw every request as the session grew.
    live = Engine(cfg)
    msgs = a_session(2)
    live_ctx = None
    for i in range(2, 60):
        live_ctx = live.prepare(a_body(msgs), ANTHROPIC)
        msgs = grow(msgs, i, 1)
    live_summary = next(
        m["content"] for m in live_ctx.substituted if n_summaries([m])
    )

    # Restart world: a fresh engine sees the final history in one shot.
    fresh = Engine(cfg)
    ctx = fresh.prepare(a_body(msgs), ANTHROPIC)
    assert ctx.compacted
    fresh_summary = next(
        m["content"] for m in ctx.substituted if n_summaries([m])
    )

    assert n_summaries(ctx.substituted) == 1
    # Old cycles fell away in replay, exactly like the live chain.
    assert "Step 2" not in fresh_summary
    assert "Step 10" not in fresh_summary
    # Bounded: same order of magnitude as the live summary, not N times it.
    assert len(fresh_summary) < 3 * max(len(live_summary), 1_000)
    # And the request is actually under threshold again.
    assert ctx.est_tokens_out <= cfg.threshold_tokens * 2


def test_reactive_gives_up_after_compaction():
    engine = Engine(Config(threshold_tokens=2_000, keep_recent=1))
    ctx = engine.prepare(a_body(a_session(10)), ANTHROPIC)
    assert ctx.compacted
    assert engine.reactive(ctx) is False  # already compacted; genuinely doesn't fit
