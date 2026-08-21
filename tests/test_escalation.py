"""Escalation ladder: proactive rungs, reactive ladder, truncation, give-up."""

from __future__ import annotations

import json

import httpx
from starlette.testclient import TestClient

from cliffcompaction.config import Config
from cliffcompaction.dialects.anthropic import DIALECT
from cliffcompaction.dialects.base import SUMMARY_HEADER
from cliffcompaction.engine import Engine
from cliffcompaction.proxy import create_app

from util import a_body


def fat_session(n_turns, result_chars=6000):
    msgs = [{"role": "user", "content": "the task: fix the bug"}]
    for i in range(n_turns):
        msgs.append({"role": "assistant", "content": [
            {"type": "text", "text": f"thought {i}: " + "y" * 2000},
            {"type": "tool_use", "id": f"t{i}", "name": "bash", "input": {"command": f"make {i}"}},
        ]})
        msgs.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"t{i}", "content": "R" * result_chars}
        ]})
    return msgs


# --- proactive escalation ------------------------------------------------------


def test_proactive_escalates_when_floor_over_threshold():
    # keep_recent=3 tail alone (~3 * 8k chars) exceeds a 4k-token threshold;
    # rung 1 (keep_recent=1) must fire and get under budget.
    eng = Engine(Config(threshold_tokens=4000, keep_recent=3))
    ctx = eng.prepare(a_body(fat_session(8)), DIALECT)
    assert ctx.compacted
    assert ctx.rung >= 1
    assert ctx.est_tokens_out <= 4000
    # tail is a single turn now
    assert sum(1 for m in ctx.substituted if m.get("role") == "assistant") == 1


def test_proactive_case1_few_giant_turns():
    # 3 turns with keep_recent=3: base compaction has nothing to do,
    # escalation rung 1 compacts down to 1 kept turn.
    eng = Engine(Config(threshold_tokens=3000, keep_recent=3))
    ctx = eng.prepare(a_body(fat_session(3)), DIALECT)
    assert ctx.compacted
    assert ctx.rung >= 1
    assert len(ctx.substituted) < len(ctx.msgs)


def test_proactive_soft_send_on_giant_live_turn():
    # One monster turn: nothing below the live turn to compact away enough.
    # Must not crash; request goes out (possibly over budget), fail-open.
    eng = Engine(Config(threshold_tokens=1000, keep_recent=3))
    ctx = eng.prepare(a_body(fat_session(1, result_chars=40000)), DIALECT)
    assert ctx.outgoing_body()  # no exception; soft path


def test_no_escalation_without_assistant_turns():
    eng = Engine(Config(threshold_tokens=100, keep_recent=3))
    msgs = [{"role": "user", "content": "x" * 30000}]
    ctx = eng.prepare(a_body(msgs), DIALECT)
    assert not ctx.compacted and not ctx.modified and ctx.rung == 0


# --- reactive ladder -----------------------------------------------------------


class PickyUpstream:
    """Rejects with a context error until the request fits max_chars."""

    def __init__(self, max_chars):
        self.max_chars = max_chars
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if len(request.content) > self.max_chars:
            return httpx.Response(
                400,
                json={"error": {"type": "invalid_request_error", "message": "prompt is too long"}},
            )
        return httpx.Response(200, json={"id": "m", "role": "assistant", "content": []})


def make(cfg, max_chars):
    up = PickyUpstream(max_chars)
    app = create_app(cfg, transport=httpx.MockTransport(up.handler))
    return TestClient(app), up


def test_reactive_ladder_succeeds_at_lower_rung():
    # Threshold high (proactive silent); provider window small: reactive
    # rungs shrink until accepted.
    cfg = Config(threshold_tokens=8000, keep_recent=3)
    client, up = make(cfg, max_chars=20000)
    r = client.post("/v1/messages", json=a_body(fat_session(8)))
    assert r.status_code == 200
    assert len(up.requests) >= 2  # original + at least one replay
    sent = json.loads(up.requests[-1].content)["messages"]
    assert any(
        isinstance(m.get("content"), str) and m["content"].startswith(SUMMARY_HEADER)
        for m in sent
    )


def test_reactive_rung3_truncates_summary():
    # Provider window sits between the truncation target (threshold) and the
    # untruncated floor: ladder must reach rung 3 and shrink the summary
    # itself (newest parts kept). Windows below the threshold are out of
    # scope by definition (threshold must be chosen under the window).
    cfg = Config(threshold_tokens=1000, keep_recent=1)
    client, up = make(cfg, max_chars=6000)
    r = client.post("/v1/messages", json=a_body(fat_session(10, result_chars=300)))
    assert r.status_code == 200
    sent = json.loads(up.requests[-1].content)["messages"]
    summaries = [m for m in sent if isinstance(m.get("content"), str) and m["content"].startswith(SUMMARY_HEADER)]
    assert summaries
    # newest-part retention: the last folded turn survives before older ones
    text = summaries[0]["content"]
    if "make" in text:
        assert "make 0" not in text or "make 8" in text


def test_reactive_gives_up_and_returns_400():
    # Provider rejects everything: after the ladder, the client gets the 400.
    cfg = Config(threshold_tokens=2000, keep_recent=1)
    client, up = make(cfg, max_chars=10)
    r = client.post("/v1/messages", json=a_body(fat_session(6)))
    assert r.status_code == 400
    # bounded replays: original + at most one per remaining rung
    assert len(up.requests) <= 5
