import asyncio
import json

import httpx
import pytest
from starlette.testclient import TestClient

from cliffcompaction.config import Config
from cliffcompaction.events import EventHub
from cliffcompaction.proxy import create_app
from cliffcompaction.watch import Watcher, agefmt, kfmt, short_model


def test_hub_is_bounded_and_ordered():
    hub = EventHub(maxlen=3)
    for i in range(6):
        hub.emit(kind="pass", i=i)
    buf = hub.backlog()
    assert [e["i"] for e in buf] == [3, 4, 5]
    assert [e["seq"] for e in buf] == [4, 5, 6]


def test_emit_never_raises_on_unserializable():
    hub = EventHub()
    hub.emit(kind="pass", obj=object())      # only json.dumps would choke
    assert len(hub.backlog()) == 1


def test_slow_subscriber_drops_instead_of_growing():
    async def go():
        hub = EventHub()
        q = hub.subscribe()
        for i in range(1000):
            hub.emit(kind="pass", i=i)
        assert q.qsize() == 512
        assert hub.dropped == 1000 - 512
        hub.unsubscribe(q)
        assert hub.subscribers == 0

    asyncio.run(go())


def _app(threshold=2000):
    cfg = Config.from_env()
    cfg.threshold_tokens = threshold
    return create_app(
        cfg,
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"ok": True})),
    )


def test_proxy_emits_one_event_per_handled_request():
    app = _app()
    client = TestClient(app)
    msgs = [{"role": "user", "content": "hello"}]
    for i in range(5):
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": "x" * 4000}]})
        msgs.append({"role": "user", "content": f"turn {i}"})
        client.post("/v1/messages", json={"model": "test-model", "messages": list(msgs)})

    events = app.state.hub.backlog()
    assert len(events) == 5
    assert {e["sid"] for e in events} == {events[0]["sid"]}      # one session
    assert events[0]["kind"] == "pass"                           # nothing stored yet
    assert any(e["kind"] == "compact" for e in events)
    # The first compaction of a tiny history can grow it (a full-text summary
    # of one turn is not smaller); by the last one the drop is real.
    compact = [e for e in events if e["kind"] == "compact"][-1]
    assert compact["est_out"] < compact["est_in"]
    assert compact["fp"] and compact["steps"] >= 1
    assert compact["model"] == "test-model"
    assert compact["dialect"] == "anthropic"


def test_events_from_two_sessions_are_separate():
    app = _app()
    client = TestClient(app)
    for opening in ("first session", "second session"):
        client.post(
            "/v1/messages",
            json={"model": "m", "messages": [{"role": "user", "content": opening}]},
        )
    assert len({e["sid"] for e in app.state.hub.backlog()}) == 2


def test_untouched_paths_emit_nothing():
    app = _app()
    client = TestClient(app)
    client.get("/v1/models")
    client.post("/v1/messages", json={"model": "m"})       # no messages array
    assert app.state.hub.backlog() == []


def test_probe_requests_emit_nothing():
    """Claude Code opens every session with an identical max_tokens=1 probe;
    counting it would collapse every session into one phantom watcher row."""
    app = _app()
    client = TestClient(app)
    probe = {"role": "user", "content": "quota"}
    client.post("/v1/messages", json={"model": "m", "max_tokens": 1, "messages": [probe]})
    client.post("/v1/chat/completions", json={"model": "m", "max_completion_tokens": 1,
                                             "messages": [probe]})
    assert app.state.hub.backlog() == []

    # ...but a real turn with the same opening message still reports.
    client.post("/v1/messages", json={"model": "m", "max_tokens": 4096, "messages": [probe]})
    assert len(app.state.hub.backlog()) == 1


def test_status_reports_watchers_and_uptime():
    body = TestClient(_app()).get("/__cliff__/status").json()
    assert body["watchers"] == 0
    assert "uptime_s" in body


def test_watcher_builds_rows_from_events():
    app = _app()
    client = TestClient(app)
    msgs = [{"role": "user", "content": "hello"}]
    for i in range(5):
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": "x" * 4000}]})
        msgs.append({"role": "user", "content": f"turn {i}"})
        client.post("/v1/messages", json={"model": "claude-sonnet-4-5-20250929", "messages": list(msgs)})

    from cliffcompaction.ui import Term

    w = Watcher(Term(), 0)
    for ev in app.state.hub.backlog():
        w.on_event(ev)
    assert w.requests == 5
    assert w.compactions >= 1
    sessions = w.live_sessions()
    assert len(sessions) == 1
    assert sessions[0].model == "sonnet-4-5"
    assert sessions[0].compactions == w.compactions
    assert "⟨cliff⟩" in w.render()


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("claude-sonnet-4-5-20250929", "sonnet-4-5"),
        ("claude-haiku-4-5", "haiku-4-5"),
        ("k3", "k3"),
        ("", "?"),
    ],
)
def test_short_model(raw, expected):
    assert short_model(raw) == expected


def test_formatters():
    assert kfmt(999) == "999"
    assert kfmt(12_400) == "12k"
    assert kfmt(2_500_000) == "2.5M"
    assert agefmt(9) == "9s"
    assert agefmt(300) == "5m"
    assert agefmt(7300) == "2h"
