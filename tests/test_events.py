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


# --- session identity ----------------------------------------------------------
#
# The watcher's sid follows a conversation and splits when it branches. It is
# display only: the store keys on full-depth chain hashes and never sees one.


def _turn(msgs, i):
    msgs.append({"role": "assistant", "content": [{"type": "text", "text": "x" * 4000}]})
    msgs.append({"role": "user", "content": f"turn {i}"})
    return msgs


def _watch(app):
    from cliffcompaction.ui import Term

    w = Watcher(Term(), 0)
    for ev in app.state.hub.backlog():
        w.on_event(ev)
    return w


def _meta(session_id):
    """Claude Code's shape: an opaque JSON blob in the documented
    `metadata.user_id` field, carrying the conversation id."""
    return {"user_id": json.dumps(
        {"device_id": "d" * 8, "account_uuid": "a" * 8, "session_id": session_id}
    )}


def test_session_key_parses_only_the_shape_it_knows():
    from cliffcompaction.dialects.anthropic import session_key
    from cliffcompaction.dialects.openai_chat import DIALECT as OPENAI

    assert session_key({"metadata": _meta("sess-1")}) == "sess-1"
    assert session_key({}) is None
    assert session_key({"metadata": "not a dict"}) is None
    assert session_key({"metadata": {"user_id": "{bad json"}}) is None
    assert session_key({"metadata": {"user_id": "{}"}}) is None
    # A bare string is the documented per-USER id. Keying on it would merge
    # every session a user has ever opened into one row.
    assert session_key({"metadata": {"user_id": "user-42"}}) is None
    assert OPENAI.session_key({"user": "user-42"}) is None


def test_background_calls_share_the_session_they_ride_on():
    # Captured from Claude Code: between typed messages it sends a
    # prompt-suggestion request — the live history plus a synthetic user turn,
    # same model, tools and sampling params as a real turn. Only the client's
    # session id says it belongs to this conversation.
    app = _app()
    client = TestClient(app)
    meta = _meta("sess-A")
    msgs = [{"role": "user", "content": "one"}]
    for word in ("two", "three", "four"):
        client.post("/v1/messages",
                    json={"model": "m", "messages": list(_turn(msgs, word)), "metadata": meta})
        suggestion = [*msgs, {"role": "user", "content": "[SUGGESTION MODE: ...]"}]
        client.post("/v1/messages",
                    json={"model": "m", "messages": suggestion, "metadata": meta})

    assert len({e["sid"] for e in app.state.hub.backlog()}) == 1
    assert len(_watch(app).live_sessions()) == 1


def test_client_session_ids_separate_identical_openings():
    app = _app()
    client = TestClient(app)
    for label in ("A", "B"):
        msgs = [{"role": "user", "content": "hello!"}]
        for i in range(2):
            client.post("/v1/messages",
                        json={"model": "m", "messages": list(_turn(msgs, i)),
                              "metadata": _meta(f"sess-{label}")})
    assert len({e["sid"] for e in app.state.hub.backlog()}) == 2
    assert len(_watch(app).live_sessions()) == 2


def test_the_id_is_hashed_not_the_client_blob():
    # The blob sits next to account and device identifiers, and this id goes
    # out on an event stream any local process can read.
    app = _app()
    client = TestClient(app)
    client.post("/v1/messages", json={"model": "m", "messages": [{"role": "user", "content": "hi"}],
                                      "metadata": _meta("sess-secret")})
    blob = json.dumps(app.state.hub.backlog())
    assert "sess-secret" not in blob
    assert "aaaaaaaa" not in blob and "dddddddd" not in blob


def test_without_a_client_session_id_the_root_hash_keys_the_row():
    app = _app()
    client = TestClient(app)
    msgs = [{"role": "user", "content": "hello"}]
    for i in range(3):
        client.post("/v1/messages", json={"model": "m", "messages": list(_turn(msgs, i))})
    # A rewind merges with its parent here. Without a client id, a branch and
    # the scaffold's own background calls are the same shape, and inventing a
    # split we cannot verify is the worse error of the two.
    branch = msgs[:-4]
    client.post("/v1/messages", json={"model": "m", "messages": list(_turn(branch, "b"))})
    assert len({e["sid"] for e in app.state.hub.backlog()}) == 1
    assert len(_watch(app).live_sessions()) == 1


def test_side_calls_keep_the_session_id_but_stay_out_of_its_row():
    # The permission classifier is labelled with the session that fired it, and
    # that is where its cost lands — but it is not one of that session's turns.
    app = _app()
    client = TestClient(app)
    meta = _meta("sess-A")
    msgs = [{"role": "user", "content": "one"}]
    client.post("/v1/messages",
                json={"model": "m", "messages": list(_turn(msgs, 0)), "metadata": meta})
    turn = app.state.hub.backlog()[0]
    for call in _classifier(3):
        client.post("/v1/messages", json={"model": "m", "messages": call, "metadata": meta})

    events = app.state.hub.backlog()
    assert [e["aux"] for e in events] == [False, True, True, True]
    assert len({e["sid"] for e in events}) == 1   # all one session

    w = _watch(app)
    (row,) = w.live_sessions()
    assert w.requests == 4                        # every call is real money
    assert row.total == turn["total"]             # the row is the turn's, only
    assert row.est == turn["est_in"]


def _classifier(n_calls):
    """Claude Code's permission classifier: fixed instruction, growing
    transcript, never an assistant message."""
    for i in range(n_calls):
        yield [
            {"role": "user", "content": "You are reviewing a tool call."},
            {"role": "user", "content": "<transcript>" + "y" * (500 * (i + 1))},
        ]


def test_classifier_calls_make_no_row_even_without_a_client_id():
    # Two user messages and no model turn cannot be an opening turn, so this
    # needs no prior sighting of the session to be recognised.
    app = _app()
    client = TestClient(app)
    for msgs in _classifier(6):
        client.post("/v1/messages", json={"model": "m", "messages": msgs})

    events = app.state.hub.backlog()
    assert len(events) == 6
    assert all(e["aux"] for e in events)

    w = _watch(app)
    assert w.live_sessions() == []               # never a session
    assert w.requests == 6                       # real requests, real money
    assert len(w.feed) == 6 and "aux" in w.render()


def test_a_background_call_after_the_session_speaks_is_a_side_call():
    # One user message, no model turn — the shape of an opening turn. What
    # rules it out is that this session has already had one.
    app = _app()
    client = TestClient(app)
    meta = _meta("sess-A")
    msgs = [{"role": "user", "content": "one"}]
    client.post("/v1/messages",
                json={"model": "m", "messages": list(_turn(msgs, 0)), "metadata": meta})
    client.post("/v1/messages",
                json={"model": "m", "messages": [{"role": "user", "content": "x" * 4000}],
                      "metadata": meta})

    opening, side = app.state.hub.backlog()
    assert not opening["aux"] and side["aux"]
    (row,) = _watch(app).live_sessions()
    assert row.est == opening["est_in"]           # the 4k call left no mark


def test_one_message_request_is_not_a_side_call():
    # A real session's opening turn must still start a row.
    app = _app()
    client = TestClient(app)
    client.post("/v1/messages", json={"model": "m", "messages": [{"role": "user", "content": "go"}]})
    ev = app.state.hub.backlog()[0]
    assert not ev["aux"]
    assert len(_watch(app).live_sessions()) == 1


def test_side_calls_never_pollute_lineage():
    app = _app()
    client = TestClient(app)
    msgs = [{"role": "user", "content": "hello"}]
    calls = _classifier(4)
    for i in range(4):
        client.post("/v1/messages", json={"model": "m", "messages": list(_turn(msgs, i))})
        client.post("/v1/messages", json={"model": "m", "messages": next(calls)})

    events = app.state.hub.backlog()
    session = [e for e in events if not e["aux"]]
    assert len({e["sid"] for e in session}) == 1  # interleaving changed nothing
    w = _watch(app)
    assert len(w.sessions) == 1
    assert sum(e["aux"] for e in events) == 4


def test_reactive_replay_stays_one_session():
    # One HTTP request emits an event per escalation rung; they are the same
    # conversation, not one session per rung.
    from test_escalation import PickyUpstream, fat_session

    cfg = Config.from_env()
    cfg.threshold_tokens = 8000
    up = PickyUpstream(max_chars=20000)
    app = create_app(cfg, transport=httpx.MockTransport(up.handler))
    r = TestClient(app).post("/v1/messages", json={"model": "m", "messages": fat_session(8)})

    assert r.status_code == 200
    events = app.state.hub.backlog()
    assert len(events) >= 2                      # at least one replay emitted
    assert len({e["sid"] for e in events}) == 1
    assert len(_watch(app).live_sessions()) == 1


def test_events_without_the_field_still_make_rows():
    # An old daemon, a new watcher: everything is a session, as before.
    from cliffcompaction.ui import Term

    w = Watcher(Term(), 0)
    w.on_event({"sid": "abcd1234", "kind": "pass", "total": 2, "est_in": 100})
    assert len(w.live_sessions()) == 1


def test_side_call_rule_across_dialects():
    from cliffcompaction.dialects import anthropic, openai_chat, openai_responses
    from cliffcompaction.proxy import is_side_call

    pairs = [
        (anthropic.DIALECT,
         [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
         [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]),
        (openai_chat.DIALECT,
         [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
         [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]),
        (openai_responses.DIALECT,
         [{"type": "message", "role": "user", "content": "a"},
          {"type": "message", "role": "user", "content": "b"}],
         # a model turn need not be an assistant message in this dialect
         [{"type": "message", "role": "user", "content": "a"},
          {"type": "function_call", "call_id": "c", "name": "bash", "arguments": "{}"}]),
    ]
    for dialect, side, session in pairs:
        assert is_side_call(side, dialect, seen=False), dialect.name
        # A model turn is a turn whether or not the session has been seen.
        assert not is_side_call(session, dialect, seen=False), dialect.name
        assert not is_side_call(session, dialect, seen=True), dialect.name
        # One user message: an opening turn until the session has spoken.
        assert not is_side_call(side[:1], dialect, seen=False), dialect.name
        assert is_side_call(side[:1], dialect, seen=True), dialect.name


def test_identity_work_can_never_break_a_request(monkeypatch):
    # The seam: session ids are display state, assigned before the request
    # goes out. A client can put anything in that field, so if reading it
    # blows up the row degrades to the root-keyed id and the request path
    # carries on — same responses, same compaction.
    import dataclasses

    from cliffcompaction.dialects import anthropic
    from cliffcompaction.engine import Engine

    def boom(body):
        raise RuntimeError("boom")

    monkeypatch.setattr(anthropic, "DIALECT", dataclasses.replace(anthropic.DIALECT, session_key=boom))

    cfg = Config.from_env()
    cfg.threshold_tokens = 2000
    engine = Engine(cfg)
    app = create_app(
        cfg, engine=engine,
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"ok": True})),
    )
    client = TestClient(app)
    msgs = [{"role": "user", "content": "hello"}]
    for i in range(5):
        r = client.post("/v1/messages", json={"model": "m", "messages": list(_turn(msgs, i)),
                                              "metadata": _meta("sess-A")})
        assert r.status_code == 200
    assert len(engine.store) > 0                 # still compacting

    from cliffcompaction.hashing import chain_hashes

    root = chain_hashes([anthropic.digest_message(msgs[0])])[0][:12]
    assert {e["sid"] for e in app.state.hub.backlog()} == {root}


def test_a_sessions_opening_turn_is_not_a_side_call():
    # Captured from Claude Code: the first request of a session is
    # [user, in-array system directive] — two messages, no model turn, which
    # the raw-length rule swept up as a classifier call. The row then only
    # appeared on the second message, the first with an assistant turn in it.
    from cliffcompaction.dialects import detect
    from cliffcompaction.proxy import is_side_call

    dialect = detect("/v1/messages")
    opening = [
        {"role": "user", "content": "hello"},
        {"role": "system", "content": "Available agent types for the Agent tool: ..."},
    ]
    assert not is_side_call(opening, dialect, seen=False)

    app = _app()
    client = TestClient(app)
    client.post("/v1/messages",
                json={"model": "m", "messages": opening, "metadata": _meta("sess-A")})
    events = app.state.hub.backlog()
    assert len(events) == 1 and not events[0]["aux"]
    assert len(_watch(app).live_sessions()) == 1          # renders immediately

    # ...and the classifier, two user messages and no model turn, still does not.
    for call in _classifier(2):
        client.post("/v1/messages", json={"model": "m", "messages": call,
                                          "metadata": _meta("sess-A")})
    assert len(_watch(app).live_sessions()) == 1
