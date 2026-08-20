"""End-to-end proxy tests against a mock upstream."""

from __future__ import annotations

import json

import httpx
from starlette.testclient import TestClient

from cliffcompaction.config import Config
from cliffcompaction.dialects.base import SUMMARY_HEADER
from cliffcompaction.proxy import create_app

from util import a_body, a_session


class Upstream:
    """Mock upstream recording every request it receives."""

    def __init__(self):
        self.requests: list[httpx.Request] = []
        self.script: list[httpx.Response] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.script:
            return self.script.pop(0)
        return httpx.Response(
            200,
            json={"id": "msg_1", "role": "assistant", "content": []},
        )

    def last_messages(self) -> list[dict]:
        return json.loads(self.requests[-1].content)["messages"]


def make(cfg: Config) -> tuple[TestClient, Upstream]:
    up = Upstream()
    app = create_app(cfg, transport=httpx.MockTransport(up.handler))
    return TestClient(app), up


def test_small_request_passthrough_bytes():
    client, up = make(Config(threshold_tokens=1_000_000))
    body = a_body(a_session(3))
    raw = json.dumps(body).encode()
    r = client.post("/v1/messages", content=raw, headers={"x-api-key": "k"})
    assert r.status_code == 200
    # verbatim: the exact bytes we sent
    assert up.requests[-1].content == raw
    assert up.requests[-1].headers["x-api-key"] == "k"


def test_over_threshold_compacted():
    client, up = make(Config(threshold_tokens=2_000, keep_recent=1))
    body = a_body(a_session(10))
    r = client.post("/v1/messages", json=body)
    assert r.status_code == 200
    sent = up.last_messages()
    assert len(sent) < len(body["messages"])
    assert sent[1]["content"].startswith(SUMMARY_HEADER)


def test_followup_substituted():
    client, up = make(Config(threshold_tokens=2_000, keep_recent=1))
    msgs = a_session(10)
    client.post("/v1/messages", json=a_body(msgs))
    # scaffold's next request: original history + one more exchange
    msgs2 = msgs + [
        {"role": "assistant", "content": [{"type": "text", "text": "next step"}]},
    ]
    client.post("/v1/messages", json=a_body(msgs2))
    sent = up.last_messages()
    assert sent[1]["content"].startswith(SUMMARY_HEADER)
    assert sent[-1]["content"][0]["text"] == "next step"


def test_shadow_mode_never_modifies():
    client, up = make(Config(threshold_tokens=2_000, keep_recent=1, shadow=True))
    body = a_body(a_session(10))
    raw = json.dumps(body).encode()
    r = client.post("/v1/messages", content=raw)
    assert r.status_code == 200
    assert up.requests[-1].content == raw  # untouched despite being over threshold


def test_reactive_retry_on_context_400():
    client, up = make(Config(threshold_tokens=1_000_000, keep_recent=1))
    up.script = [
        httpx.Response(
            400,
            json={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "prompt is too long: 210000 tokens > 200000 maximum",
                },
            },
        ),
    ]
    r = client.post("/v1/messages", json=a_body(a_session(10)))
    assert r.status_code == 200  # replayed after compaction
    assert len(up.requests) == 2
    sent = up.last_messages()
    assert sent[1]["content"].startswith(SUMMARY_HEADER)


def test_non_context_400_passed_through():
    client, up = make(Config(threshold_tokens=1_000_000))
    up.script = [
        httpx.Response(
            400,
            json={"type": "error", "error": {"type": "invalid_request_error", "message": "max_tokens must be positive"}},
        )
    ]
    r = client.post("/v1/messages", json=a_body(a_session(4)))
    assert r.status_code == 400
    assert len(up.requests) == 1
    assert "max_tokens" in r.text


def test_malformed_json_fail_open():
    client, up = make(Config(threshold_tokens=1))
    r = client.post("/v1/messages", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 200
    assert up.requests[-1].content == b"{not json"


def test_count_tokens_passthrough():
    client, up = make(Config(threshold_tokens=1, keep_recent=1))
    body = a_body(a_session(10))
    raw = json.dumps(body).encode()
    r = client.post("/v1/messages/count_tokens", content=raw)
    assert r.status_code == 200
    assert up.requests[-1].content == raw


def test_upstream_routing():
    cfg = Config(
        threshold_tokens=1_000_000,
        anthropic_upstream="https://anthropic.example",
        openai_upstream="https://openai.example",
    )
    client, up = make(cfg)
    client.post("/v1/messages", json=a_body(a_session(2)))
    assert up.requests[-1].url.host == "anthropic.example"
    client.post(
        "/v1/chat/completions",
        json={"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert up.requests[-1].url.host == "openai.example"
    client.get("/v1/models")
    assert up.requests[-1].url.host == "anthropic.example"


def test_leftover_paths_follow_single_configured_upstream():
    # Only the OpenAI upstream configured: misc paths follow it.
    cfg = Config(threshold_tokens=1_000_000, openai_upstream="https://openai.example")
    client, up = make(cfg)
    client.get("/v1/models")
    assert up.requests[-1].url.host == "openai.example"
    client.head("/api/hello")
    assert up.requests[-1].url.host == "openai.example"
    # Dialect paths still route by dialect.
    client.post("/v1/messages", json=a_body(a_session(2)))
    assert up.requests[-1].url.host == "api.anthropic.com"

    # Only the Anthropic upstream configured: misc paths follow it (as before).
    cfg = Config(threshold_tokens=1_000_000, anthropic_upstream="https://kimi.example")
    client, up = make(cfg)
    client.get("/v1/models")
    assert up.requests[-1].url.host == "kimi.example"
    client.post("/v1/messages/count_tokens", json=a_body(a_session(2)))
    assert up.requests[-1].url.host == "kimi.example"


def test_streaming_bytes_relayed():
    client, up = make(Config(threshold_tokens=1_000_000))
    sse = b'event: message_start\ndata: {"type":"message_start"}\n\nevent: done\ndata: {}\n\n'
    up.script = [
        httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})
    ]
    r = client.post("/v1/messages", json=a_body(a_session(2)))
    assert r.status_code == 200
    assert r.content == sse
    assert r.headers["content-type"] == "text/event-stream"


def test_timing_logged_at_debug(caplog):
    import logging

    client, _up = make(Config(threshold_tokens=1_000_000))
    with caplog.at_level(logging.DEBUG, logger="cliffcompaction"):
        r = client.post("/v1/messages", json=a_body(a_session(2)))
    assert r.status_code == 200
    messages = [rec.message for rec in caplog.records if "timing:" in rec.message]
    assert any("upstream_headers" in m for m in messages)
    assert any("first_byte" in m for m in messages)


def test_status_endpoint():
    client, _up = make(Config())
    r = client.get("/__cliff__/status")
    assert r.status_code == 200
    data = r.json()
    assert data["name"] == "cliffcompaction"
    assert data["shadow"] is False


def test_debug_dir_dumps_requests(tmp_path):
    import os
    cfg = Config(threshold_tokens=2_000, keep_recent=1, debug_dir=str(tmp_path))
    client, up = make(cfg)
    r = client.post("/v1/messages", json=a_body(a_session(10)))
    assert r.status_code == 200
    files = sorted(os.listdir(tmp_path))
    assert len(files) == 1
    rec = json.loads((tmp_path / files[0]).read_text())
    assert rec["dialect"] == "anthropic"
    assert rec["modified"] is True
    assert len(rec["incoming_messages"]) == len(a_session(10))
    assert rec["outgoing_messages"][1]["content"].startswith(SUMMARY_HEADER)
    # upstream still got the compacted body — dump is observability only
    assert len(up.last_messages()) == len(rec["outgoing_messages"])
