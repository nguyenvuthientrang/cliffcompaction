"""The transparent proxy.

Fail-open: any failure (unparseable body, unknown path, engine error)
results in verbatim passthrough. Shadow mode runs the full pipeline but
forwards every request verbatim.

The single exception is strict mode, off by default: there a request that
is still over budget once the escalation ladder is exhausted is refused
rather than forwarded. Every other failure still fails open, strict or not.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import logging
import os
import re
import time
from collections import OrderedDict
from pathlib import Path
from contextlib import asynccontextmanager

import httpx
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from . import __version__
from .config import DEFAULT_ANTHROPIC_UPSTREAM, DEFAULT_OPENAI_UPSTREAM, Config
from .dialects import detect
from .engine import Engine
from .events import EventHub, sse

logger = logging.getLogger("cliffcompaction")

_STRIP_REQUEST_HEADERS = {
    "host",
    "content-length",
    "connection",
    "transfer-encoding",
    "accept-encoding",
    "expect",
}
_STRIP_RESPONSE_HEADERS = {
    "content-length",
    "transfer-encoding",
    "connection",
    "content-encoding",
}

# Providers phrase context-overflow errors inconsistently; match broadly.
_CONTEXT_ERROR_RE = re.compile(
    r"context.?length|context.?window|prompt is too long|input.tokens"
    r"|maximum.context|exceeds.limit|too.many.tokens"
    r"|reduce.the.(?:length|input)|context_length_exceeded",
    re.IGNORECASE,
)


def is_context_error(body: bytes) -> bool:
    try:
        text = body.decode("utf-8", errors="replace")
    except Exception:
        return False
    return bool(_CONTEXT_ERROR_RE.search(text))


def code_mtime() -> float:
    """Newest source file of the running package.

    A supervised daemon keeps serving the code it started with, so an upgrade
    underneath it changes nothing until a restart — with no symptom, since the
    old process stays perfectly healthy. Comparing this against the daemon's
    start time is how `cliff status` notices.
    """
    try:
        return max(p.stat().st_mtime for p in Path(__file__).parent.rglob("*.py"))
    except Exception:
        return 0.0


def is_side_call(msgs: list[dict], dialect, seen: bool) -> bool:
    """No model turn: a scaffold's own call, not a turn of the conversation.

    `compact` bails on exactly this condition, and every escalation rung sits
    downstream of it, so such a request is outside cliff's unit of work at any
    threshold. It still costs money and still belongs to a session, so it is
    reported — but it is kept out of that session's row, whose depth and size
    describe a conversation these calls are not part of.

    A session's opening turn has no model turn either, and must not be swept up
    here: it is what puts a new session on screen. Two signals separate them.
    An opening turn carries exactly one user message ([user] or [user, in-array
    system directive]), so two or more is proof of something else — Claude
    Code's permission classifier, a fixed instruction and a growing transcript,
    is the volume case. And a session that has already spoken cannot be opening
    again, which catches the single-message background calls that shape alone
    cannot distinguish from a first turn.
    """
    try:
        if any(dialect.is_assistant(m) for m in msgs):
            return False
        if seen:
            return True
        return sum(1 for m in msgs if m.get("role") == "user") >= 2
    except Exception:
        return False  # unknown shape: treat it as a session


MAX_SESSIONS = 256


def session_id(ctx, dialect) -> str:
    """Display id for one request's conversation.

    Prefer what the client says. Claude Code labels every request — including
    its own background calls, which are otherwise indistinguishable from a
    person typing — with the session it belongs to; no amount of inspecting
    the message array recovers that. Without one, fall back to the history's
    root hash: stable for the life of a conversation, and it merges a branch
    with its parent rather than inventing a split we cannot verify.

    Hashed, not passed through: the client's blob sits next to account and
    device identifiers, and this id goes out on an event stream any local
    process can read. 12 hex chars, as the watcher's columns assume.
    """
    root = ctx.chain[0][:12]
    try:
        key = dialect.session_key(ctx.body)
    except Exception:
        return root
    if not key:
        return root
    return hashlib.sha256(f"session:{key}".encode("utf-8")).hexdigest()[:12]


def create_app(
    cfg: Config,
    engine: Engine | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Starlette:
    engine = engine or Engine(cfg)
    hub = EventHub()
    # Distinct conversations that have taken a turn, for `cliff status` — and
    # the memory `is_side_call` reads. Bounded, and display only.
    seen_sids: OrderedDict[str, None] = OrderedDict()
    started_at = time.time()
    _debug_seq = itertools.count(1)

    # Output caps a scaffold uses for probes rather than turns.
    _CAP_KEYS = ("max_tokens", "max_output_tokens", "max_completion_tokens")

    def is_probe(body: dict) -> bool:
        """A request that asks for at most one output token is a liveness or
        quota probe, not a turn. Claude Code opens every session with one
        (`content: "quota"`, max_tokens 1) — byte-identical across sessions, so
        it would otherwise share a chain hash and show up as one phantom
        session in the watcher."""
        for key in _CAP_KEYS:
            cap = body.get(key)
            if isinstance(cap, int) and cap <= 1:
                return True
        return False

    def emit_request(ctx, dialect) -> None:
        """One event per handled request. Observability only, and it runs
        before the request goes out: failures never propagate."""
        try:
            _emit_request(ctx, dialect)
        except Exception:
            logger.exception("event emit failed (request unaffected)")

    def _emit_request(ctx, dialect) -> None:
        if ctx is None or not ctx.chain:
            return
        if isinstance(ctx.body, dict) and is_probe(ctx.body):
            return
        model = ctx.body.get("model") if isinstance(ctx.body, dict) else None
        # A side call is labelled with the session that fired it, so its cost
        # stays attributable, but never enters the tracker: only a turn makes a
        # session "seen", and only that keeps the opening-turn exemption from
        # applying to every background call that follows.
        sid = session_id(ctx, dialect)
        aux = is_side_call(ctx.msgs, dialect, sid in seen_sids)
        if not aux:
            seen_sids[sid] = None
            seen_sids.move_to_end(sid)
            while len(seen_sids) > MAX_SESSIONS:
                seen_sids.popitem(last=False)
        hub.emit(
            kind="compact" if ctx.compacted else ("match" if ctx.modified else "pass"),
            sid=sid,
            aux=aux,
            model=model if isinstance(model, str) else "?",
            dialect=dialect.name,
            cut=ctx.base_cut,
            total=len(ctx.msgs),
            out_msgs=ctx.out_msgs,
            est_in=ctx.est_tokens_in,
            est_out=ctx.est_tokens_out or ctx.est_tokens_in,
            steps=ctx.chain_steps,
            rung=ctx.rung,
            fp=ctx.summary_fp,
        )

    def debug_dump(path: str, dialect, ctx) -> None:
        """Write what cliff saw and what it sent. Failures never propagate."""
        try:
            os.makedirs(cfg.debug_dir, exist_ok=True)
            n = next(_debug_seq)
            fname = os.path.join(
                cfg.debug_dir,
                f"{time.strftime('%Y%m%d-%H%M%S')}-{n:04d}.json",
            )
            record = {
                "path": path,
                "dialect": dialect.name,
                "modified": ctx.modified,
                "compacted": ctx.compacted,
                "est_tokens_in": ctx.est_tokens_in,
                "est_tokens_out": ctx.est_tokens_out,
                # Everything but the message array: model, sampling params,
                # output caps, tools. What distinguishes a scaffold's own
                # background calls from a conversational turn lives here.
                "request_fields": {
                    k: (f"<{len(v)} tools>" if k == "tools" and isinstance(v, list) else v)
                    for k, v in ctx.body.items()
                    if k != dialect.messages_key and k != "system"
                },
                "incoming_messages": ctx.msgs,
                "outgoing_messages": ctx.substituted if ctx.modified else None,
            }
            with open(fname, "w") as f:
                json.dump(record, f, ensure_ascii=False, indent=1)
            logger.info("debug: dumped request to %s", fname)
        except Exception:
            logger.exception("debug dump failed (request unaffected)")

    client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=30.0, read=600.0, write=600.0, pool=30.0),
        transport=transport,
    )

    def pick_upstream(path: str) -> str:
        p = path.rstrip("/")
        if p.endswith("/chat/completions") or p.endswith("/responses") or p.endswith("/responses/compact"):
            return cfg.openai_upstream
        if "/v1/messages" in p or p.endswith("/messages"):
            return cfg.anthropic_upstream
        # Neither dialect claims this path (/v1/models, /api/hello, ...) and
        # the path alone cannot decide (/v1/models exists on both APIs).
        # Follow the upstream the user explicitly configured when that is
        # unambiguous; otherwise keep the Anthropic default.
        if (
            cfg.openai_upstream != DEFAULT_OPENAI_UPSTREAM
            and cfg.anthropic_upstream == DEFAULT_ANTHROPIC_UPSTREAM
        ):
            return cfg.openai_upstream
        return cfg.anthropic_upstream

    async def send_upstream(request: Request, upstream: str, body: bytes) -> httpx.Response:
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in _STRIP_REQUEST_HEADERS
        }
        headers["accept-encoding"] = "identity"
        url = upstream.rstrip("/") + request.url.path
        if request.url.query:
            url += "?" + request.url.query
        req = client.build_request(request.method, url, headers=headers, content=body)
        return await client.send(req, stream=True)

    def relay(resp: httpx.Response, t_start: float | None = None) -> StreamingResponse:
        headers = {
            k: v
            for k, v in resp.headers.items()
            if k.lower() not in _STRIP_RESPONSE_HEADERS
        }

        async def body():
            first = t_start is not None
            n_bytes = 0
            started = time.monotonic()
            try:
                async for chunk in resp.aiter_raw():
                    if first:
                        logger.debug(
                            "timing: first_byte %.0fms",
                            (time.monotonic() - t_start) * 1000,
                        )
                        first = False
                    n_bytes += len(chunk)
                    yield chunk
            except httpx.StreamConsumed:
                # Content was already loaded (e.g. mock transports); serve it.
                if first:
                    logger.debug(
                        "timing: first_byte %.0fms",
                        (time.monotonic() - t_start) * 1000,
                    )
                yield resp.content
            except Exception as exc:
                # Mid-stream upstream failure: the client sees a truncated
                # response; leave a trace so it is attributable.
                logger.warning(
                    "relay: upstream stream failed after %d bytes / %.1fs: %r",
                    n_bytes,
                    time.monotonic() - started,
                    exc,
                )
                raise
            except (GeneratorExit, BaseException):
                # Client hung up (or task cancelled) while we were relaying.
                # Normal client behavior (Codex closes the stream as soon as
                # it has response.completed; users Ctrl-C) — debug, not a
                # warning: nothing here is attributable to cliff or upstream.
                logger.debug(
                    "relay: client disconnected after %d bytes / %.1fs",
                    n_bytes,
                    time.monotonic() - started,
                )
                raise

        return StreamingResponse(
            body(),
            status_code=resp.status_code,
            headers=headers,
            background=BackgroundTask(resp.aclose),
        )

    async def handle(request: Request) -> Response:
        t0 = time.monotonic()
        raw = await request.body()
        t_read = time.monotonic()
        path = request.url.path
        upstream = pick_upstream(path)
        dialect = detect(path)

        ctx = None
        out_body = raw
        if dialect is not None and request.method == "POST" and raw:
            try:
                body = json.loads(raw)
                msgs = body.get(dialect.messages_key) if isinstance(body, dict) else None
                if (
                    isinstance(msgs, list)
                    and msgs
                    and all(isinstance(m, dict) for m in msgs)
                ):
                    ctx = engine.prepare(body, dialect)
                    if ctx.modified:
                        if cfg.shadow:
                            logger.info(
                                "shadow: would send ~%dk instead of ~%dk est tokens",
                                ctx.est_tokens_out // 1000,
                                ctx.est_tokens_in // 1000,
                            )
                        else:
                            out_body = json.dumps(
                                ctx.outgoing_body(), ensure_ascii=False
                            ).encode("utf-8")
            except Exception:
                logger.exception("prepare failed; passing through verbatim")
                ctx = None
                out_body = raw
            if cfg.debug_dir and ctx is not None:
                debug_dump(path, dialect, ctx)
            emit_request(ctx, dialect)

        # Strict mode: a request still over threshold with the ladder
        # exhausted never reaches the provider. The soft send is the right
        # default for real work, but it spends more context than the
        # configured budget — which a measurement run must not do silently.
        # Failing here is the point: the harness sees the run fail rather
        # than a quietly oversized turn. Inert under --shadow, which
        # modifies nothing by definition.
        if cfg.strict and not cfg.shadow and ctx is not None and ctx.over_budget:
            logger.warning(
                "strict: refusing request at ~%dk est tokens after escalation "
                "(budget %dk, rung %d)",
                ctx.est_tokens_out // 1000,
                cfg.threshold_tokens // 1000,
                ctx.rung,
            )
            return JSONResponse(
                {
                    "type": "error",
                    "error": {
                        "type": "cliff_over_budget",
                        "message": (
                            f"cliff strict mode: request is ~{ctx.est_tokens_out} "
                            f"est tokens with the escalation ladder exhausted "
                            f"(rung {ctx.rung}), over the configured budget of "
                            f"{cfg.threshold_tokens}. Refusing to forward."
                        ),
                    },
                },
                status_code=400,
            )

        t_prep = time.monotonic()
        try:
            resp = await send_upstream(request, upstream, out_body)
        except httpx.HTTPError as exc:
            logger.error("upstream error: %s", exc)
            return JSONResponse(
                {"error": {"type": "upstream_error", "message": str(exc)}},
                status_code=502,
            )
        logger.debug(
            "access: %s %s -> %d (%.0fms)",
            request.method,
            path,
            resp.status_code,
            (time.monotonic() - t0) * 1000,
        )
        if resp.status_code >= 400:
            logger.warning(
                "upstream returned %d for %s %s",
                resp.status_code,
                request.method,
                path,
            )
        if ctx is not None:
            logger.debug(
                "timing: read %.0fms, prepare %.0fms, upstream_headers %.0fms, status=%d",
                (t_read - t0) * 1000,
                (t_prep - t_read) * 1000,
                (time.monotonic() - t_prep) * 1000,
                resp.status_code,
            )

        # Reactive fallback: on a context-length 400, walk the escalation
        # ladder (base force-compact -> keep_recent=1 -> +caps -> truncated
        # summary), replaying after each rung until the provider accepts.
        if resp.status_code == 400 and ctx is not None and not cfg.shadow:
            data = await resp.aread()
            await resp.aclose()
            if is_context_error(data):
                try:
                    while engine.reactive(ctx):
                        retry_body = json.dumps(
                            ctx.outgoing_body(), ensure_ascii=False
                        ).encode("utf-8")
                        logger.info(
                            "reactive: replaying compacted request (rung %d)",
                            ctx.rung,
                        )
                        emit_request(ctx, dialect)
                        t_replay = time.monotonic()
                        resp2 = await send_upstream(request, upstream, retry_body)
                        logger.debug(
                            "access: %s %s -> %d (%.0fms, replay)",
                            request.method,
                            path,
                            resp2.status_code,
                            (time.monotonic() - t_replay) * 1000,
                        )
                        if resp2.status_code == 400:
                            data2 = await resp2.aread()
                            await resp2.aclose()
                            if is_context_error(data2):
                                data = data2  # still too big: next rung
                                continue
                            headers2 = {
                                k: v
                                for k, v in resp2.headers.items()
                                if k.lower() not in _STRIP_RESPONSE_HEADERS
                            }
                            return Response(
                                content=data2, status_code=400, headers=headers2
                            )
                        if resp2.status_code >= 400:
                            logger.warning(
                                "replay still failed: upstream returned %d",
                                resp2.status_code,
                            )
                        return relay(resp2, t_start=t0)
                    logger.warning("reactive: escalation exhausted; returning 400")
                except httpx.HTTPError as exc:
                    logger.error("upstream error on replay: %s", exc)
                except Exception:
                    logger.exception("reactive compaction failed; returning original 400")
            headers = {
                k: v
                for k, v in resp.headers.items()
                if k.lower() not in _STRIP_RESPONSE_HEADERS
            }
            return Response(content=data, status_code=400, headers=headers)

        return relay(resp, t_start=t0 if ctx is not None else None)

    async def status(_request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "name": "cliffcompaction",
                "version": __version__,
                "shadow": cfg.shadow,
                "strict": cfg.strict,
                "threshold_tokens": cfg.threshold_tokens,
                "keep_recent": cfg.keep_recent,
                "store_entries": len(engine.store),
                "sessions_tracked": len(seen_sids),
                # Installed under a running daemon: it is still serving the
                # code it started with, and looks perfectly healthy doing it.
                "code_stale": code_mtime() > started_at,
                "watchers": hub.subscribers,
                "uptime_s": int(time.time() - started_at),
            }
        )

    async def events(request: Request) -> StreamingResponse:
        """SSE stream for `cliff watch`: recent backlog, then live events."""
        q = hub.subscribe()
        backlog = hub.backlog()

        async def gen():
            try:
                for ev in backlog:
                    yield sse(ev)
                while True:
                    try:
                        ev = await asyncio.wait_for(q.get(), timeout=10.0)
                    except asyncio.TimeoutError:
                        yield b": keepalive\n\n"   # detect a gone client
                        continue
                    yield sse(ev)
            finally:
                hub.unsubscribe(q)

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"cache-control": "no-store", "x-accel-buffering": "no"},
        )

    @asynccontextmanager
    async def lifespan(_app: Starlette):
        try:
            yield
        finally:
            await client.aclose()

    app = Starlette(
        routes=[
            Route("/__cliff__/status", status, methods=["GET"]),
            Route("/__cliff__/events", events, methods=["GET"]),
            Route(
                "/{path:path}",
                handle,
                methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
            ),
        ],
        lifespan=lifespan,
    )
    app.state.hub = hub   # tests and introspection; not part of the request path
    return app
