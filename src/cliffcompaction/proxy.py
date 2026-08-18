"""The transparent proxy.

Fail-open: any failure (unparseable body, unknown path, engine error)
results in verbatim passthrough. Shadow mode runs the full pipeline but
forwards every request verbatim.
"""

from __future__ import annotations

import json
import logging
import re
import time
from contextlib import asynccontextmanager

import httpx
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from . import __version__
from .config import Config
from .dialects import detect
from .engine import Engine

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


def create_app(
    cfg: Config,
    engine: Engine | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Starlette:
    engine = engine or Engine(cfg)
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=30.0, read=600.0, write=600.0, pool=30.0),
        transport=transport,
    )

    def pick_upstream(path: str) -> str:
        if path.rstrip("/").endswith("/chat/completions"):
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
                logger.warning(
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
                msgs = body.get("messages") if isinstance(body, dict) else None
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

        # Reactive fallback: on a context-length 400, compact and replay once.
        if resp.status_code == 400 and ctx is not None and not cfg.shadow:
            data = await resp.aread()
            await resp.aclose()
            if is_context_error(data):
                try:
                    if engine.reactive(ctx):
                        retry_body = json.dumps(
                            ctx.outgoing_body(), ensure_ascii=False
                        ).encode("utf-8")
                        logger.info("reactive: replaying compacted request")
                        t_replay = time.monotonic()
                        resp2 = await send_upstream(request, upstream, retry_body)
                        logger.debug(
                            "access: %s %s -> %d (%.0fms, replay)",
                            request.method,
                            path,
                            resp2.status_code,
                            (time.monotonic() - t_replay) * 1000,
                        )
                        if resp2.status_code >= 400:
                            logger.warning(
                                "replay still failed: upstream returned %d",
                                resp2.status_code,
                            )
                        return relay(resp2, t_start=t0)
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
                "threshold_tokens": cfg.threshold_tokens,
                "keep_recent": cfg.keep_recent,
                "store_entries": len(engine.store),
            }
        )

    @asynccontextmanager
    async def lifespan(_app: Starlette):
        try:
            yield
        finally:
            await client.aclose()

    return Starlette(
        routes=[
            Route("/__cliff__/status", status, methods=["GET"]),
            Route(
                "/{path:path}",
                handle,
                methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
            ),
        ],
        lifespan=lifespan,
    )
