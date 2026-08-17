"""API dialects: Anthropic Messages and OpenAI Chat Completions."""

from __future__ import annotations

from . import anthropic, openai_chat
from .base import Dialect


def detect(path: str) -> Dialect | None:
    """Pick the dialect for a request path, or None for raw passthrough.

    Note /v1/messages/count_tokens is deliberately NOT matched: token-count
    probes pass through verbatim (compacting them would falsify counts the
    scaffold relies on without saving any real tokens).
    """
    p = path.rstrip("/")
    if p.endswith("/v1/messages") or p.endswith("/messages"):
        return anthropic.DIALECT
    if p.endswith("/chat/completions"):
        return openai_chat.DIALECT
    return None


__all__ = ["Dialect", "detect", "anthropic", "openai_chat"]
