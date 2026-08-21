"""API dialects: Anthropic Messages, OpenAI Chat Completions, OpenAI Responses."""

from __future__ import annotations

from . import anthropic, openai_chat, openai_responses
from .base import Dialect


def detect(path: str) -> Dialect | None:
    """Pick the dialect for a request path, or None for raw passthrough.

    Note /v1/messages/count_tokens is deliberately NOT matched: token-count
    probes pass through verbatim (compacting them would falsify counts the
    scaffold relies on without saving any real tokens). Same for
    /responses/compact (a provider-side compaction endpoint, not a turn).
    """
    p = path.rstrip("/")
    if p.endswith("/v1/messages") or p.endswith("/messages"):
        return anthropic.DIALECT
    if p.endswith("/chat/completions"):
        return openai_chat.DIALECT
    if p.endswith("/responses"):
        return openai_responses.DIALECT
    return None


__all__ = ["Dialect", "detect", "anthropic", "openai_chat", "openai_responses"]
