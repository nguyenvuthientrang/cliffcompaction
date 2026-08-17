"""Anthropic Messages API dialect.

Message shape notes:
- `system` is a top-level request field, not a message — it passes through
  untouched and is not part of the hash chain.
- Tool results are user-role messages whose content blocks are typed
  `tool_result`; human prompts are user-role messages with `text` blocks.
  A single user message may mix both.
- `cache_control` markers move between requests (clients pin them to the
  newest message), so they are excluded from the canonical digest.
"""

from __future__ import annotations

import hashlib

from ..config import Config
from ..hashing import canonical_json, digest_obj
from .base import SUMMARY_HEADER, Dialect, strip_task_notifications, truncate


# --- canonicalization -------------------------------------------------------


def _result_text(content) -> str:
    """Extract the text of a tool_result's content (str or block list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
            elif isinstance(p, str):
                parts.append(p)
        return "\n".join(parts)
    return "" if content is None else str(content)


def _canon_block(block: dict):
    t = block.get("type", "")
    if t == "text":
        return ["text", block.get("text", "")]
    if t == "tool_use":
        return [
            "tool_use",
            block.get("id", ""),
            block.get("name", ""),
            canonical_json(block.get("input", {})),
        ]
    if t == "tool_result":
        return [
            "tool_result",
            block.get("tool_use_id", ""),
            _result_text(block.get("content", "")),
            bool(block.get("is_error")),
        ]
    if t == "thinking":
        # Content only — the signature is provider metadata.
        return ["thinking", block.get("thinking", "")]
    if t == "redacted_thinking":
        return ["redacted_thinking", block.get("data", "")]
    if t in ("image", "document"):
        src = block.get("source", {}) or {}
        payload = src.get("data") or src.get("url") or ""
        h = hashlib.sha256(str(payload).encode("utf-8")).hexdigest()
        return [t, src.get("type", ""), h]
    # Unknown block type: canonical dump minus volatile fields.
    reduced = {k: v for k, v in block.items() if k != "cache_control"}
    return ["other", canonical_json(reduced)]


def digest_message(msg: dict) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        blocks = [["text", content]]
    elif isinstance(content, list):
        blocks = [_canon_block(b) for b in content if isinstance(b, dict)]
    else:
        blocks = []
    return digest_obj([msg.get("role", ""), blocks])


# --- classification ----------------------------------------------------------


def is_assistant(msg: dict) -> bool:
    return msg.get("role") == "assistant"


def is_summary_message(msg: dict) -> bool:
    if msg.get("role") != "user":
        return False
    content = msg.get("content")
    if isinstance(content, str):
        return content.startswith(SUMMARY_HEADER)
    if isinstance(content, list):
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                return (b.get("text") or "").startswith(SUMMARY_HEADER)
    return False


# --- summarization ------------------------------------------------------------


def _summarize_assistant(msg: dict, cfg: Config) -> list[str]:
    content = msg.get("content")
    texts: list[str] = []
    thinkings: list[str] = []
    sigs: list[str] = []
    if isinstance(content, str):
        texts.append(content)
    elif isinstance(content, list):
        for b in content:
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text":
                texts.append(b.get("text", ""))
            elif t == "thinking":
                # Kept as TEXT: the signed block itself is never re-sent from
                # the compacted region, only its content.
                if cfg.keep_thinking:
                    thinkings.append(b.get("thinking", ""))
            elif t == "tool_use":
                args = canonical_json(b.get("input", {}))
                sigs.append(f"[{b.get('name', '?')}] {truncate(args, cfg.cmd_max_chars)}")
            # redacted_thinking (encrypted) / images: dropped from summaries
    lines: list[str] = []
    thinking = truncate("\n".join(x for x in thinkings if x.strip()).strip(), cfg.thinking_max_chars)
    if thinking:
        lines.append(f"thinking: {thinking}")
    thought = truncate("\n".join(t for t in texts if t.strip()).strip(), cfg.thought_max_chars)
    if thought:
        lines.append(f"assistant: {thought}")
    if sigs:
        lines.append("\n".join(sigs))
    return ["\n".join(lines)] if lines else []


def _summarize_user(msg: dict, cfg: Config) -> list[str]:
    """Human text: verbatim (sanity-capped). Tool results: keep iff short.
    A prior cliff summary: dropped entirely (never merged forward)."""
    parts: list[str] = []
    content = msg.get("content")
    if isinstance(content, str):
        if content.startswith(SUMMARY_HEADER):
            return []
        text = strip_task_notifications(content)
        if text.strip():
            parts.append(f"user: {truncate(text.strip(), cfg.human_max_chars)}")
        return parts
    if not isinstance(content, list):
        return parts
    for b in content:
        if not isinstance(b, dict):
            continue
        t = b.get("type")
        if t == "text":
            text = b.get("text", "")
            if text.startswith(SUMMARY_HEADER):
                continue
            text = strip_task_notifications(text)
            if not text.strip():
                continue
            parts.append(f"user: {truncate(text.strip(), cfg.human_max_chars)}")
        elif t == "tool_result":
            text = _result_text(b.get("content", "")).strip()
            if text and len(text) <= cfg.result_max_chars:
                parts.append(f"result: {text}")
            # long or empty observations dropped entirely
        # images/documents in the compacted region: dropped
    return parts


def summarize_message(msg: dict, cfg: Config) -> list[str]:
    if is_assistant(msg):
        return _summarize_assistant(msg, cfg)
    if msg.get("role") == "user":
        return _summarize_user(msg, cfg)
    return []


def user_message(text: str) -> dict:
    return {"role": "user", "content": text}


DIALECT = Dialect(
    name="anthropic",
    digest_message=digest_message,
    is_assistant=is_assistant,
    summarize_message=summarize_message,
    user_message=user_message,
    is_summary_message=is_summary_message,
)
