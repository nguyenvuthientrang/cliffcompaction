"""OpenAI Chat Completions dialect.

Message shape notes:
- Tool results have their own role ("tool"), so classification is trivial.
- Assistant tool calls live in `tool_calls` (function name + JSON-string
  arguments); assistant `content` is the visible thought.
- System/developer messages found mid-conversation are treated like human
  text (kept verbatim, capped) — they are instructions, not observations.
"""

from __future__ import annotations

import hashlib

from ..config import Config
from ..hashing import canonical_json, digest_obj
from .base import SUMMARY_HEADER, Dialect, strip_task_notifications, truncate


# --- canonicalization -------------------------------------------------------


def _canon_part(part: dict):
    t = part.get("type", "")
    if t == "text":
        return ["text", part.get("text", "")]
    if t == "image_url":
        url = (part.get("image_url") or {}).get("url", "")
        return ["image", hashlib.sha256(str(url).encode("utf-8")).hexdigest()]
    return ["other", canonical_json(part)]


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        )
    return "" if content is None else str(content)


def digest_message(msg: dict) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        blocks = [["text", content]]
    elif isinstance(content, list):
        blocks = [_canon_part(p) for p in content if isinstance(p, dict)]
    else:
        blocks = []
    tool_calls = [
        [
            tc.get("id", ""),
            (tc.get("function") or {}).get("name", ""),
            (tc.get("function") or {}).get("arguments", ""),
        ]
        for tc in (msg.get("tool_calls") or [])
        if isinstance(tc, dict)
    ]
    return digest_obj(
        [
            msg.get("role", ""),
            blocks,
            tool_calls,
            msg.get("tool_call_id", ""),
            msg.get("name", ""),
        ]
    )


# --- classification ----------------------------------------------------------


def is_assistant(msg: dict) -> bool:
    return msg.get("role") == "assistant"


def is_summary_message(msg: dict) -> bool:
    return msg.get("role") == "user" and _content_text(msg.get("content")).startswith(
        SUMMARY_HEADER
    )


def session_key(body: dict) -> str | None:
    """No conversation id in this dialect. `user` is a per-user id, not a
    per-session one, so keying on it would merge a user's sessions."""
    return None


# --- summarization ------------------------------------------------------------


def _summarize_assistant(msg: dict, cfg: Config) -> list[str]:
    # Some OpenAI-compatible providers (DeepSeek, Kimi, ...) attach visible
    # reasoning as `reasoning_content`/`reasoning`; keep it as text.
    thinking = ""
    if cfg.keep_thinking:
        raw = msg.get("reasoning_content") or msg.get("reasoning") or ""
        if isinstance(raw, str):
            thinking = truncate(raw.strip(), cfg.thinking_max_chars)
    thought = truncate(_content_text(msg.get("content")).strip(), cfg.thought_max_chars)
    sigs: list[str] = []
    for tc in msg.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        args = fn.get("arguments", "")
        if not isinstance(args, str):
            args = canonical_json(args)
        sigs.append(f"[{fn.get('name', '?')}] {truncate(args, cfg.cmd_max_chars)}")
    lines: list[str] = []
    if thinking:
        lines.append(f"thinking: {thinking}")
    if thought:
        lines.append(f"assistant: {thought}")
    if sigs:
        lines.append("\n".join(sigs))
    return ["\n".join(lines)] if lines else []


def summarize_message(msg: dict, cfg: Config) -> list[str]:
    role = msg.get("role")
    if role == "assistant":
        return _summarize_assistant(msg, cfg)
    if role == "tool":
        text = _content_text(msg.get("content")).strip()
        if text and len(text) <= cfg.result_max_chars:
            return [f"result: {text}"]
        return []
    if role == "user":
        text = _content_text(msg.get("content")).strip()
        if not text or text.startswith(SUMMARY_HEADER):
            return []
        text = strip_task_notifications(text).strip()
        if not text:
            return []
        return [f"user: {truncate(text, cfg.human_max_chars)}"]
    if role in ("system", "developer"):
        text = _content_text(msg.get("content")).strip()
        if not text:
            return []
        return [f"{role}: {truncate(text, cfg.human_max_chars)}"]
    return []


def user_message(text: str) -> dict:
    return {"role": "user", "content": text}


DIALECT = Dialect(
    name="openai",
    digest_message=digest_message,
    is_assistant=is_assistant,
    summarize_message=summarize_message,
    user_message=user_message,
    is_summary_message=is_summary_message,
    session_key=session_key,
)
