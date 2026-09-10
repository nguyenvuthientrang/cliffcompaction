"""Dialect interface."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from ..config import Config

# The marker by which a previously injected summary is recognized (and dropped
# on re-compaction). Keep it stable: changing a byte breaks recognition of
# summaries produced by earlier versions.
SUMMARY_HEADER = (
    "The following is a summary of your previous actions "
    "(long observations omitted):"
)


# Harness-injected subagent reports inside user messages. Stripped from the
# compacted region only (summaries): the assistant's next message always
# restates the result, and the output-file paths are temporary.
_TASK_NOTIFICATION_RE = re.compile(
    r"<task-notification>.*?</task-notification>\s*", re.DOTALL
)


def strip_task_notifications(text: str) -> str:
    return _TASK_NOTIFICATION_RE.sub("", text)


def truncate(text: str, max_chars: int) -> str:
    """Truncate with ellipsis; max_chars <= 0 means unlimited."""
    text = text or ""
    if max_chars and max_chars > 0 and len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


@dataclass(frozen=True)
class Dialect:
    """Operations over raw request-body message dicts. Kept messages are
    never rebuilt, only selected."""

    name: str
    # Canonical content digest of one message (volatile fields excluded).
    digest_message: Callable[[dict], str]
    # True if this message is a model turn (starts an assistant-step turn).
    is_assistant: Callable[[dict], bool]
    # Summarize one message into zero or more summary parts.
    summarize_message: Callable[[dict, Config], list[str]]
    # Build the synthetic summary message.
    user_message: Callable[[str], dict]
    # True if this message IS a previously injected cliff summary.
    is_summary_message: Callable[[dict], bool]
    # A stable per-conversation key the CLIENT supplied, or None. Display only
    # (`cliff watch`); nothing in the request path reads it.
    session_key: Callable[[dict], str | None]
    # Request-body key holding the message list ("messages"; "input" for the
    # Responses API).
    messages_key: str = "messages"
    # Optional dialect-specific turn grouping. None = the default rule (every
    # assistant message starts a turn). Dialects where one model step spans
    # several items (Responses: reasoning + message/function_call) override
    # this so a step is never split across the keep_recent boundary.
    group_turns: Callable[[list[dict]], list[list[dict]]] | None = None
    # Optional predicate: True for messages that must not sit at the end of
    # the verbatim head (the summary is inserted right after the head, and
    # some message kinds may not precede a user message — Anthropic system
    # messages must precede an assistant message or end the array). Matching
    # messages are trimmed off the head into the compacted region.
    trim_from_head: Callable[[dict], bool] | None = None
    # How many user messages a session's opening turn may carry, above which
    # a request with no model turn is a scaffold side call rather than a first
    # turn. Display only (`cliff watch`). One for a client that sends just the
    # person's message; Codex prepends an <environment_context> message, so a
    # genuine Responses opening turn carries two.
    opening_user_messages: int = 1
