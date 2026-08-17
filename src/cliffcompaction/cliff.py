from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .dialects.base import SUMMARY_HEADER, Dialect


@dataclass
class CompactResult:
    messages: list[dict]  # the compacted message list
    head_len: int  # number of verbatim head messages
    summary: dict  # the synthetic summary message
    cut: int  # index into the INPUT list: messages[cut:] were kept verbatim


def group_turns(body: list[dict], dialect: Dialect) -> list[list[dict]]:
    """Group messages into assistant-step turns.

    A turn starts at each assistant message and includes the following
    non-assistant messages (its observations). Leading non-assistant
    messages (e.g. a prior summary) form their own group so they can be
    dropped cleanly on re-compaction.
    """
    turns: list[list[dict]] = []
    current: list[dict] | None = None
    for msg in body:
        if dialect.is_assistant(msg):
            if current is not None:
                turns.append(current)
            current = [msg]
        elif current is None:
            current = [msg]
        else:
            current.append(msg)
    if current is not None:
        turns.append(current)
    return turns


def compact(messages: list[dict], dialect: Dialect, cfg: Config) -> CompactResult | None:
    """Compact a message list. Returns None if there is nothing to gain.

    Does not mutate the input; kept messages are passed by reference,
    never rebuilt.
    """
    # Head: everything before the first assistant message...
    first_assistant = next(
        (i for i, m in enumerate(messages) if dialect.is_assistant(m)), None
    )
    if first_assistant is None:
        return None
    head_len = first_assistant
    # ...minus trailing summary messages (re-compaction must not absorb them).
    while head_len > 0 and dialect.is_summary_message(messages[head_len - 1]):
        head_len -= 1

    body = messages[head_len:]
    turns = group_turns(body, dialect)

    keep_recent = max(0, cfg.keep_recent)
    if len(turns) <= keep_recent:
        return None
    to_compact = turns[: len(turns) - keep_recent]
    to_keep = turns[len(turns) - keep_recent :]

    parts: list[str] = []
    for turn in to_compact:
        for msg in turn:
            parts.extend(dialect.summarize_message(msg, cfg))

    summary_text = (
        SUMMARY_HEADER + "\n\n" + "\n\n---\n\n".join(parts) if parts else SUMMARY_HEADER
    )
    summary = dialect.user_message(summary_text)

    kept: list[dict] = []
    for turn in to_keep:
        kept.extend(turn)

    new_messages = [*messages[:head_len], summary, *kept]
    if len(new_messages) >= len(messages):
        return None  # no reduction — treat as nothing to compact

    cut = len(messages) - len(kept)
    return CompactResult(messages=new_messages, head_len=head_len, summary=summary, cut=cut)
