"""Configuration. Settable via CLIFF_* environment variables; CLI flags
override."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


DEFAULT_ANTHROPIC_UPSTREAM = "https://api.anthropic.com"
DEFAULT_OPENAI_UPSTREAM = "https://api.openai.com"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Config:
    # --- compaction knobs ---
    # Proactive trigger: compact when the outgoing request exceeds this
    # (estimated at chars/4).
    threshold_tokens: int = 128_000
    # Number of recent assistant-step turns kept verbatim.
    keep_recent: int = 3
    # Max chars of assistant text kept per summarized turn. 0 = unlimited.
    # Set e.g. 300 for aggressive low-threshold configurations.
    thought_max_chars: int = 0
    # Max chars of a tool-call signature (serialized arguments).
    cmd_max_chars: int = 150
    # Tool results longer than this are dropped from the summary entirely.
    result_max_chars: int = 500
    # Sanity cap for human text inside the summary (protects against a
    # giant paste making the summary bigger than what it replaced).
    human_max_chars: int = 20_000
    # Keep thinking/reasoning content in summaries as text. Dropping it
    # entirely (False) is a legitimate leaner configuration. Signed thinking
    # BLOCKS are never re-sent from the compacted region either way — only
    # their text.
    keep_thinking: bool = True
    # Max chars of thinking text kept per summarized turn. Independent of
    # thought_max_chars. 0 = unlimited (default).
    thinking_max_chars: int = 0

    # --- proxy behavior ---
    # Shadow mode: observe, hash, log — but never modify a request.
    shadow: bool = False
    anthropic_upstream: str = DEFAULT_ANTHROPIC_UPSTREAM
    openai_upstream: str = DEFAULT_OPENAI_UPSTREAM
    host: str = "127.0.0.1"
    port: int = 8399

    # --- state store ---
    store_max_entries: int = 4096

    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            threshold_tokens=_env_int("CLIFF_THRESHOLD_TOKENS", 128_000),
            keep_recent=_env_int("CLIFF_KEEP_RECENT", 3),
            thought_max_chars=_env_int("CLIFF_THOUGHT_MAX_CHARS", 0),
            cmd_max_chars=_env_int("CLIFF_CMD_MAX_CHARS", 150),
            result_max_chars=_env_int("CLIFF_RESULT_MAX_CHARS", 500),
            human_max_chars=_env_int("CLIFF_HUMAN_MAX_CHARS", 20_000),
            keep_thinking=_env_bool("CLIFF_KEEP_THINKING", True),
            thinking_max_chars=_env_int("CLIFF_THINKING_MAX_CHARS", 0),
            shadow=_env_bool("CLIFF_SHADOW", False),
            anthropic_upstream=os.environ.get(
                "CLIFF_ANTHROPIC_UPSTREAM", DEFAULT_ANTHROPIC_UPSTREAM
            ),
            openai_upstream=os.environ.get(
                "CLIFF_OPENAI_UPSTREAM", DEFAULT_OPENAI_UPSTREAM
            ),
            host=os.environ.get("CLIFF_HOST", "127.0.0.1"),
            port=_env_int("CLIFF_PORT", 8399),
            store_max_entries=_env_int("CLIFF_STORE_MAX_ENTRIES", 4096),
        )
