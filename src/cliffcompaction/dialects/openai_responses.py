"""OpenAI Responses API dialect (stateless clients: full `input` resent).

Item shape notes:
- History rides in `input` (not `messages`); the system prompt rides in the
  separate `instructions` field, which is never touched here.
- A model step spans MULTIPLE items: [reasoning?] then message/function_call,
  followed by function_call_output items. Providers reject a function_call
  whose paired reasoning item is missing, so turn grouping keeps contiguous
  model-output runs together (see group_turns below).
- Reasoning items carry `encrypted_content` (opaque; dropped with their turn)
  and optionally readable `summary` blocks, which fold into summaries as
  thinking text.
- Item `id` and `status` are excluded from digests: ids are stripped by some
  clients for non-OpenAI upstreams, and status is lifecycle metadata.
"""

from __future__ import annotations

from ..config import Config
from ..hashing import canonical_json, digest_obj
from .base import SUMMARY_HEADER, Dialect, strip_task_notifications, truncate

# Item types produced by the model (as opposed to inputs like
# function_call_output or user messages). A contiguous run of these starts a
# turn; an assistant-role message counts via _is_model_output.
_MODEL_ITEM_TYPES = {
    "reasoning",
    "function_call",
    "custom_tool_call",
    "local_shell_call",
    "web_search_call",
    "tool_search_call",
    "image_generation_call",
}


def _item_type(item: dict) -> str:
    t = item.get("type")
    if t:
        return t
    # Bare {role, content} items are legal input; treat as messages.
    return "message" if "role" in item else "other"


def _is_model_output(item: dict) -> bool:
    t = _item_type(item)
    if t == "message":
        return item.get("role") == "assistant"
    return t in _MODEL_ITEM_TYPES


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict)
            and p.get("type") in ("input_text", "output_text", "text")
        )
    return "" if content is None else str(content)


# --- canonicalization -------------------------------------------------------


def _canon_content(content):
    if isinstance(content, str):
        return [["text", content]]
    if isinstance(content, list):
        out = []
        for p in content:
            if not isinstance(p, dict):
                continue
            t = p.get("type", "")
            if t in ("input_text", "output_text", "text"):
                out.append(["text", p.get("text", "")])
            else:
                out.append(["other", canonical_json(p)])
        return out
    return []


def digest_message(item: dict) -> str:
    t = _item_type(item)
    if t == "message":
        return digest_obj(["message", item.get("role", ""), _canon_content(item.get("content"))])
    if t == "reasoning":
        return digest_obj(
            [
                "reasoning",
                item.get("encrypted_content") or "",
                canonical_json(item.get("summary") or []),
            ]
        )
    if t == "function_call_output":
        out = item.get("output")
        if not isinstance(out, str):
            out = canonical_json(out)
        return digest_obj(["function_call_output", item.get("call_id", ""), out])
    if t.endswith("_call") or t == "function_call":
        args = item.get("arguments", "")
        if not isinstance(args, str):
            args = canonical_json(args)
        return digest_obj([t, item.get("call_id", ""), item.get("name", ""), args])
    # Unknown item types: canonical dump minus volatile fields.
    stripped = {k: v for k, v in item.items() if k not in ("id", "status")}
    return digest_obj(["other", canonical_json(stripped)])


# --- classification ----------------------------------------------------------


def is_assistant(item: dict) -> bool:
    return _is_model_output(item)


def is_summary_message(item: dict) -> bool:
    return (
        _item_type(item) == "message"
        and item.get("role") == "user"
        and _content_text(item.get("content")).startswith(SUMMARY_HEADER)
    )


def group_turns(items: list[dict]) -> list[list[dict]]:
    """A turn starts at a model-output item whose predecessor is not one, so
    a step's [reasoning, message/function_call] run stays in one turn with
    its function_call_output observations. Leading non-model items form their
    own group (same contract as the base grouping)."""
    turns: list[list[dict]] = []
    current: list[dict] | None = None
    prev_model = False
    for item in items:
        model = _is_model_output(item)
        if model and not prev_model:
            if current is not None:
                turns.append(current)
            current = [item]
        elif current is None:
            current = [item]
        else:
            current.append(item)
        prev_model = model
    if current is not None:
        turns.append(current)
    return turns


# --- summarization ------------------------------------------------------------


def summarize_message(item: dict, cfg: Config) -> list[str]:
    t = _item_type(item)
    if t == "message":
        role = item.get("role", "")
        text = _content_text(item.get("content")).strip()
        if not text:
            return []
        if role == "assistant":
            return [f"assistant: {truncate(text, cfg.thought_max_chars)}"]
        if text.startswith(SUMMARY_HEADER):
            return []
        if role == "user":
            text = strip_task_notifications(text).strip()
            if not text:
                return []
            return [f"user: {truncate(text, cfg.human_max_chars)}"]
        return [f"{role}: {truncate(text, cfg.human_max_chars)}"]
    if t == "reasoning":
        if not cfg.keep_thinking:
            return []
        text = "\n".join(
            p.get("text", "")
            for p in item.get("summary") or []
            if isinstance(p, dict)
        ).strip()
        if not text:
            return []  # encrypted-only reasoning: nothing foldable
        return [f"thinking: {truncate(text, cfg.thinking_max_chars)}"]
    if t == "function_call_output":
        out = item.get("output")
        if not isinstance(out, str):
            out = _content_text(out) or canonical_json(out)
        out = out.strip()
        if out and len(out) <= cfg.result_max_chars:
            return [f"result: {out}"]
        return []
    if t.endswith("_call"):
        args = item.get("arguments", "")
        if not isinstance(args, str):
            args = canonical_json(args)
        name = item.get("name") or t
        return [f"[{name}] {truncate(args, cfg.cmd_max_chars)}"]
    return []


def user_message(text: str) -> dict:
    return {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": text}],
    }


DIALECT = Dialect(
    name="openai-responses",
    digest_message=digest_message,
    is_assistant=is_assistant,
    summarize_message=summarize_message,
    user_message=user_message,
    is_summary_message=is_summary_message,
    messages_key="input",
    group_turns=group_turns,
)
