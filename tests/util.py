"""Fixture builders: realistic Anthropic / OpenAI agent message lists."""

from __future__ import annotations

import json


# --- Anthropic Messages ------------------------------------------------------


def a_user(text: str) -> dict:
    return {"role": "user", "content": text}


def a_assistant(text: str, tool: tuple[str, str, dict] | None = None) -> dict:
    """Assistant message with optional (tool_use_id, name, input)."""
    content: list[dict] = []
    if text:
        content.append({"type": "text", "text": text})
    if tool:
        tid, name, inp = tool
        content.append({"type": "tool_use", "id": tid, "name": name, "input": inp})
    return {"role": "assistant", "content": content}


def a_result(tool_use_id: str, text: str) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": tool_use_id, "content": text}
        ],
    }


def a_session(n_turns: int, long_result_chars: int = 3000) -> list[dict]:
    """task + n_turns of (assistant tool call, tool result). Even turns get a
    long observation, odd turns a short one."""
    msgs = [a_user("Fix the failing test in repo X.")]
    for i in range(n_turns):
        tid = f"tu_{i}"
        msgs.append(
            a_assistant(
                f"Step {i}: I will inspect module {i} to find the bug.",
                (tid, "bash", {"command": f"pytest tests/test_{i}.py -x"}),
            )
        )
        if i % 2 == 0:
            msgs.append(a_result(tid, "X" * long_result_chars))
        else:
            msgs.append(a_result(tid, f"test_{i} passed (short output)"))
    return msgs


def a_body(msgs: list[dict], system: str = "You are a coding agent.") -> dict:
    return {
        "model": "claude-sonnet-5",
        "max_tokens": 4096,
        "system": system,
        "messages": msgs,
    }


# --- OpenAI Chat Completions -------------------------------------------------


def o_system(text: str) -> dict:
    return {"role": "system", "content": text}


def o_user(text: str) -> dict:
    return {"role": "user", "content": text}


def o_assistant(text: str, tool: tuple[str, str, dict] | None = None) -> dict:
    msg: dict = {"role": "assistant", "content": text}
    if tool:
        tid, name, args = tool
        msg["tool_calls"] = [
            {
                "id": tid,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ]
    return msg


def o_tool(tool_call_id: str, text: str) -> dict:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": text}


def o_session(n_turns: int, long_result_chars: int = 3000) -> list[dict]:
    msgs = [o_system("You are a coding agent."), o_user("Fix the failing test.")]
    for i in range(n_turns):
        tid = f"call_{i}"
        msgs.append(
            o_assistant(
                f"Step {i}: inspecting module {i}.",
                (tid, "bash", {"command": f"pytest tests/test_{i}.py -x"}),
            )
        )
        if i % 2 == 0:
            msgs.append(o_tool(tid, "Y" * long_result_chars))
        else:
            msgs.append(o_tool(tid, f"test_{i} passed"))
    return msgs


def o_body(msgs: list[dict]) -> dict:
    return {"model": "gpt-5", "messages": msgs}
