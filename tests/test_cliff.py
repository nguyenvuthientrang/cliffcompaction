"""Core compaction semantics, both dialects."""

from cliffcompaction.cliff import compact, group_turns
from cliffcompaction.config import Config
from cliffcompaction.dialects.anthropic import DIALECT as ANTHROPIC
from cliffcompaction.dialects.base import SUMMARY_HEADER
from cliffcompaction.dialects.openai_chat import DIALECT as OPENAI

from util import a_assistant, a_result, a_session, a_user, o_session


def cfg(**kw) -> Config:
    return Config(**kw)


# --- structure ----------------------------------------------------------------


def test_basic_structure_anthropic():
    msgs = a_session(10)
    res = compact(msgs, ANTHROPIC, cfg(keep_recent=2))
    assert res is not None
    out = res.messages
    # head (task) verbatim
    assert out[0] is msgs[0]
    # one summary message
    assert out[1]["role"] == "user"
    assert out[1]["content"].startswith(SUMMARY_HEADER)
    # last 2 turns (4 messages) verbatim, same objects
    assert out[2:] == msgs[-4:]
    assert out[2] is msgs[-4]
    # cut points at the kept tail in input coordinates
    assert msgs[res.cut :] == msgs[-4:]
    # tail begins with an assistant message (turn boundary, no orphan tool_result)
    assert out[2]["role"] == "assistant"


def test_basic_structure_openai():
    msgs = o_session(10)
    res = compact(msgs, OPENAI, cfg(keep_recent=2))
    assert res is not None
    out = res.messages
    # head = system + task, verbatim
    assert out[0] is msgs[0] and out[1] is msgs[1]
    assert out[2]["content"].startswith(SUMMARY_HEADER)
    assert out[3:] == msgs[-4:]
    assert out[3]["role"] == "assistant"


def test_nothing_to_compact():
    msgs = a_session(2)
    assert compact(msgs, ANTHROPIC, cfg(keep_recent=3)) is None


def test_no_assistant_yet():
    msgs = [a_user("task")]
    assert compact(msgs, ANTHROPIC, cfg(keep_recent=1)) is None


# --- content classes -----------------------------------------------------------


def test_long_results_dropped_short_kept():
    msgs = a_session(10, long_result_chars=3000)
    res = compact(msgs, ANTHROPIC, cfg(keep_recent=1))
    summary = res.messages[1]["content"]
    assert "X" * 600 not in summary  # long observations gone
    assert "result: test_1 passed (short output)" in summary


def test_tool_signatures_present_and_truncated():
    msgs = a_session(6)
    res = compact(msgs, ANTHROPIC, cfg(keep_recent=1, cmd_max_chars=20))
    summary = res.messages[1]["content"]
    assert "[bash]" in summary
    # signature truncated to 20 chars + ellipsis
    for line in summary.splitlines():
        if line.startswith("[bash] "):
            assert len(line) <= len("[bash] ") + 23


def test_assistant_text_full_by_default():
    long_thought = "T" * 2000
    msgs = [
        a_user("task"),
        a_assistant(long_thought, ("t0", "bash", {"command": "ls"})),
        a_result("t0", "ok"),
        a_assistant("done", None),
        a_result("t0", "bye"),
    ]
    res = compact(msgs, ANTHROPIC, cfg(keep_recent=1))
    assert long_thought in res.messages[1]["content"]


def test_assistant_text_capped_when_configured():
    long_thought = "T" * 2000
    msgs = [
        a_user("task"),
        a_assistant(long_thought, ("t0", "bash", {"command": "ls"})),
        a_result("t0", "ok"),
        a_assistant("done", None),
        a_result("t0", "bye"),
    ]
    res = compact(msgs, ANTHROPIC, cfg(keep_recent=1, thought_max_chars=300))
    summary = res.messages[1]["content"]
    assert long_thought not in summary
    assert "T" * 300 + "..." in summary


def test_human_text_kept_verbatim():
    msgs = a_session(8)
    human = "IMPORTANT: use the staging database, never prod. " * 20  # > 500 chars
    msgs.insert(5, a_user(human))
    res = compact(msgs, ANTHROPIC, cfg(keep_recent=1))
    assert human.strip() in res.messages[1]["content"]


def test_mixed_message_human_text_and_tool_result():
    long_out = "Z" * 5000
    mixed = {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "t0", "content": long_out},
            {"type": "text", "text": "actually, stop and use branch dev-2"},
        ],
    }
    msgs = [
        a_user("task"),
        a_assistant("step 0", ("t0", "bash", {"command": "ls"})),
        mixed,
        a_assistant("step 1", ("t1", "bash", {"command": "pwd"})),
        a_result("t1", "ok"),
        a_assistant("step 2", ("t2", "bash", {"command": "id"})),
        a_result("t2", "ok"),
    ]
    res = compact(msgs, ANTHROPIC, cfg(keep_recent=1))
    summary = res.messages[1]["content"]
    assert "actually, stop and use branch dev-2" in summary
    assert long_out not in summary


def _thinking_session() -> list[dict]:
    msgs = [a_user("task")]
    asst = {
        "role": "assistant",
        "content": [
            {
                "type": "thinking",
                "thinking": "The mock is not reset between cases, that is the real bug.",
                "signature": "EqQBCgIYAsignedblob",
            },
            {"type": "text", "text": "step 0 visible text"},
            {"type": "tool_use", "id": "t0", "name": "bash", "input": {"command": "ls"}},
        ],
    }
    msgs.append(asst)
    msgs.append(a_result("t0", "ok"))
    msgs.append(a_assistant("step 1", ("t1", "bash", {"command": "pwd"})))
    msgs.append(a_result("t1", "ok"))
    msgs.append(a_assistant("step 2", None))
    msgs.append(a_result("t2", "ok"))
    return msgs


def test_thinking_kept_as_text_in_summary():
    res = compact(_thinking_session(), ANTHROPIC, cfg(keep_recent=1))
    summary = res.messages[1]["content"]
    assert "thinking: The mock is not reset between cases" in summary
    assert "step 0 visible text" in summary
    assert "signedblob" not in summary  # the signed block itself is never re-sent


def test_thinking_cap_independent_of_thought_cap():
    # The thought cap must NOT touch thinking...
    res = compact(_thinking_session(), ANTHROPIC, cfg(keep_recent=1, thought_max_chars=10))
    summary = res.messages[1]["content"]
    assert "thinking: The mock is not reset between cases, that is the real bug." in summary
    assert "step 0 vis..." in summary  # visible thought capped at 10
    # ...and the thinking cap must not touch the visible thought.
    res2 = compact(_thinking_session(), ANTHROPIC, cfg(keep_recent=1, thinking_max_chars=12))
    summary2 = res2.messages[1]["content"]
    assert "thinking: The mock is ..." in summary2
    assert "step 0 visible text" in summary2


def test_thinking_droppable():
    res = compact(_thinking_session(), ANTHROPIC, cfg(keep_recent=1, keep_thinking=False))
    summary = res.messages[1]["content"]
    assert "thinking:" not in summary
    assert "step 0 visible text" in summary


def test_openai_reasoning_content_kept_as_text():
    from util import o_session

    msgs = o_session(6)
    msgs[2]["reasoning_content"] = "I should check the failing test first."
    res = compact(msgs, OPENAI, cfg(keep_recent=1))
    summary = res.messages[2]["content"]
    assert "thinking: I should check the failing test first." in summary


def test_speaker_tags_in_summary():
    msgs = a_session(8)
    msgs.insert(5, a_user("please target the dev branch"))
    res = compact(msgs, ANTHROPIC, cfg(keep_recent=1))
    summary = res.messages[1]["content"]
    assert "user: please target the dev branch" in summary
    assert "assistant: Step 0:" in summary
    assert "result: test_1 passed" in summary  # results keep their own tag


def test_task_notifications_stripped_from_summary():
    notif = (
        "<task-notification>\n<task-id>abc</task-id>\n"
        "<result>" + "R" * 3000 + "</result>\n</task-notification>"
    )
    msgs = [
        a_user("task"),
        a_assistant("step 0", ("t0", "bash", {"command": "ls"})),
        a_user(notif + "\nalso: please use the dev branch"),
        a_assistant("step 1", ("t1", "bash", {"command": "pwd"})),
        a_result("t1", "ok"),
        a_assistant("step 2", None),
        a_result("t2", "ok"),
    ]
    res = compact(msgs, ANTHROPIC, cfg(keep_recent=1))
    summary = res.messages[1]["content"]
    assert "task-notification" not in summary
    assert "RRRR" not in summary
    assert "also: please use the dev branch" in summary  # surrounding text kept
    # a notification-only user message contributes nothing
    msgs2 = list(msgs)
    msgs2[2] = a_user(notif)
    res2 = compact(msgs2, ANTHROPIC, cfg(keep_recent=1))
    assert "task-notification" not in res2.messages[1]["content"]


def test_images_dropped_from_summary():
    img_msg = {
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
            },
            {"type": "text", "text": "see the screenshot"},
        ],
    }
    msgs = [
        a_user("task"),
        a_assistant("step 0", ("t0", "bash", {"command": "ls"})),
        img_msg,
        a_assistant("step 1", ("t1", "bash", {"command": "pwd"})),
        a_result("t1", "ok"),
        a_assistant("step 2", None),
        a_result("t2", "ok"),
    ]
    res = compact(msgs, ANTHROPIC, cfg(keep_recent=1))
    summary = res.messages[1]["content"]
    assert "AAAA" not in summary
    assert "see the screenshot" in summary


# --- re-compaction semantics -----------------------------------------------------


def test_recompaction_stays_flat():
    msgs = a_session(10)
    res1 = compact(msgs, ANTHROPIC, cfg(keep_recent=1))
    # scaffold-side: history keeps growing; proxy-side: C + new tail
    grown = list(res1.messages)
    for i in range(10, 16):
        tid = f"tu_{i}"
        grown.append(a_assistant(f"Step {i}", (tid, "bash", {"command": f"cmd {i}"})))
        grown.append(a_result(tid, "K" * 2000))
    res2 = compact(grown, ANTHROPIC, cfg(keep_recent=1))
    assert res2 is not None
    out = res2.messages
    headers = [
        m
        for m in out
        if m["role"] == "user"
        and isinstance(m["content"], str)
        and m["content"].startswith(SUMMARY_HEADER)
    ]
    assert len(headers) == 1  # exactly one summary — never nested
    # old summary content is NOT merged forward
    assert "Step 0" not in headers[0]["content"]
    assert "Step 14" in headers[0]["content"]
    # head survived both compactions
    assert out[0] is msgs[0]


def test_turn_grouping_orphan_summary():
    msgs = a_session(4)
    res = compact(msgs, ANTHROPIC, cfg(keep_recent=1))
    turns = group_turns(res.messages[1:], ANTHROPIC)
    # first group is the summary (non-assistant orphan), rest assistant-led
    assert turns[0][0]["content"].startswith(SUMMARY_HEADER)
    assert all(t[0]["role"] == "assistant" for t in turns[1:])


def test_system_messages_never_precede_the_summary():
    """A content-ful in-array system message may only precede an assistant
    message or end the array. Head trimming must keep that invariant when
    the user-role summary is injected (real incident: Claude Code session
    with history [user, system, assistant, ...])."""
    from cliffcompaction.dialects.anthropic import DIALECT as A

    msgs = [
        {"role": "user", "content": "the task"},
        {"role": "system", "content": "directive: be careful"},
    ]
    for i in range(6):
        msgs.append(
            {"role": "assistant", "content": [
                {"type": "text", "text": f"step {i}"},
                {"type": "tool_use", "id": f"t{i}", "name": "bash", "input": {"command": f"make {i}"}},
            ]}
        )
        msgs.append(
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": f"t{i}", "content": "R" * 2000}
            ]}
        )
    res = compact(msgs, A, Config(keep_recent=2))
    assert res is not None
    # system message trimmed out of the head...
    assert res.head_len == 1
    assert res.messages[0]["role"] == "user"
    assert res.messages[1]["content"].startswith(SUMMARY_HEADER)
    # ...its content folded into the summary...
    assert "system: directive: be careful" in res.messages[1]["content"]
    # ...and the output violates no placement rule: every content-ful system
    # message precedes an assistant message or ends the array.
    for i, m in enumerate(res.messages):
        if m.get("role") == "system" and m.get("content"):
            assert i == len(res.messages) - 1 or res.messages[i + 1]["role"] == "assistant"


def test_system_message_in_tail_and_directive_only_dropped():
    from cliffcompaction.dialects.anthropic import DIALECT as A

    msgs = [{"role": "user", "content": "task"}]
    for i in range(5):
        msgs.append({"role": "assistant", "content": f"step {i}"})
        msgs.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": f"t{i}", "content": "R" * 1500}
        ]})
        if i == 1:
            # directive-only system message mid-history (allowed anywhere)
            msgs.append({"role": "system", "content": []})
    res = compact(msgs, A, Config(keep_recent=2))
    assert res is not None
    text = res.messages[res.head_len]["content"]
    assert "system:" not in text  # directive-only folds to nothing
    # kept tail preserved verbatim
    assert res.messages[-1] == msgs[-1]
