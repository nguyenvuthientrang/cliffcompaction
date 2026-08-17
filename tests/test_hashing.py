"""Canonicalization: volatile fields must not affect digests."""

from cliffcompaction.dialects import anthropic, openai_chat
from cliffcompaction.hashing import chain_hashes

from util import a_assistant, a_result, a_user, o_assistant, o_tool, o_user


def test_cache_control_ignored():
    plain = {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    marked = {
        "role": "user",
        "content": [
            {"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}
        ],
    }
    assert anthropic.digest_message(plain) == anthropic.digest_message(marked)


def test_string_vs_block_content_equal():
    s = {"role": "user", "content": "hello"}
    b = {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    assert anthropic.digest_message(s) == anthropic.digest_message(b)
    assert openai_chat.digest_message(s) == openai_chat.digest_message(b)


def test_content_changes_digest():
    a = anthropic.digest_message(a_user("one"))
    b = anthropic.digest_message(a_user("two"))
    assert a != b


def test_tool_result_string_vs_blocks_equal():
    s = a_result("tu_1", "output text")
    b = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "tu_1",
                "content": [{"type": "text", "text": "output text"}],
            }
        ],
    }
    assert anthropic.digest_message(s) == anthropic.digest_message(b)


def test_chain_prefix_property():
    msgs = [a_user("task"), a_assistant("step", ("t1", "bash", {"command": "ls"}))]
    d = [anthropic.digest_message(m) for m in msgs]
    chain2 = chain_hashes(d)
    chain1 = chain_hashes(d[:1])
    assert chain2[0] == chain1[0]
    assert chain2[1] != chain2[0]


def test_openai_tool_call_identity():
    m1 = o_assistant("thinking", ("c1", "bash", {"command": "ls"}))
    m2 = o_assistant("thinking", ("c1", "bash", {"command": "ls"}))
    m3 = o_assistant("thinking", ("c1", "bash", {"command": "rm"}))
    assert openai_chat.digest_message(m1) == openai_chat.digest_message(m2)
    assert openai_chat.digest_message(m1) != openai_chat.digest_message(m3)
    assert openai_chat.digest_message(o_tool("c1", "ok")) != openai_chat.digest_message(
        o_user("ok")
    )
