"""Tests for mem_agent_rag: chunk_messages_by_turns and DB helper."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from fp_int_agents.agents.mem_agent_rag import chunk_messages_by_turns
from fp_int_agents.storage.db import (
    create_project,
    create_thread,
    get_thread,
    open_db,
    update_thread_memory_count,
)


def _turn(human: str, ai: str) -> list:
    return [HumanMessage(content=human), AIMessage(content=ai)]


# --- chunk_messages_by_turns ---


def test_chunk_empty() -> None:
    assert chunk_messages_by_turns([]) == []


def test_chunk_single_turn() -> None:
    msgs = _turn("hello", "hi")
    chunks = chunk_messages_by_turns(msgs)
    assert len(chunks) == 1
    first_idx, _last_idx, text = chunks[0]
    assert first_idx == 0
    assert "Human: hello" in text
    assert "Assistant: hi" in text


def test_chunk_windowing() -> None:
    # 6 turns (12 messages), chunk_turns=3, overlap_turns=1 → step=2
    msgs = []
    for i in range(6):
        msgs.extend(_turn(f"q{i}", f"a{i}"))

    chunks = chunk_messages_by_turns(msgs, chunk_turns=3, overlap_turns=1)
    # windows: [0,1,2], [2,3,4], [4,5] → 3 chunks
    assert len(chunks) == 3

    # adjacent chunks share 1 overlapping turn
    _, _last0, text0 = chunks[0]
    _first1, _, text1 = chunks[1]
    # turn index of last turn in chunk 0 should equal first turn of chunk 1
    # turn index = msg_index // 2 for HumanMessage starts
    assert "Assistant: a2" in text0
    assert "Human: q2" in text1


def test_chunk_skips_already_stored() -> None:
    # 6 turns already stored (start_message_index=12), add 2 more turns
    msgs = []
    for i in range(8):
        msgs.extend(_turn(f"q{i}", f"a{i}"))

    chunks = chunk_messages_by_turns(
        msgs, chunk_turns=3, overlap_turns=1, start_message_index=12
    )
    # All returned chunks must have at least one message at index >= 12
    for first_idx, last_idx, _ in chunks:
        assert last_idx >= 12


def test_chunk_skips_when_nothing_new() -> None:
    msgs = []
    for i in range(3):
        msgs.extend(_turn(f"q{i}", f"a{i}"))

    # start_message_index equals total messages → nothing new
    chunks = chunk_messages_by_turns(msgs, start_message_index=len(msgs))
    assert chunks == []


def test_chunk_trailing_human() -> None:
    # Odd-length: last turn has no AI reply
    msgs = _turn("q0", "a0") + [HumanMessage(content="q1")]
    chunks = chunk_messages_by_turns(msgs)
    assert len(chunks) >= 1
    _, _, text = chunks[-1]
    assert "Human: q1" in text


def test_chunk_non_human_ai_messages_included() -> None:
    # SystemMessage at start should be folded into first turn
    msgs = [SystemMessage(content="sys")] + _turn("q0", "a0")
    chunks = chunk_messages_by_turns(msgs)
    assert len(chunks) == 1
    _, _, text = chunks[0]
    assert "Human: q0" in text


# --- DB helper ---


@pytest.mark.asyncio
async def test_update_thread_memory_count(tmp_path) -> None:
    db_path = str(tmp_path / "app.db")
    async with open_db(db_path) as db:
        project = await create_project(db.conn, "P", "llama3.2")
        thread = await create_thread(db.conn, project.id)

        assert thread.memory_message_count is None

        await update_thread_memory_count(db.conn, thread.id, 10)
        reloaded = await get_thread(db.conn, thread.id)
        assert reloaded is not None
        assert reloaded.memory_message_count == 10

        await update_thread_memory_count(db.conn, thread.id, 22)
        reloaded2 = await get_thread(db.conn, thread.id)
        assert reloaded2 is not None
        assert reloaded2.memory_message_count == 22
