import asyncio

import aiosqlite
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from fp_int_agents.config import LlmConfig, Project
from fp_int_agents.llm.client import get_embedding_model
from fp_int_agents.storage import vectorstore

_DEFAULT_CHUNK_TURNS = 5
_DEFAULT_OVERLAP_TURNS = 1
_DEFAULT_EMBEDDING_MODEL = "multilingual-e5-small"


def chunk_messages_by_turns(
    messages: list[AnyMessage],
    chunk_turns: int = _DEFAULT_CHUNK_TURNS,
    overlap_turns: int = _DEFAULT_OVERLAP_TURNS,
    start_message_index: int = 0,
) -> list[tuple[int, int, str]]:
    """Return (first_msg_idx, last_msg_idx, rendered_text) for each new chunk.

    A turn starts at each HumanMessage; all messages until the next HumanMessage
    belong to that turn.  Chunks slide with overlap; turns entirely before
    start_message_index are skipped.
    """
    if not messages:
        return []

    # Group messages into turns: list of (start_idx, [messages])
    turns: list[tuple[int, list[AnyMessage]]] = []
    current: list[AnyMessage] = []
    current_start = 0
    for i, msg in enumerate(messages):
        if isinstance(msg, HumanMessage) and current:
            turns.append((current_start, current))
            current = [msg]
            current_start = i
        else:
            if not current:
                current_start = i
            current.append(msg)
    if current:
        turns.append((current_start, current))

    if not turns:
        return []

    first_new_turn = start_message_index // 2
    window_start = max(0, first_new_turn - overlap_turns)
    step = max(1, chunk_turns - overlap_turns)

    result: list[tuple[int, int, str]] = []
    i = window_start
    while i < len(turns):
        window = turns[i : i + chunk_turns]
        first_idx = window[0][0]
        last_msg_list = window[-1][1]
        last_idx = (
            turns[min(i + chunk_turns - 1, len(turns) - 1)][0] + len(last_msg_list) - 1
        )

        # Skip chunks that end entirely within already-stored range
        if last_idx < start_message_index:
            i += step
            continue

        lines: list[str] = []
        for _, turn_msgs in window:
            for msg in turn_msgs:
                if isinstance(msg, HumanMessage):
                    lines.append(f"Human: {msg.content}")
                elif isinstance(msg, AIMessage):
                    lines.append(f"Assistant: {msg.content}")
                else:
                    lines.append(f"[{msg.type}]: {msg.content}")
        result.append((first_idx, last_idx, "\n".join(lines)))
        i += step

    return result


async def memorize_thread(
    project: Project,
    messages: list[AnyMessage],
    llm_config: LlmConfig,
    conn: aiosqlite.Connection | None,
    memory_message_count: int,
    thread_id: str,
) -> tuple[str, int]:
    chunks = chunk_messages_by_turns(messages, start_message_index=memory_message_count)
    if not chunks:
        return "No new memory chunks.", memory_message_count

    embedding_model = (
        project.init_config.get("embedding_model") or _DEFAULT_EMBEDDING_MODEL
    )
    embeddings = get_embedding_model(
        base_url=llm_config.base_url,
        model=embedding_model,
        api_key=llm_config.api_key,
    )

    for first_idx, last_idx, text in chunks:
        title = f"turns {first_idx // 2 + 1}–{last_idx // 2 + 1}"
        await asyncio.to_thread(
            vectorstore.add_chunks,
            embeddings,
            project.id,
            thread_id,
            [text],
            title,
            None,
            "memory",
        )

    return f"Stored {len(chunks)} memory chunk(s).", len(messages)
