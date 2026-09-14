"""Thread title is set to the first human message on leave."""

import pytest
from fp_int_agents.storage.db import (
    create_project,
    create_thread,
    get_thread,
    open_db,
    update_thread_title,
)
from langchain_core.messages import AIMessage, HumanMessage


@pytest.mark.asyncio
async def test_update_thread_title(tmp_path) -> None:
    db_path = str(tmp_path / "app.db")
    async with open_db(db_path) as db:
        project = await create_project(db.conn, "P", "llama3.2")
        thread = await create_thread(db.conn, project.id)
        assert thread.title is None

        await update_thread_title(db.conn, thread.id, "Hello world")

        updated = await get_thread(db.conn, thread.id)
        assert updated is not None
        assert updated.title == "Hello world"


@pytest.mark.asyncio
async def test_update_thread_title_overwrite(tmp_path) -> None:
    db_path = str(tmp_path / "app.db")
    async with open_db(db_path) as db:
        project = await create_project(db.conn, "P", "llama3.2")
        thread = await create_thread(db.conn, project.id)

        await update_thread_title(db.conn, thread.id, "First title")
        await update_thread_title(db.conn, thread.id, "Second title")

        updated = await get_thread(db.conn, thread.id)
        assert updated is not None
        assert updated.title == "Second title"


def test_title_from_messages_first_human() -> None:
    from fp_int_agents.app import _title_from_messages

    messages = [
        HumanMessage(content="What is the capital of France?"),
        AIMessage(content="Paris."),
    ]
    assert _title_from_messages(messages) == "What is the capital of France?"


def test_title_from_messages_truncates() -> None:
    from fp_int_agents.app import _title_from_messages

    messages = [HumanMessage(content="x" * 200)]
    title = _title_from_messages(messages)
    assert len(title) <= 60
    assert title.endswith("…")


def test_title_from_messages_no_human() -> None:
    from fp_int_agents.app import _title_from_messages

    messages = [AIMessage(content="Hello!")]
    assert _title_from_messages(messages) is None


def test_title_from_messages_empty() -> None:
    from fp_int_agents.app import _title_from_messages

    assert _title_from_messages([]) is None
