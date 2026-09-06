"""Thread settings (tools + chat_model) persist and reload correctly."""

import pytest

from fp_int_agents.storage.db import (
    create_project,
    create_thread,
    get_thread,
    open_db,
    update_thread_settings,
)


@pytest.mark.asyncio
async def test_chat_model_persists(tmp_path) -> None:
    db_path = str(tmp_path / "app.db")
    async with open_db(db_path) as db:
        project = await create_project(db.conn, "P", "llama3.2")
        thread = await create_thread(db.conn, project.id, "T")

        await update_thread_settings(db.conn, thread.id, ["tool_a"], "mistral")

        reloaded = await get_thread(db.conn, thread.id)
        assert reloaded is not None
        assert reloaded.chat_model == "mistral"
        assert reloaded.active_tools == ["tool_a"]


@pytest.mark.asyncio
async def test_chat_model_none_by_default(tmp_path) -> None:
    db_path = str(tmp_path / "app.db")
    async with open_db(db_path) as db:
        project = await create_project(db.conn, "P", "llama3.2")
        thread = await create_thread(db.conn, project.id, "T")

        reloaded = await get_thread(db.conn, thread.id)
        assert reloaded is not None
        assert reloaded.chat_model is None


@pytest.mark.asyncio
async def test_settings_cleared_when_model_none(tmp_path) -> None:
    db_path = str(tmp_path / "app.db")
    async with open_db(db_path) as db:
        project = await create_project(db.conn, "P", "llama3.2")
        thread = await create_thread(db.conn, project.id, "T")

        await update_thread_settings(db.conn, thread.id, [], "qwen2.5")
        await update_thread_settings(db.conn, thread.id, ["tool_b"], None)

        reloaded = await get_thread(db.conn, thread.id)
        assert reloaded is not None
        assert reloaded.chat_model is None
        assert reloaded.active_tools == ["tool_b"]
