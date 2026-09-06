"""Regression test: tool selection persists across chat switches.

Bug: selecting tools in chat A, switching to chat B, then back to A rendered
0 tools because the switch path read a stale in-memory Thread object cached in
the sidebar at startup rather than the freshly-saved DB state.
"""

import pytest

from fp_int_agents.storage.db import (
    create_project,
    create_thread,
    get_thread,
    open_db,
    update_thread_tools,
)


@pytest.mark.asyncio
async def test_saved_tools_are_reloaded_by_thread_id(tmp_path) -> None:
    db_path = str(tmp_path / "app.db")
    async with open_db(db_path) as db:
        project = await create_project(db.conn, "P", "llama3.2")
        thread_a = await create_thread(db.conn, project.id, "A")
        thread_b = await create_thread(db.conn, project.id, "B")

        # Select tools in chat A.
        await update_thread_tools(db.conn, thread_a.id, ["ping", "shell"])

        # Switching away to B and back to A must reload A's current tools from
        # the DB by id, not trust a stale cached Thread object.
        reloaded_a = await get_thread(db.conn, thread_a.id)
        assert reloaded_a is not None
        assert reloaded_a.active_tools == ["ping", "shell"]

        reloaded_b = await get_thread(db.conn, thread_b.id)
        assert reloaded_b is not None
        assert reloaded_b.active_tools == []
