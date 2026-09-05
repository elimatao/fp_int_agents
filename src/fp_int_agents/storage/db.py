"""SQLite storage layer: app tables (projects, threads) + checkpointer lifecycle."""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from fp_int_agents.config import Project, Thread

_DDL = """
CREATE TABLE IF NOT EXISTS projects (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    agent         TEXT NOT NULL DEFAULT 'simple',
    chat_model    TEXT NOT NULL,
    system_prompt TEXT,
    init_config   TEXT NOT NULL DEFAULT '{}',
    config        TEXT NOT NULL DEFAULT '{}',
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS threads (
    id           TEXT PRIMARY KEY,
    project_id   TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    title        TEXT,
    active_tools TEXT NOT NULL DEFAULT '[]',
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


async def _setup_tables(conn: aiosqlite.Connection) -> None:
    await conn.executescript(_DDL)
    await conn.commit()


def _row_to_project(row: tuple, cols: list[str]) -> Project:
    d = dict(zip(cols, row))
    d["init_config"] = json.loads(d["init_config"])
    d["config"] = json.loads(d["config"])
    return Project.model_validate(d)


def _row_to_thread(row: tuple, cols: list[str]) -> Thread:
    d = dict(zip(cols, row))
    d["active_tools"] = json.loads(d["active_tools"])
    return Thread.model_validate(d)


def _cols(cursor: aiosqlite.Cursor) -> list[str]:
    return [d[0] for d in cursor.description]


@dataclass
class DB:
    conn: aiosqlite.Connection
    checkpointer: AsyncSqliteSaver


@asynccontextmanager
async def open_db(path: str) -> AsyncIterator[DB]:
    async with aiosqlite.connect(path) as conn:
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA foreign_keys=ON")
        await _setup_tables(conn)
        async with AsyncSqliteSaver.from_conn_string(path) as checkpointer:
            await checkpointer.setup()
            yield DB(conn=conn, checkpointer=checkpointer)


# --- Projects ---


async def create_project(
    conn: aiosqlite.Connection,
    name: str,
    chat_model: str,
    *,
    agent: str = "simple",
    system_prompt: str | None = None,
    init_config: dict | None = None,
    config: dict | None = None,
) -> Project:
    project = Project(
        name=name,
        chat_model=chat_model,
        agent=agent,
        system_prompt=system_prompt,
        init_config=init_config or {},
        config=config or {},
    )
    await conn.execute(
        "INSERT INTO projects (id, name, agent, chat_model, system_prompt, init_config, config) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            project.id,
            project.name,
            project.agent,
            project.chat_model,
            project.system_prompt,
            json.dumps(project.init_config),
            json.dumps(project.config),
        ),
    )
    await conn.commit()
    return project


async def get_project(conn: aiosqlite.Connection, project_id: str) -> Project | None:
    async with conn.execute(
        "SELECT id, name, agent, chat_model, system_prompt, init_config, config, created_at FROM projects WHERE id = ?",
        (project_id,),
    ) as cursor:
        row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_project(row, _cols(cursor))


async def list_projects(conn: aiosqlite.Connection) -> list[Project]:
    async with conn.execute(
        "SELECT id, name, agent, chat_model, system_prompt, init_config, config, created_at FROM projects ORDER BY created_at DESC"
    ) as cursor:
        cols = _cols(cursor)
        return [_row_to_project(row, cols) async for row in cursor]


async def update_project(
    conn: aiosqlite.Connection,
    project_id: str,
    *,
    name: str | None = None,
    system_prompt: str | None = None,
    config: dict | None = None,
) -> None:
    """Update mutable project fields. Immutable fields (id, agent, chat_model) are not accepted."""
    fields: dict = {}
    if name is not None:
        fields["name"] = name
    if system_prompt is not None:
        fields["system_prompt"] = system_prompt
    if config is not None:
        fields["config"] = json.dumps(config)
    if not fields:
        return
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    await conn.execute(
        f"UPDATE projects SET {set_clause} WHERE id = ?",
        (*fields.values(), project_id),
    )
    await conn.commit()


# --- Threads ---


async def create_thread(
    conn: aiosqlite.Connection, project_id: str, title: str | None = None
) -> Thread:
    thread = Thread(project_id=project_id, title=title)
    await conn.execute(
        "INSERT INTO threads (id, project_id, title, active_tools) VALUES (?, ?, ?, ?)",
        (thread.id, thread.project_id, thread.title, json.dumps(thread.active_tools)),
    )
    await conn.commit()
    return thread


async def list_threads(conn: aiosqlite.Connection, project_id: str) -> list[Thread]:
    async with conn.execute(
        "SELECT id, project_id, title, active_tools, created_at FROM threads WHERE project_id = ? ORDER BY created_at DESC",
        (project_id,),
    ) as cursor:
        cols = _cols(cursor)
        return [_row_to_thread(row, cols) async for row in cursor]


async def update_thread_tools(
    conn: aiosqlite.Connection, thread_id: str, active_tools: list[str]
) -> None:
    await conn.execute(
        "UPDATE threads SET active_tools = ? WHERE id = ?",
        (json.dumps(active_tools), thread_id),
    )
    await conn.commit()


async def delete_thread(
    conn: aiosqlite.Connection, checkpointer: AsyncSqliteSaver, thread_id: str
) -> None:
    await conn.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
    await conn.commit()
    await checkpointer.adelete_thread(thread_id)


async def delete_project(
    conn: aiosqlite.Connection,
    checkpointer: AsyncSqliteSaver,
    project_id: str,
    thread_ids: list[str],
) -> None:
    await conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    await conn.commit()
    for tid in thread_ids:
        await checkpointer.adelete_thread(tid)
