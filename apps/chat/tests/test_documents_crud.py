"""documents table CRUD roundtrip."""

import pytest
from fp_int_agents.storage.db import (
    create_document,
    create_project,
    list_documents,
    open_db,
)


@pytest.mark.asyncio
async def test_document_crud(tmp_path) -> None:
    db_path = str(tmp_path / "app.db")
    async with open_db(db_path) as db:
        project = await create_project(db.conn, "P", "llama3.2")
        doc = await create_document(
            db.conn, project.id, original="hello world", summary="hi"
        )

        rows = await list_documents(db.conn, project.id)
        assert len(rows) == 1
        assert rows[0].id == doc.id
        assert rows[0].summary == "hi"
        assert rows[0].original == "hello world"


@pytest.mark.asyncio
async def test_document_summary_optional(tmp_path) -> None:
    db_path = str(tmp_path / "app.db")
    async with open_db(db_path) as db:
        project = await create_project(db.conn, "P", "llama3.2")
        await create_document(db.conn, project.id, original="body only")

        rows = await list_documents(db.conn, project.id)
        assert len(rows) == 1
        assert rows[0].summary is None
        assert rows[0].original == "body only"
