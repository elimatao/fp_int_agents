"""Project.ingestor and init_config persist and reload."""

import pytest
from fp_int_agents.storage.db import create_project, get_project, open_db


@pytest.mark.asyncio
async def test_ingestor_and_init_config_persist(tmp_path) -> None:
    db_path = str(tmp_path / "app.db")
    async with open_db(db_path) as db:
        project = await create_project(
            db.conn,
            "R",
            "llama3.2",
            agent="BundesRAG",
            ingestor="BundesRAG",
            init_config={"embedding_model": "text-embedding-3-small"},
        )

        reloaded = await get_project(db.conn, project.id)
        assert reloaded is not None
        assert reloaded.agent == "BundesRAG"
        assert reloaded.ingestor == "BundesRAG"
        assert reloaded.init_config == {"embedding_model": "text-embedding-3-small"}


@pytest.mark.asyncio
async def test_ingestor_none_by_default(tmp_path) -> None:
    db_path = str(tmp_path / "app.db")
    async with open_db(db_path) as db:
        project = await create_project(db.conn, "P", "llama3.2")

        reloaded = await get_project(db.conn, project.id)
        assert reloaded is not None
        assert reloaded.ingestor is None
