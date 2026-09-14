import pytest
import pytest_asyncio
from fp_int_agents.config import AppConfig, Project
from fp_int_agents.llm.models import list_models
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


@pytest.fixture
def project_config() -> Project:
    return Project(
        name="test",
        agent="simple",
        chat_model="llama3.2",
    )


@pytest_asyncio.fixture
async def memory_checkpointer():
    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        yield checkpointer


async def get_model_id(app_cfg: AppConfig) -> str:
    models = await list_models(
        base_url=app_cfg.llm.base_url, api_key=app_cfg.llm.api_key
    )
    assert models, "No models available"
    return models[0].id
