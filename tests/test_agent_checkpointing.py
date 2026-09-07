"""Integration tests for AsyncSqliteSaver conversation persistence."""

import tempfile
from pathlib import Path

import pytest
from conftest import get_model_id
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from fp_int_agents.agents.conv_agent_simple import build_agent
from fp_int_agents.config import QueryConfig, load_config


def _thread_cfg(thread_id: str, model_id: str) -> dict:
    return {
        "configurable": QueryConfig(
            thread_id=thread_id,
            project_id="ckpt-test",
            chat_model=model_id,
            active_tools=[],
        ).model_dump()
    }


@pytest.mark.asyncio
async def test_checkpoint_persists_history_across_turns(project_config) -> None:
    app_cfg = await load_config()
    model_id = await get_model_id(app_cfg)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        thread_cfg = _thread_cfg("ckpt-thread-persist", model_id)

        async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
            agent = build_agent(project_config, checkpointer=checkpointer, llm_config=app_cfg.llm)
            await agent.ainvoke(
                {"messages": [HumanMessage(content="My name is Alice.")]},
                config=thread_cfg,
            )

        async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
            agent = build_agent(project_config, checkpointer=checkpointer, llm_config=app_cfg.llm)
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content="What is my name?")]},
                config=thread_cfg,
            )

        assert "alice" in result["messages"][-1].content.lower()
    finally:
        Path(db_path).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_checkpoint_isolates_different_threads(project_config) -> None:
    app_cfg = await load_config()
    model_id = await get_model_id(app_cfg)

    async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
        agent = build_agent(project_config, checkpointer=checkpointer, llm_config=app_cfg.llm)

        await agent.ainvoke(
            {"messages": [HumanMessage(content="My name is Bob.")]},
            config=_thread_cfg("ckpt-thread-a", model_id),
        )
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content="What is my name?")]},
            config=_thread_cfg("ckpt-thread-b", model_id),
        )

    assert "bob" not in result["messages"][-1].content.lower()
