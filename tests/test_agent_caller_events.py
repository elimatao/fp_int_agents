"""Integration test: call_agent emits structured tool-call events."""

import pytest
from conftest import get_model_id
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from fp_int_agents.agents.agent_caller import (
    ToolCallFinished,
    ToolCallStarted,
    call_agent,
)
from fp_int_agents.config import QueryConfig, load_config
from fp_int_agents.tools.registry import register

TOOL_RESPONSE = "poooong"


@tool
def ping() -> str:
    """Call this tool when asked to ping."""
    return TOOL_RESPONSE


@pytest.fixture(autouse=True)
def register_ping_tool():
    register(ping)


@pytest.mark.asyncio
async def test_call_agent_emits_tool_events(project_config, memory_checkpointer):
    app_cfg = load_config()
    model_id = await get_model_id(app_cfg)
    query_config = QueryConfig(
        thread_id="test-thread",
        project_id="test",
        chat_model=model_id,
        active_tools=["ping"],
    )

    started: list[ToolCallStarted] = []
    finished: list[ToolCallFinished] = []
    async for event in call_agent(
        project=project_config,
        query_config=query_config,
        message=HumanMessage(
            content="Please ping. Call the ping tool and reply with only its one-word response."
        ),
        checkpointer=memory_checkpointer,
        llm_config=app_cfg.llm,
    ):
        if isinstance(event, ToolCallStarted):
            started.append(event)
        elif isinstance(event, ToolCallFinished):
            finished.append(event)

    assert any(e.name == "ping" for e in started)
    assert any(e.name == "ping" and TOOL_RESPONSE in e.result for e in finished)
