"""Integration test: simple agent calls a tool and returns its output."""

import pytest
from conftest import get_model_id
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from fp_int_agents.agents.registry import CONVERSATIONAL_AGENTS
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
async def test_agent_calls_tool_and_returns_response(project_config):
    app_cfg = load_config()
    model_id = await get_model_id(app_cfg)

    agent = CONVERSATIONAL_AGENTS["simple"](project_config, None, app_cfg.llm)

    stream = await agent.astream_events(
        {
            "messages": [
                HumanMessage(
                    content="Please ping. Call the ping tool and reply with only its one-word response, nothing else."
                )
            ]
        },
        config={
            "configurable": QueryConfig(
                thread_id="test-thread",
                project_id="test",
                chat_model=model_id,
                active_tools=["ping"],
            ).model_dump()
        },
        version="v3",
    )
    async for message in stream.messages:
        async for delta in message.text:
            print(delta, end="_", flush=True)

    final_state = await stream.output()
    last_message = final_state["messages"][-1]

    print(final_state)
    print(last_message)
    print(last_message.content)
    assert last_message.content[0]["text"].strip().lower() == TOOL_RESPONSE
