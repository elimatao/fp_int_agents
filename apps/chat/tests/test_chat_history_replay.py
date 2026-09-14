"""Integration test: Chat replays past tool calls from checkpointer history."""

import pytest
from conftest import get_model_id
from fp_int_agents.agents.conv_agent_simple import build_agent
from fp_int_agents.config import Project, QueryConfig, load_config
from fp_int_agents.storage.db import open_db
from fp_int_agents.tools.registry import register
from fp_int_agents.widgets.chat import Chat
from fp_int_agents.widgets.tool_call_bubble import ToolCallBubble
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from textual.app import App, ComposeResult

TOOL_RESPONSE = "poooong"


@tool
def ping() -> str:
    """Call this tool when asked to ping."""
    return TOOL_RESPONSE


@pytest.fixture(autouse=True)
def register_ping_tool():
    register(ping)


@pytest.mark.asyncio
async def test_history_replay_renders_tool_call_bubble(tmp_path):
    app_cfg = await load_config()
    model_id = await get_model_id(app_cfg)
    project = Project(name="test", agent="simple", chat_model=model_id)
    query_config = QueryConfig(
        thread_id="replay-thread",
        project_id=project.id,
        chat_model=model_id,
        active_tools=["ping"],
    )

    db_path = str(tmp_path / "app.db")
    async with open_db(db_path) as db:
        agent = build_agent(
            project, checkpointer=db.checkpointer, llm_config=app_cfg.llm
        )
        await agent.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content="Please ping. Call the ping tool and reply with only its one-word response."
                    )
                ]
            },
            config=query_config.to_runnable_config(),
        )

        class _App(App):
            def compose(self) -> ComposeResult:
                yield Chat(query_config)

        app = _App()
        app.db = db  # Chat reads self.app.db
        async with app.run_test() as pilot:
            await pilot.pause()
            bubbles = app.query(ToolCallBubble)
            assert len(bubbles) >= 1
            assert any(b._name == "ping" for b in bubbles)
