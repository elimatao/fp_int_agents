from collections.abc import AsyncIterator

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver

from fp_int_agents.agents.registry import AGENTS
from fp_int_agents.config import Project, QueryConfig


async def call_agent(
    project: Project,
    query_config: QueryConfig,
    message: HumanMessage,
    checkpointer: BaseCheckpointSaver,
) -> AsyncIterator[str]:
    agent = AGENTS[project.agent](project, checkpointer)
    runnable_config = query_config.to_runnable_config()
    async for event in agent.astream_events(
        {"messages": [message]}, runnable_config, version="v2"
    ):
        if event["event"] == "on_chat_model_stream" and event["data"]["chunk"].content:
            yield event["data"]["chunk"].content
