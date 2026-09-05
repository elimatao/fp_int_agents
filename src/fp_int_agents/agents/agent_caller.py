from collections.abc import AsyncIterator

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import BaseModel

from fp_int_agents.agents.registry import AGENTS
from fp_int_agents.config import Project, QueryConfig

RESULT_PREVIEW_CHARS = 200


class TextToken(BaseModel):
    """A chunk of streamed assistant text."""

    text: str


class ToolCallStarted(BaseModel):
    """A tool has begun executing."""

    run_id: str
    name: str
    args: dict


class ToolCallFinished(BaseModel):
    """A tool has finished; result is truncated to a preview."""

    run_id: str
    name: str
    result: str


StreamEvent = TextToken | ToolCallStarted | ToolCallFinished


def preview_result(value: object) -> str:
    """Truncate a tool result to a short preview for display."""
    text = value if isinstance(value, str) else str(value)
    if len(text) > RESULT_PREVIEW_CHARS:
        return text[:RESULT_PREVIEW_CHARS] + "…"
    return text


async def call_agent(
    project: Project,
    query_config: QueryConfig,
    message: HumanMessage,
    checkpointer: BaseCheckpointSaver,
) -> AsyncIterator[StreamEvent]:
    agent = AGENTS[project.agent](project, checkpointer)
    runnable_config = query_config.to_runnable_config()
    async for event in agent.astream_events(
        {"messages": [message]}, runnable_config, version="v2"
    ):
        kind = event["event"]
        if kind == "on_chat_model_stream" and event["data"]["chunk"].content:
            yield TextToken(text=event["data"]["chunk"].content)
        elif kind == "on_tool_start":
            yield ToolCallStarted(
                run_id=event["run_id"],
                name=event["name"],
                args=event["data"].get("input", {}) or {},
            )
        elif kind == "on_tool_end":
            output = event["data"].get("output")
            content = getattr(output, "content", output)
            yield ToolCallFinished(
                run_id=event["run_id"],
                name=event["name"],
                result=preview_result(content),
            )
