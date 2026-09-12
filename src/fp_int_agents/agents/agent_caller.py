from collections.abc import AsyncIterator

import aiosqlite
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import BaseModel

from fp_int_agents.agents.registry import CONVERSATIONAL_AGENTS, MEMORY_AGENTS
from fp_int_agents.config import LlmConfig, Project, QueryConfig

RESULT_PREVIEW_CHARS = 200

_DEFAULT_LLM_CONFIG = LlmConfig()


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


class RagStepStarted(BaseModel):
    """A RAG pipeline step (rewriter / judge) has begun."""

    node: str


class RagStepFinished(BaseModel):
    """A RAG pipeline step has produced its output."""

    node: str
    result: str


class RerankerFinished(BaseModel):
    """The reranker node has selected its top documents."""

    docs: list[str]


StreamEvent = TextToken | ToolCallStarted | ToolCallFinished | RagStepStarted | RagStepFinished | RerankerFinished

# LangGraph node names whose LLM output is internal to the RAG pipeline.
_RAG_INTERNAL_NODES = {"query_rewriter", "relevance_judge"}

_RAG_NODE_LABELS = {
    "query_rewriter": "Query rewrite",
    "relevance_judge": "Relevance check",
    "reranker": "Reranked docs",
}


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
    llm_config: LlmConfig = _DEFAULT_LLM_CONFIG,
    conn: aiosqlite.Connection | None = None,
) -> AsyncIterator[StreamEvent]:
    agent = CONVERSATIONAL_AGENTS[project.agent](project, checkpointer, llm_config, conn)
    runnable_config = query_config.to_runnable_config()
    rag_node_buffers: dict[str, str] = {}  # run_id -> accumulated text for internal nodes

    async for event in agent.astream_events(
        {"messages": [message]}, runnable_config, version="v2"
    ):
        kind = event["event"]
        node = event.get("metadata", {}).get("langgraph_node", "")

        if kind == "on_chat_model_start" and node in _RAG_INTERNAL_NODES:
            run_id = event["run_id"]
            rag_node_buffers[run_id] = ""
            yield RagStepStarted(node=node)

        elif kind == "on_chat_model_stream":
            chunk_content = event["data"]["chunk"].content
            if not chunk_content:
                continue
            run_id = event["run_id"]
            if run_id in rag_node_buffers:
                rag_node_buffers[run_id] += chunk_content
            else:
                yield TextToken(text=chunk_content)

        elif kind == "on_chat_model_end" and event["run_id"] in rag_node_buffers:
            run_id = event["run_id"]
            yield RagStepFinished(node=node, result=rag_node_buffers.pop(run_id))

        elif kind == "on_chain_end" and node == "reranker":
            docs = (event.get("data", {}).get("output") or {}).get("documents") or []
            seen: set[str] = set()
            titles: list[str] = []
            for d in docs:
                t = d.metadata.get("title") or d.metadata.get("doc_id", "unknown")
                if t not in seen:
                    seen.add(t)
                    titles.append(t)
            yield RerankerFinished(docs=titles)

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


async def summarize_thread(
    project: Project,
    query_config: QueryConfig,
    messages: list[AnyMessage],
    llm_config: LlmConfig = _DEFAULT_LLM_CONFIG,
) -> str:
    """Summarize a thread's messages and return the summary string."""
    agent = MEMORY_AGENTS[project.mem_agent](project, llm_config)
    result = await agent.ainvoke(
        {"messages": messages, "summary": None},
        query_config.to_runnable_config(),
    )
    return result["summary"]
