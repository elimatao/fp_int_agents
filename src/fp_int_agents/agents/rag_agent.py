import asyncio
import operator
from typing import Annotated, Literal

import aiosqlite
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from typing_extensions import TypedDict

from fp_int_agents.config import LlmConfig, Project, QueryConfig
from fp_int_agents.llm.client import get_chat_model, get_embedding_model
from fp_int_agents.storage import vectorstore
from fp_int_agents.tools.registry import dispatch, get_tools

TOP_K = 5


class RagAgentInitConfig(BaseModel):
    """Immutable config for the RAG agent (stored in Project.init_config)."""

    embedding_model: str = "nomic-embed-text"


class RagAgentConfig(BaseModel):
    """Mutable config for the RAG agent (stored in Project.config)."""


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


_DEFAULT_LLM_CONFIG = LlmConfig()


def build_agent(
    project: Project,
    checkpointer: BaseCheckpointSaver | None = None,
    llm_config: LlmConfig = _DEFAULT_LLM_CONFIG,
    conn: aiosqlite.Connection | None = None,
) -> CompiledStateGraph:
    init = RagAgentInitConfig.model_validate(project.init_config)
    RagAgentConfig.model_validate(project.config)

    def _embeddings() -> Embeddings:
        return get_embedding_model(
            base_url=llm_config.base_url,
            model=init.embedding_model,
            api_key=llm_config.api_key,
        )

    def _build_model(config: RunnableConfig) -> Runnable[LanguageModelInput, AIMessage]:
        qc = QueryConfig.from_runnable_config(config)
        tools = get_tools(qc.active_tools)
        model = get_chat_model(
            base_url=llm_config.base_url, model=qc.chat_model, api_key=llm_config.api_key
        )
        if tools:
            model = model.bind_tools(tools)
        return model

    async def _retrieve(project_id: str, query: str) -> str:
        try:
            docs = await asyncio.to_thread(
                vectorstore.search, _embeddings(), project_id, query, TOP_K
            )
        except Exception:  # noqa: BLE001 - best-effort: nothing ingested / endpoint down
            return ""
        return "\n\n".join(f"- {d.page_content}" for d in docs)

    async def _build_system_prompt(state: AgentState, config: RunnableConfig) -> str:
        base = project.system_prompt or "You are a helpful assistant."
        qc = QueryConfig.from_runnable_config(config)
        last_human = next(
            (m.content for m in reversed(state["messages"])
             if isinstance(m, HumanMessage)),
            "",
        )
        context = await _retrieve(qc.project_id, str(last_human)) if last_human else ""
        if not context:
            return base
        return (
            f"{base}\n\n## Retrieved context\n"
            "Use the following retrieved passages to answer if relevant:\n"
            f"{context}"
        )

    async def _llm_call(state: AgentState, config: RunnableConfig) -> AgentState:
        return {
            "messages": [
                _build_model(config).invoke(
                    [SystemMessage(content=await _build_system_prompt(state, config))]
                    + state["messages"]
                )
            ]
        }

    async def _tool_node(state: AgentState, config: RunnableConfig) -> AgentState:
        tool_calls = state["messages"][-1].tool_calls
        return {"messages": await dispatch(tool_calls)}

    def _should_continue(state: AgentState) -> Literal["tool_node", "__end__"]:
        return "tool_node" if state["messages"][-1].tool_calls else END

    builder = StateGraph(AgentState)
    builder.add_node("llm_call", _llm_call)
    builder.add_node("tool_node", _tool_node)
    builder.add_edge(START, "llm_call")
    builder.add_conditional_edges("llm_call", _should_continue, ["tool_node", END])
    builder.add_edge("tool_node", "llm_call")
    return builder.compile(checkpointer=checkpointer)
