"""Advanced RAG agent graph.

Pipeline:
  query_rewriter -> hybrid_search -> reranker -> generate -> relevance_judge
                        ^                                          |
                        |__________ (rewrite_count < 2) __________|
"""

import asyncio
import operator
from typing import Annotated, Literal

import aiosqlite
from langchain_core.documents import Document
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

TOP_K_RETRIEVE = 20
TOP_K_RERANK = 5
MAX_REWRITES = 2

_REWRITER_PROMPT = (
    "You are a query rewriter for a RAG system. "
    "Rewrite the user's question to be clearer and more specific for document retrieval. "
    "Output ONLY the improved query, nothing else."
)

_JUDGE_PROMPT = (
    "You are a relevance judge. Given the question and the answer, decide if the answer "
    "adequately addresses the question using the retrieved context. "
    "Reply with exactly one word: 'relevant' or 'not_relevant'."
)


class RagAgentInitConfig(BaseModel):
    embedding_model: str = "nomic-embed-text"


class RagAgentConfig(BaseModel):
    pass


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    query: str
    documents: list[Document]
    generation: str
    rewrite_count: int
    # routing signal written by relevance_judge; read by _route_judge_verdict
    _judge_verdict: str


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

    def _chat(config: RunnableConfig) -> Runnable[LanguageModelInput, AIMessage]:
        qc = QueryConfig.from_runnable_config(config)
        return get_chat_model(
            base_url=llm_config.base_url, model=qc.chat_model, api_key=llm_config.api_key
        )

    def _chat_with_tools(config: RunnableConfig) -> Runnable[LanguageModelInput, AIMessage]:
        qc = QueryConfig.from_runnable_config(config)
        tools = get_tools(qc.active_tools)
        model = get_chat_model(
            base_url=llm_config.base_url, model=qc.chat_model, api_key=llm_config.api_key
        )
        return model.bind_tools(tools) if tools else model

    async def query_rewriter(state: AgentState, config: RunnableConfig) -> dict:
        last_human = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            "",
        )
        prior_query = state.get("query") or str(last_human)
        response = await _chat(config).ainvoke(
            [SystemMessage(content=_REWRITER_PROMPT), HumanMessage(content=prior_query)]
        )
        return {"query": str(response.content).strip()}

    async def hybrid_search(state: AgentState, config: RunnableConfig) -> dict:
        qc = QueryConfig.from_runnable_config(config)
        try:
            docs = await asyncio.to_thread(
                vectorstore.search,
                _embeddings(),
                qc.project_id,
                state["query"],
                TOP_K_RETRIEVE,
            )
        except Exception:  # noqa: BLE001
            docs = []
        return {"documents": docs}

    async def reranker(state: AgentState, config: RunnableConfig) -> dict:
        docs = state.get("documents") or []
        query_terms = set(state["query"].lower().split())

        def _score(doc: Document) -> int:
            return sum(1 for t in query_terms if t in doc.page_content.lower())

        return {"documents": sorted(docs, key=_score, reverse=True)[:TOP_K_RERANK]}

    async def generate(state: AgentState, config: RunnableConfig) -> dict:
        docs = state.get("documents") or []
        context = "\n\n".join(f"- {d.page_content}" for d in docs)
        base = project.system_prompt or "You are a helpful assistant."
        system = (
            f"{base}\n\n## Retrieved context\n"
            "Use the following retrieved passages to answer if relevant:\n"
            f"{context}"
            if context
            else base
        )
        response = await _chat_with_tools(config).ainvoke(
            [SystemMessage(content=system)] + state["messages"]
        )
        return {"messages": [response], "generation": str(response.content)}

    async def tool_node(state: AgentState, config: RunnableConfig) -> dict:
        return {"messages": await dispatch(state["messages"][-1].tool_calls)}

    async def relevance_judge(state: AgentState, config: RunnableConfig) -> dict:
        last_human = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            "",
        )
        response = await _chat(config).ainvoke(
            [
                SystemMessage(content=_JUDGE_PROMPT),
                HumanMessage(content=f"Question: {last_human}\n\nAnswer: {state['generation']}"),
            ]
        )
        verdict = str(response.content).strip().lower()
        is_relevant = "relevant" in verdict and "not_relevant" not in verdict and "not relevant" not in verdict
        current_count = state.get("rewrite_count", 0)

        if is_relevant or current_count >= MAX_REWRITES:
            return {"rewrite_count": current_count, "_judge_verdict": END}
        return {"rewrite_count": current_count + 1, "_judge_verdict": "query_rewriter"}

    def _after_generate(state: AgentState) -> Literal["tool_node", "relevance_judge"]:
        return "tool_node" if getattr(state["messages"][-1], "tool_calls", None) else "relevance_judge"

    def _after_judge(state: AgentState) -> Literal["query_rewriter", "__end__"]:
        return state.get("_judge_verdict", END)  # type: ignore[return-value]

    builder: StateGraph = StateGraph(AgentState)
    builder.add_node("query_rewriter", query_rewriter)
    builder.add_node("hybrid_search", hybrid_search)
    builder.add_node("reranker", reranker)
    builder.add_node("generate", generate)
    builder.add_node("tool_node", tool_node)
    builder.add_node("relevance_judge", relevance_judge)

    builder.add_edge(START, "query_rewriter")
    builder.add_edge("query_rewriter", "hybrid_search")
    builder.add_edge("hybrid_search", "reranker")
    builder.add_edge("reranker", "generate")
    builder.add_conditional_edges("generate", _after_generate, ["tool_node", "relevance_judge"])
    builder.add_edge("tool_node", "generate")
    builder.add_conditional_edges(
        "relevance_judge",
        _after_judge,
        {"query_rewriter": "query_rewriter", END: END},
    )

    return builder.compile(checkpointer=checkpointer)
