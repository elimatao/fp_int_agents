"""Advanced RAG agent: query rewriter, hybrid search, reranker, generate, relevance judge."""

import pytest
from fp_int_agents.agents import rag_agent
from fp_int_agents.config import Project, QueryConfig
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage


def _project(**kwargs) -> Project:
    return Project(
        name="R",
        agent="rag",
        chat_model="llama3.2",
        init_config={"embedding_model": "multilingual-e5-small"},
        **kwargs,
    )


def _qc(project: Project) -> QueryConfig:
    return QueryConfig(thread_id="t1", project_id=project.id, chat_model="llama3.2")


def _fake_docs() -> list[Document]:
    return [Document(page_content="FACT-A"), Document(page_content="FACT-B")]


# ---------------------------------------------------------------------------
# Query rewriter uses the LLM to expand the raw question
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_rewriter_expands_question(monkeypatch) -> None:
    """Rewriter node calls the LLM with the original question."""
    rewrite_inputs: list[str] = []

    class FakeModel:
        _call = 0

        async def ainvoke(self, messages):
            FakeModel._call += 1
            if FakeModel._call == 1:
                rewrite_inputs.append(str(messages[-1].content))
                return AIMessage(content="expanded question")
            # judge — always pass
            return AIMessage(content="relevant")

        def bind_tools(self, _tools):
            return self

    monkeypatch.setattr(rag_agent, "get_chat_model", lambda **_: FakeModel())
    monkeypatch.setattr(rag_agent, "get_embedding_model", lambda **_: object())
    monkeypatch.setattr(rag_agent.vectorstore, "search", lambda *a, **kw: _fake_docs())

    project = _project()
    agent = rag_agent.build_agent(project)
    await agent.ainvoke(
        {"messages": [HumanMessage(content="what is X?")]},
        _qc(project).to_runnable_config(),
    )

    assert rewrite_inputs, "rewriter was never called"
    assert "what is X?" in rewrite_inputs[0]


# ---------------------------------------------------------------------------
# Hybrid search is called with the rewritten query
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hybrid_search_called_with_rewritten_query(monkeypatch) -> None:
    searched: list[str] = []

    class FakeModel:
        async def ainvoke(self, messages):
            return AIMessage(content="rewritten query")

        def bind_tools(self, _tools):
            return self

    monkeypatch.setattr(rag_agent, "get_chat_model", lambda **_: FakeModel())
    monkeypatch.setattr(rag_agent, "get_embedding_model", lambda **_: object())

    def _search(_emb, _pid, query, top_k=5, client=None):
        searched.append(query)
        return _fake_docs()

    monkeypatch.setattr(rag_agent.vectorstore, "search", _search)

    project = _project()
    agent = rag_agent.build_agent(project)
    await agent.ainvoke(
        {"messages": [HumanMessage(content="original question")]},
        _qc(project).to_runnable_config(),
    )

    assert searched, "search was never called"
    assert searched[0] == "rewritten query"


# ---------------------------------------------------------------------------
# Reranker trims docs to TOP_K_RERANK
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reranker_reduces_docs(monkeypatch) -> None:
    class FakeModel:
        async def ainvoke(self, messages):
            return AIMessage(content="relevant")

        def bind_tools(self, _tools):
            return self

    monkeypatch.setattr(rag_agent, "get_chat_model", lambda **_: FakeModel())
    monkeypatch.setattr(rag_agent, "get_embedding_model", lambda **_: object())
    monkeypatch.setattr(
        rag_agent.vectorstore,
        "search",
        lambda *a, **kw: [Document(page_content=f"doc-{i}") for i in range(20)],
    )

    project = _project()
    agent = rag_agent.build_agent(project)
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        _qc(project).to_runnable_config(),
    )

    assert len(result["documents"]) <= rag_agent.TOP_K_RERANK


# ---------------------------------------------------------------------------
# Relevance judge passes a relevant answer straight to END
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_relevance_judge_passes_relevant_answer(monkeypatch) -> None:
    call_log: list[str] = []

    models = iter(["rewriter", "generator", "judge"])

    class SequencedModel:
        def __init__(self, role: str):
            self._role = role

        async def ainvoke(self, messages):
            call_log.append(self._role)
            if self._role == "judge":
                return AIMessage(content="relevant")
            return AIMessage(content=f"{self._role}-output")

        def bind_tools(self, _tools):
            return self

    monkeypatch.setattr(
        rag_agent, "get_chat_model", lambda **_: SequencedModel(next(models))
    )
    monkeypatch.setattr(rag_agent, "get_embedding_model", lambda **_: object())
    monkeypatch.setattr(rag_agent.vectorstore, "search", lambda *a, **kw: _fake_docs())

    project = _project()
    agent = rag_agent.build_agent(project)
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="q")]},
        _qc(project).to_runnable_config(),
    )

    assert "judge" in call_log
    assert isinstance(result["messages"][-1], AIMessage)
    assert result["rewrite_count"] == 0


# ---------------------------------------------------------------------------
# rewrite_count is capped at MAX_REWRITES even when judge always fails
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rewrite_count_capped_at_max(monkeypatch) -> None:
    """Graph must terminate after MAX_REWRITES even if judge always says not_relevant."""
    call_counts: dict[str, int] = {"rewrite": 0, "generate": 0, "judge": 0}

    # We distinguish roles by call order within the cycle:
    # each pipeline pass = rewriter -> (search/rerank are pure) -> generator -> judge
    pass_stage: list[str] = []

    class CyclingModel:
        """Cycles through rewriter -> generator -> judge on each ainvoke call."""

        _n = 0

        async def ainvoke(self, messages):
            n = CyclingModel._n % 3
            CyclingModel._n += 1
            if n == 0:
                call_counts["rewrite"] += 1
                pass_stage.append("rewrite")
                return AIMessage(content="rewritten")
            if n == 1:
                call_counts["generate"] += 1
                pass_stage.append("generate")
                return AIMessage(content="answer")
            call_counts["judge"] += 1
            pass_stage.append("judge")
            return AIMessage(content="not_relevant")  # always fail

        def bind_tools(self, _tools):
            return self

    monkeypatch.setattr(rag_agent, "get_chat_model", lambda **_: CyclingModel())
    monkeypatch.setattr(rag_agent, "get_embedding_model", lambda **_: object())
    monkeypatch.setattr(rag_agent.vectorstore, "search", lambda *a, **kw: _fake_docs())

    project = _project()
    agent = rag_agent.build_agent(project)
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="test question")]},
        _qc(project).to_runnable_config(),
    )

    assert isinstance(result["messages"][-1], AIMessage)
    assert result["rewrite_count"] <= rag_agent.MAX_REWRITES


# ---------------------------------------------------------------------------
# Graceful degradation when vector store is unavailable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rag_degrades_without_context(monkeypatch) -> None:
    class FakeModel:
        async def ainvoke(self, messages):
            return AIMessage(content="ok")

        def bind_tools(self, _tools):
            return self

    def _boom(*_a, **_kw):
        raise RuntimeError("no vector store yet")

    monkeypatch.setattr(rag_agent, "get_chat_model", lambda **_: FakeModel())
    monkeypatch.setattr(rag_agent, "get_embedding_model", lambda **_: object())
    monkeypatch.setattr(rag_agent.vectorstore, "search", _boom)

    project = _project(system_prompt="BASE")
    agent = rag_agent.build_agent(project)
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        _qc(project).to_runnable_config(),
    )

    assert isinstance(result["messages"][-1], AIMessage)
