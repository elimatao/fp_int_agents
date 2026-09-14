import httpx
import pytest
from fp_int_agents.agents import rag_agent
from fp_int_agents.config import Project, QueryConfig
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage


def _project(**kwargs) -> Project:
    return Project(
        name="RerankerTest",
        agent="rag",
        chat_model="default_model",
        init_config={
            "embedding_model": "multilingual-e5-small",
            "reranker_url": "http://127.0.0.1:8001/v1/rerank",
        },
        **kwargs,
    )


def _qc(project: Project) -> QueryConfig:
    return QueryConfig(
        thread_id="t1", project_id=project.id, chat_model="default_model"
    )


class FakeChatModel:
    async def ainvoke(self, messages):
        last_content = str(messages[-1].content)
        if "target" in last_content:
            return AIMessage(content="target")
        return AIMessage(content="relevant")

    def bind_tools(self, _tools):
        return self


@pytest.mark.asyncio
async def test_reranker_calls_http_service_successfully(monkeypatch) -> None:
    """Reranker node calls the HTTP service and reorders docs by returned score."""
    docs = [
        Document(page_content="doc 0 - unrelated"),
        Document(page_content="doc 1 - highly relevant answer"),
    ]

    class MockResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "results": [
                    {"index": 1, "relevance_score": 0.98},
                    {"index": 0, "relevance_score": 0.05},
                ]
            }

    called_payloads: list[dict] = []

    async def mock_post(_self, url, json=None, **kwargs):
        called_payloads.append(json)
        return MockResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)
    monkeypatch.setattr(rag_agent, "get_chat_model", lambda **_: FakeChatModel())
    monkeypatch.setattr(rag_agent, "get_embedding_model", lambda **_: object())
    monkeypatch.setattr(rag_agent.vectorstore, "search", lambda *a, **kw: docs)

    project = _project()
    agent = rag_agent.build_agent(project)
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="relevant question")]},
        _qc(project).to_runnable_config(),
    )

    assert len(called_payloads) == 1
    assert called_payloads[0]["query"] != ""
    assert len(called_payloads[0]["documents"]) == 2
    # doc 1 should have been ranked first because of score 0.98
    assert result["documents"][0].page_content == "doc 1 - highly relevant answer"


@pytest.mark.asyncio
async def test_reranker_falls_back_to_lexical_when_service_fails(monkeypatch) -> None:
    """Reranker falls back gracefully to lexical matching if HTTP call fails."""
    docs = [
        Document(page_content="something unrelated"),
        Document(page_content="contains target word"),
    ]

    async def mock_post_fail(*args, **kwargs):
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post_fail)
    monkeypatch.setattr(rag_agent, "get_chat_model", lambda **_: FakeChatModel())
    monkeypatch.setattr(rag_agent, "get_embedding_model", lambda **_: object())
    monkeypatch.setattr(rag_agent.vectorstore, "search", lambda *a, **kw: docs)

    project = _project()
    agent = rag_agent.build_agent(project)
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="target")]},
        _qc(project).to_runnable_config(),
    )

    assert len(result["documents"]) == 2
    # Lexical fallback will prioritize "contains target word"
    assert result["documents"][0].page_content == "contains target word"
