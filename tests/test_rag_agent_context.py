"""RAG agent injects retrieved passages into the system prompt."""

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from fp_int_agents.agents import rag_agent
from fp_int_agents.config import Project, QueryConfig


@pytest.mark.asyncio
async def test_rag_injects_retrieved_context(monkeypatch) -> None:
    captured: dict[str, str] = {}

    class FakeModel:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            captured["system"] = str(messages[0].content)
            return AIMessage(content="ok")

    monkeypatch.setattr(rag_agent, "get_chat_model", lambda **_: FakeModel())
    monkeypatch.setattr(rag_agent, "get_embedding_model", lambda **_: object())
    monkeypatch.setattr(
        rag_agent.vectorstore,
        "search",
        lambda embeddings, project_id, query, top_k=5: [
            Document(page_content="SECRET-FACT-42", metadata={"project_id": project_id})
        ],
    )

    project = Project(
        name="R",
        agent="rag",
        chat_model="llama3.2",
        init_config={"embedding_model": "nomic-embed-text"},
    )
    agent = rag_agent.build_agent(project)
    qc = QueryConfig(thread_id="t1", project_id=project.id, chat_model="llama3.2")

    await agent.ainvoke(
        {"messages": [HumanMessage(content="tell me the fact")]},
        qc.to_runnable_config(),
    )

    assert "SECRET-FACT-42" in captured["system"]


@pytest.mark.asyncio
async def test_rag_degrades_without_context(monkeypatch) -> None:
    captured: dict[str, str] = {}

    class FakeModel:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            captured["system"] = str(messages[0].content)
            return AIMessage(content="ok")

    monkeypatch.setattr(rag_agent, "get_chat_model", lambda **_: FakeModel())
    monkeypatch.setattr(rag_agent, "get_embedding_model", lambda **_: object())

    def _boom(*_args, **_kwargs):
        raise RuntimeError("no vector store yet")

    monkeypatch.setattr(rag_agent.vectorstore, "search", _boom)

    project = Project(name="R", agent="rag", chat_model="llama3.2", system_prompt="BASE")
    agent = rag_agent.build_agent(project)
    qc = QueryConfig(thread_id="t1", project_id=project.id, chat_model="llama3.2")

    await agent.ainvoke(
        {"messages": [HumanMessage(content="hi")]}, qc.to_runnable_config()
    )

    assert captured["system"] == "BASE"
    assert "Retrieved context" not in captured["system"]
