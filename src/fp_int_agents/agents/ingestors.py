"""Document ingestors: plain async functions registered in INGESTOR_AGENTS.

- ``simple``: summarize the text and store {summary, original} in SQLite. No
  embeddings; the summary is injected into the conv_agent_simple system prompt.
- ``rag``: chunk, dense-embed + BM25-sparse, store chunks in Qdrant (hybrid).
"""

import asyncio

import aiosqlite
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from fp_int_agents.agents.registry import register_ingestor
from fp_int_agents.config import LlmConfig, Project
from fp_int_agents.llm.client import get_chat_model, get_embedding_model
from fp_int_agents.storage import db, vectorstore

_SUMMARY_SYSTEM_PROMPT = (
    "You summarize documents. Produce a concise, factual summary capturing the "
    "key points, entities, and purpose of the text. No preamble, no meta-commentary."
)

_CHUNK_SIZE = 1000
_CHUNK_OVERLAP = 150
_DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"


async def ingest_simple(
    project: Project,
    text: str,
    llm_config: LlmConfig,
    conn: aiosqlite.Connection,
) -> None:
    model = get_chat_model(
        base_url=llm_config.base_url,
        model=project.chat_model,
        api_key=llm_config.api_key,
    )
    response = await model.ainvoke(
        [SystemMessage(content=_SUMMARY_SYSTEM_PROMPT), HumanMessage(content=text)]
    )
    await db.create_document(
        conn, project_id=project.id, original=text, summary=str(response.content)
    )


async def ingest_rag(
    project: Project,
    text: str,
    llm_config: LlmConfig,
    conn: aiosqlite.Connection,
) -> None:
    doc = await db.create_document(conn, project_id=project.id, original=text)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE, chunk_overlap=_CHUNK_OVERLAP
    )
    chunks = splitter.split_text(text)
    if not chunks:
        return
    embeddings = get_embedding_model(
        base_url=llm_config.base_url,
        model=project.init_config.get("embedding_model") or _DEFAULT_EMBEDDING_MODEL,
        api_key=llm_config.api_key,
    )
    await asyncio.to_thread(
        vectorstore.add_chunks, embeddings, project.id, doc.id, chunks
    )


register_ingestor("simple", ingest_simple)
register_ingestor("rag", ingest_rag)
