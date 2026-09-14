"""Local (on-disk) Qdrant vector store for hybrid RAG retrieval.

Local mode holds a single-process directory lock, so a module-level client
singleton is reused for the whole app. Tests inject their own client via the
optional ``client`` parameter. ``QdrantVectorStore`` methods are synchronous;
callers should wrap them in ``asyncio.to_thread`` to keep the UI responsive.
"""

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http import models

from fp_int_agents.config import DATA_DIR

COLLECTION = "documents"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
SPARSE_MODEL_NAME = "Qdrant/bm25"
DEFAULT_TOP_K = 5

_client: QdrantClient | None = None
_sparse: FastEmbedSparse | None = None


def _qdrant_path() -> str:
    p = DATA_DIR / "qdrant"
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def get_client(path: str | None = None) -> QdrantClient:
    """Return a process-wide local Qdrant client (single-process file lock)."""
    global _client
    if _client is None:
        _client = QdrantClient(path=path or _qdrant_path())
    return _client


def get_sparse_embeddings() -> FastEmbedSparse:
    """Lazy BM25 sparse embedder. Downloads the model on first use."""
    global _sparse
    if _sparse is None:
        _sparse = FastEmbedSparse(model_name=SPARSE_MODEL_NAME)
    return _sparse


def _detect_dim(embeddings: Embeddings) -> int:
    return len(embeddings.embed_query("dimension probe"))


def _ensure_collection(client: QdrantClient, dim: int) -> None:
    if client.collection_exists(COLLECTION):
        return
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=dim, distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={SPARSE_VECTOR_NAME: models.SparseVectorParams()},
    )


def _ensure_payload_indexes(client: QdrantClient) -> None:
    for field in ("metadata.project_id", "metadata.source_type"):
        try:
            client.create_payload_index(
                COLLECTION,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception:  # noqa: BLE001, S110
            pass


def get_vector_store(
    embeddings: Embeddings, client: QdrantClient | None = None
) -> QdrantVectorStore:
    """Get/create the hybrid vector store for the given dense embedding model."""
    client = client or get_client()
    _ensure_collection(client, _detect_dim(embeddings))
    _ensure_payload_indexes(client)
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION,
        embedding=embeddings,
        sparse_embedding=get_sparse_embeddings(),
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=DENSE_VECTOR_NAME,
        sparse_vector_name=SPARSE_VECTOR_NAME,
    )


def add_chunks(
    embeddings: Embeddings,
    project_id: str,
    doc_id: str,
    chunks: list[str],
    title: str | None = None,
    client: QdrantClient | None = None,
    source_type: str = "document",
) -> list[str]:
    """Embed (dense + sparse) and store chunks; payload carries doc_id + project_id + title."""
    store = get_vector_store(embeddings, client)
    docs = [
        Document(
            page_content=chunk,
            metadata={
                "project_id": project_id,
                "doc_id": doc_id,
                "title": title or doc_id,
                "chunk_index": i,
                "source_type": source_type,
            },
        )
        for i, chunk in enumerate(chunks)
    ]
    return store.add_documents(docs)


def delete_by_doc_id(doc_id: str, client: QdrantClient | None = None) -> None:
    """Delete all chunks whose doc_id matches (thread memory or document chunks)."""
    c = client or get_client()
    if not c.collection_exists(COLLECTION):
        return
    c.delete(
        COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.doc_id",
                        match=models.MatchValue(value=doc_id),
                    )
                ]
            )
        ),
    )


def delete_by_project_id(project_id: str, client: QdrantClient | None = None) -> None:
    """Delete all chunks belonging to a project."""
    c = client or get_client()
    if not c.collection_exists(COLLECTION):
        return
    c.delete(
        COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.project_id",
                        match=models.MatchValue(value=project_id),
                    )
                ]
            )
        ),
    )


def search(
    embeddings: Embeddings,
    project_id: str,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    client: QdrantClient | None = None,
) -> list[Document]:
    """Hybrid (dense + sparse) search, filtered to one project via payload."""
    store = get_vector_store(embeddings, client)
    flt = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.project_id",
                match=models.MatchValue(value=project_id),
            )
        ]
    )
    return store.similarity_search(query, k=top_k, filter=flt)
