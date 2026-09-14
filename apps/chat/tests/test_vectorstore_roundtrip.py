"""Hybrid add + search roundtrip against a local Qdrant store.

Marked ``network`` because FastEmbedSparse downloads the Qdrant/bm25 model on
first use. Run with ``pytest -m network``.
"""

import pytest
from langchain_core.embeddings import Embeddings
from qdrant_client import QdrantClient

from fp_int_agents.storage import vectorstore


class FakeEmbeddings(Embeddings):
    """Deterministic dense embeddings so the test needs no live endpoint."""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for i, byte in enumerate(text.encode()):
            v[i % self.dim] += byte / 255.0
        return v

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


@pytest.mark.network
def test_hybrid_roundtrip_scoped_by_project(tmp_path) -> None:
    client = QdrantClient(path=str(tmp_path / "qd"))
    emb = FakeEmbeddings()

    vectorstore.add_chunks(
        emb,
        "proj-1",
        "doc-1",
        ["alpha content about cats", "beta content about dogs"],
        client=client,
    )
    vectorstore.add_chunks(
        emb, "proj-2", "doc-2", ["unrelated content about cars"], client=client
    )

    hits = vectorstore.search(emb, "proj-1", "cats", top_k=5, client=client)

    assert hits
    assert all(h.metadata["project_id"] == "proj-1" for h in hits)
