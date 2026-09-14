"""Tests for the embeddings server."""
import importlib
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client_and_model():
    with patch("sentence_transformers.SentenceTransformer") as mock_cls:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_cls.return_value = mock_model

        sys.modules.pop("services.model_server.serve_embeddings", None)
        import services.model_server.serve_embeddings as mod
        importlib.reload(mod)
        yield TestClient(mod.app), mock_model


def test_health(client_and_model: tuple) -> None:
    client, _ = client_and_model
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model" in data
    assert "device" in data


def test_embed_single_string(client_and_model: tuple) -> None:
    client, mock_model = client_and_model
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    resp = client.post("/v1/embeddings", json={"input": "hello world", "model": "multilingual-e5-small"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    assert body["data"][0]["object"] == "embedding"
    assert isinstance(body["data"][0]["embedding"], list)


def test_embed_list(client_and_model: tuple) -> None:
    client, mock_model = client_and_model
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    resp = client.post("/v1/embeddings", json={"input": ["hello", "world"], "model": "multilingual-e5-small"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["data"]) == 2
    assert body["data"][0]["index"] == 0
    assert body["data"][1]["index"] == 1


def test_text_passed_through_unchanged(client_and_model: tuple) -> None:
    """Server should not add any prefixes — that's the client's responsibility."""
    client, mock_model = client_and_model
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    client.post("/v1/embeddings", json={"input": "some text"})
    encoded = mock_model.encode.call_args[0][0]
    assert encoded == ["some text"]


def test_usage_tokens(client_and_model: tuple) -> None:
    client, mock_model = client_and_model
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    resp = client.post("/v1/embeddings", json={"input": "hello world"})
    body = resp.json()
    assert "usage" in body
    assert body["usage"]["prompt_tokens"] == 2
    assert body["usage"]["total_tokens"] == 2
