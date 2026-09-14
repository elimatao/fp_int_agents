import pytest
from fp_int_agents.config import load_config


@pytest.mark.asyncio
async def test_load_config_merges_root_urls_and_app_defaults(
    tmp_path, monkeypatch
) -> None:
    # 1. Create root config.toml with URLs only
    root_toml = tmp_path / "config.toml"
    root_toml.write_text(
        """
[llm]
base_url = "http://custom-proxy:4000/v1"
api_key = "secret-key-123"

[reranker]
url = "http://custom-reranker:8001/v1/rerank"
""",
        encoding="utf-8",
    )

    # 2. Create apps/chat/config.toml with app defaults only
    app_dir = tmp_path / "apps" / "chat"
    app_dir.mkdir(parents=True, exist_ok=True)
    app_toml = app_dir / "config.toml"
    app_toml.write_text(
        """
[defaults]
chat_model = "custom-llm"
embedding_model = "custom-embed"
agent = "BundesRAG"
mem_agent = "simple"
db_path = "custom/path.db"
""",
        encoding="utf-8",
    )

    # Monkeypatch paths
    monkeypatch.chdir(tmp_path)

    cfg = await load_config()

    # Root URLs verified
    assert cfg.llm.base_url == "http://custom-proxy:4000/v1"
    assert cfg.llm.api_key == "secret-key-123"
    assert cfg.reranker_url == "http://custom-reranker:8001/v1/rerank"

    # App defaults verified
    assert cfg.chat_model == "custom-llm"
    assert cfg.embedding_model == "custom-embed"
    assert cfg.agent == "BundesRAG"
    assert cfg.mem_agent == "simple"
    assert cfg.db_path == "custom/path.db"


@pytest.mark.asyncio
async def test_load_config_fallback_defaults(tmp_path, monkeypatch) -> None:
    # Empty dir - no config files
    monkeypatch.chdir(tmp_path)
    cfg = await load_config()

    assert cfg.llm.base_url == "http://localhost:4000/v1"
    assert cfg.reranker_url == "http://127.0.0.1:8001/v1/rerank"
    assert cfg.embedding_model == "multilingual-e5-small"
    assert cfg.agent == "simple"
