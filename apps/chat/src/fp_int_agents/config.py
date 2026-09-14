import pathlib
import tomllib
import uuid
from typing import Any

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

DATA_DIR = pathlib.Path("data")


class Project(BaseModel):
    """A project groups threads under a shared agent configuration.

    Immutable after creation: id, agent, chat_model, init_config.
    Mutable: name, system_prompt, config.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    agent: str = "simple"
    mem_agent: str | None = None  # None = memory disabled for this project
    ingestor: str | None = None  # None = ingestion disabled for this project
    chat_model: str
    system_prompt: str | None = None
    init_config: dict = {}  # Agent-Specific, Immutable
    config: dict = {}  # Agent-Specific, Mutable
    created_at: str | None = None


class Thread(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    title: str | None = None
    active_tools: list[str] = []
    chat_model: str | None = None  # None = use project default
    summary: str | None = None
    summary_message_count: int | None = None
    memory_message_count: int | None = None
    created_at: str | None = None


class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    title: str | None = None
    summary: str | None = None
    original: str
    created_at: str | None = None


class QueryConfig(BaseModel):
    """Per-invocation config passed via LangGraph configurable."""

    thread_id: str
    project_id: str
    chat_model: str
    active_tools: list[str] = []

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig) -> "QueryConfig":
        return cls.model_validate(config["configurable"])

    def to_runnable_config(self) -> RunnableConfig:
        return {"configurable": self.model_dump()}


class LlmConfig(BaseModel):
    base_url: str = "http://localhost:4000/v1"
    api_key: str = "none"


class AppConfig(BaseModel):
    chat_model: str | None = "qwen-kleine-anfragen"
    embedding_model: str = "multilingual-e5-small"
    reranker_url: str = "http://127.0.0.1:8001/v1/rerank"
    agent: str = "simple"
    mem_agent: str | None = None
    ingestor: str | None = None
    llm: LlmConfig = LlmConfig()
    db_path: str = "data/app.db"


def _load_toml(path: pathlib.Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


async def load_config() -> AppConfig:
    from fp_int_agents.llm.models import list_models

    # 1. Load app-specific defaults from apps/chat/config.toml (or local config.toml if in apps/chat/)
    app_toml = pathlib.Path("apps/chat/config.toml")
    if (
        not app_toml.exists()
        and pathlib.Path("config.toml").exists()
        and not pathlib.Path("apps").exists()
    ):
        app_toml = pathlib.Path("config.toml")

    app_defaults: dict[str, Any] = {}
    if app_toml.exists():
        raw_app = _load_toml(app_toml)
        app_defaults = raw_app.get("defaults", raw_app)

    # 2. Load root-level URLs / endpoints from root config.toml
    root_toml = pathlib.Path("config.toml")
    if not root_toml.exists() and pathlib.Path("../../config.toml").exists():
        root_toml = pathlib.Path("../../config.toml")

    llm_raw: dict[str, Any] = {}
    root_urls: dict[str, Any] = {}

    if root_toml.exists():
        raw_root = _load_toml(root_toml)
        llm_raw = raw_root.get("llm", {})

        if "reranker_url" in raw_root.get("llm", {}):
            root_urls["reranker_url"] = raw_root["llm"]["reranker_url"]

        # Backwards-compatibility: if defaults were in root config.toml and not yet in app config
        if not app_defaults and "defaults" in raw_root:
            app_defaults = raw_root["defaults"]

    root_urls = {k: v for k, v in root_urls.items() if v is not None}

    merged = {
        **app_defaults,
        **root_urls,
    }
    if llm_raw:
        merged["llm"] = llm_raw

    cfg = AppConfig.model_validate(merged)

    if cfg.chat_model is None:
        try:
            models = await list_models(cfg.llm.base_url, cfg.llm.api_key)
            if models:
                cfg.chat_model = models[0].id
        except Exception:  # noqa: BLE001, S110
            pass

    return cfg
