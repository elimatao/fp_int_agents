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
    created_at: str | None = None


class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
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
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"


class AppConfig(BaseModel):
    chat_model: str | None = None
    embedding_model: str = "nomic-embed-text"
    agent: str = "simple"
    llm: LlmConfig = LlmConfig()
    db_path: str = "data/app.db"


def _load_toml(path: pathlib.Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


async def load_config() -> AppConfig:
    from fp_int_agents.llm.models import list_models

    override_path = pathlib.Path("config.toml")
    if not override_path.exists():
        cfg = AppConfig()
    else:
        raw = _load_toml(override_path)
        llm_raw = raw.pop("llm", {})
        defaults = raw.pop("defaults", {})
        cfg = AppConfig.model_validate({**defaults, "llm": llm_raw})

    if cfg.chat_model is None:
        models = await list_models(cfg.llm.base_url, cfg.llm.api_key)
        if models:
            cfg.chat_model = models[0].id

    return cfg
