import pathlib
import tomllib
from typing import Any

from pydantic import BaseModel

DATA_DIR = pathlib.Path("data")


class ProjectInit(BaseModel, frozen=True):
    """Frozen at project-init time. Never mutated after creation."""

    project_id: str
    agent: str
    embedding_model: str
    extra: dict = {}


class ProjectMeta(BaseModel):
    """Mutable project metadata — editable after creation."""

    name: str
    persona: str | None = None


class ProjectConfig(BaseModel):
    """Full project state: frozen init config + mutable metadata."""

    init: ProjectInit
    meta: ProjectMeta


class QueryConfig(BaseModel):
    """Per-invocation config passed via LangGraph configurable."""

    thread_id: str
    project_id: str
    chat_model: str
    retrieval_effort: str = "medium"  # "low" | "medium" | "high"
    active_tools: list[str] = []


class LlmConfig(BaseModel):
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"


class AppConfig(BaseModel):
    chat_model: str = "llama3.2"
    embedding_model: str = "nomic-embed-text"
    agent: str = "simple"
    llm: LlmConfig = LlmConfig()


def _load_toml(path: pathlib.Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_config() -> AppConfig:
    override_path = pathlib.Path("config.toml")
    if not override_path.exists():
        return AppConfig()
    raw = _load_toml(override_path)
    llm_raw = raw.pop("llm", {})
    return AppConfig.model_validate({**raw, "llm": llm_raw})
