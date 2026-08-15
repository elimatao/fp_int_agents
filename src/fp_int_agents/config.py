import pathlib
import tomllib

from pydantic import BaseModel

DATA_DIR = pathlib.Path(__file__).parent.parent.parent / "data"


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


class Defaults(BaseModel):
    chat_model: str = "ollama/llama3.2"
    embedding_model: str = "ollama/nomic-embed-text"
    agent: str = "simple"


def load_defaults() -> Defaults:
    path = DATA_DIR / "defaults.toml"
    if not path.exists():
        return Defaults()
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return Defaults.model_validate(raw.get("defaults", {}))
