from collections.abc import Callable

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from fp_int_agents.config import Project

AgentFactory = Callable[[Project, BaseCheckpointSaver | None], CompiledStateGraph]

AGENTS: dict[str, AgentFactory] = {}


def register_agent(name: str, factory: AgentFactory) -> None:
    AGENTS[name] = factory
