from collections.abc import Callable

from langgraph.graph.state import CompiledStateGraph

from fp_int_agents.config import ProjectConfig

AgentFactory = Callable[[ProjectConfig], CompiledStateGraph]

AGENTS: dict[str, AgentFactory] = {}


def register_agent(name: str, factory: AgentFactory) -> None:
    AGENTS[name] = factory
