from collections.abc import Awaitable, Callable

import aiosqlite
from langchain_core.messages import AnyMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from fp_int_agents.config import LlmConfig, Project

ConversationalAgentFactory = Callable[
    [Project, BaseCheckpointSaver | None, LlmConfig, aiosqlite.Connection | None],
    CompiledStateGraph,
]

MemoryFn = Callable[
    [Project, list[AnyMessage], LlmConfig, aiosqlite.Connection | None, int, str],
    Awaitable[tuple[str, int]],
]

IngestorFn = Callable[[Project, str, LlmConfig, aiosqlite.Connection], Awaitable[None]]

CONVERSATIONAL_AGENTS: dict[str, ConversationalAgentFactory] = {}

MEMORY_AGENTS: dict[str, MemoryFn] = {}

INGESTOR_AGENTS: dict[str, IngestorFn] = {}


def register_conversational_agent(
    name: str, factory: ConversationalAgentFactory
) -> None:
    CONVERSATIONAL_AGENTS[name] = factory


def register_memory_agent(name: str, fn: MemoryFn) -> None:
    MEMORY_AGENTS[name] = fn


def register_ingestor(name: str, fn: IngestorFn) -> None:
    INGESTOR_AGENTS[name] = fn
