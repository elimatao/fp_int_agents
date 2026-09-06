from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from typing_extensions import TypedDict

from fp_int_agents.config import LlmConfig, Project, QueryConfig
from fp_int_agents.llm.client import get_chat_model

_SYSTEM_PROMPT = (
    "You are a memory assistant. Given a conversation history, produce a concise summary "
    "of the key facts, decisions, and topics discussed. Be brief and factual."
)

_SUMMARIZE_PROMPT = "Summarize the conversation above into a short paragraph."


class MemAgentInitConfig(BaseModel):
    """Immutable config for the memory agent (stored in Project.init_config)."""


class MemAgentConfig(BaseModel):
    """Mutable config for the memory agent (stored in Project.config)."""


class MemAgentState(TypedDict):
    messages: list[AnyMessage]
    summary: str | None


_DEFAULT_LLM_CONFIG = LlmConfig()


def build_agent(
    project: Project,
    llm_config: LlmConfig = _DEFAULT_LLM_CONFIG,
) -> CompiledStateGraph:
    MemAgentInitConfig.model_validate(project.init_config)
    MemAgentConfig.model_validate(project.config)

    def _summarize(state: MemAgentState, config: RunnableConfig) -> MemAgentState:
        qc = QueryConfig.from_runnable_config(config)
        model = get_chat_model(
            base_url=llm_config.base_url,
            model=qc.chat_model,
            api_key=llm_config.api_key,
        )
        response = model.invoke(
            [SystemMessage(content=_SYSTEM_PROMPT)]
            + state["messages"]
            + [HumanMessage(content=_SUMMARIZE_PROMPT)]
        )
        return {"messages": state["messages"], "summary": response.content}

    builder = StateGraph(MemAgentState)
    builder.add_node("summarize", _summarize)
    builder.add_edge(START, "summarize")
    builder.add_edge("summarize", END)
    return builder.compile()
