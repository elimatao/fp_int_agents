import operator
from typing import Annotated, Literal

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from typing_extensions import TypedDict

from fp_int_agents.config import LlmConfig, Project, QueryConfig
from fp_int_agents.llm.client import get_chat_model
from fp_int_agents.tools.registry import dispatch, get_tools


class SimpleAgentInitConfig(BaseModel):
    """Immutable agent-specific config for the simple agent (stored in Project.init_config)."""


class SimpleAgentConfig(BaseModel):
    """Mutable agent-specific config for the simple agent (stored in Project.config)."""


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


_DEFAULT_LLM_CONFIG = LlmConfig()


def build_agent(
    project: Project,
    checkpointer: BaseCheckpointSaver | None = None,
    llm_config: LlmConfig = _DEFAULT_LLM_CONFIG,
) -> CompiledStateGraph:
    SimpleAgentInitConfig.model_validate(project.init_config)
    SimpleAgentConfig.model_validate(project.config)

    def _build_model(config: RunnableConfig) -> Runnable[LanguageModelInput, AIMessage]:
        qc = QueryConfig.from_runnable_config(config)
        tools = get_tools(qc.active_tools)
        model = get_chat_model(
            base_url=llm_config.base_url, model=qc.chat_model, api_key=llm_config.api_key
        )
        if tools:
            model = model.bind_tools(tools)
        return model

    def _llm_call(state: AgentState, config: RunnableConfig) -> AgentState:
        return {
            "messages": [
                _build_model(config).invoke(
                    [SystemMessage(content="You are a helpful assistant.")]
                    + state["messages"]
                )
            ]
        }

    async def _tool_node(state: AgentState, config: RunnableConfig) -> AgentState:
        tool_calls = state["messages"][-1].tool_calls
        return {"messages": await dispatch(tool_calls)}

    def _should_continue(state: AgentState) -> Literal["tool_node", "__end__"]:
        return "tool_node" if state["messages"][-1].tool_calls else END

    builder = StateGraph(AgentState)
    builder.add_node("llm_call", _llm_call)
    builder.add_node("tool_node", _tool_node)
    builder.add_edge(START, "llm_call")
    builder.add_conditional_edges("llm_call", _should_continue, ["tool_node", END])
    builder.add_edge("tool_node", "llm_call")
    return builder.compile(checkpointer=checkpointer)
