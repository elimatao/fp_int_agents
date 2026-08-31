"""
Simple ReAct agent
"""

import operator
from typing import Annotated, Literal

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from typing_extensions import TypedDict

from fp_int_agents.config import Project, QueryConfig
from fp_int_agents.llm.client import get_chat_model
from fp_int_agents.tools.registry import get_tool, get_tools


class SimpleAgentInitConfig(BaseModel):
    """Immutable agent-specific config stored in Project.init_config for the simple agent.

    Deserialize from a project with:
        init_cfg = SimpleAgentInitConfig.model_validate(project.init_config)
    """


class SimpleAgentConfig(BaseModel):
    """Mutable agent-specific config stored in Project.config for the simple agent.

    Deserialize from a project with:
        agent_cfg = SimpleAgentConfig.model_validate(project.config)
    """


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


def _build_model(config: RunnableConfig) -> Runnable[LanguageModelInput, AIMessage]:
    """Instantiate the chat model and bind any active tools from runtime config."""
    from fp_int_agents.config import load_config

    qc = QueryConfig.from_runnable_config(config)
    app_cfg = load_config()
    tools = get_tools(qc.active_tools)
    model = get_chat_model(
        base_url=app_cfg.llm.base_url, model=qc.chat_model, api_key=app_cfg.llm.api_key
    )
    if tools:
        model = model.bind_tools(tools)
    return model


def _llm_call(state: AgentState, config: RunnableConfig) -> AgentState:
    model = _build_model(config)
    return {
        "messages": [
            model.invoke(
                [SystemMessage(content="You are a helpful assistant.")]
                + state["messages"]
            )
        ]
    }


def _tool_node(state: AgentState, config: RunnableConfig) -> AgentState:
    results = [
        ToolMessage(
            content=get_tool(tc["name"]).invoke(tc["args"]), tool_call_id=tc["id"]
        )
        for tc in state["messages"][-1].tool_calls
    ]
    return {"messages": results}


def _should_continue(state: AgentState) -> Literal["tool_node", "__end__"]:
    return "tool_node" if state["messages"][-1].tool_calls else END


def build_agent(
    project: Project,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    _ = SimpleAgentInitConfig.model_validate(project.init_config)
    _ = SimpleAgentConfig.model_validate(project.config)
    builder = StateGraph(AgentState)
    builder.add_node("llm_call", _llm_call)
    builder.add_node("tool_node", _tool_node)
    builder.add_edge(START, "llm_call")
    builder.add_conditional_edges("llm_call", _should_continue, ["tool_node", END])
    builder.add_edge("tool_node", "llm_call")
    return builder.compile(checkpointer=checkpointer)
