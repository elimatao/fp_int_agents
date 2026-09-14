import fp_int_agents.agents.ingestors  # noqa: F401 - registers ingestors on import
from fp_int_agents.agents.conv_agent_simple import (
    build_agent as build_simple_conv_agent,
)
from fp_int_agents.agents.mem_agent_rag import memorize_thread as rag_mem_fn
from fp_int_agents.agents.mem_agent_simple import memorize_thread as simple_mem_fn
from fp_int_agents.agents.rag_agent import build_agent as build_rag_conv_agent
from fp_int_agents.agents.registry import (
    register_conversational_agent,
    register_memory_agent,
)

register_conversational_agent("simple", build_simple_conv_agent)
register_conversational_agent("BundesRAG", build_rag_conv_agent)
register_memory_agent("simple", simple_mem_fn)
register_memory_agent("BundesRAG", rag_mem_fn)
