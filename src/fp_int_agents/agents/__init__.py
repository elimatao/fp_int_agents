from fp_int_agents.agents.conv_agent_simple import (
    build_agent as build_simple_conv_agent,
)
from fp_int_agents.agents.mem_agent_simple import build_agent as build_simple_mem_agent
from fp_int_agents.agents.registry import (
    register_conversational_agent,
    register_memory_agent,
)

register_conversational_agent("simple", build_simple_conv_agent)
register_memory_agent("simple", build_simple_mem_agent)
