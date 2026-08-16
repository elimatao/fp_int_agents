from fp_int_agents.agents.agent_simple import build_agent
from fp_int_agents.agents.registry import register_agent

register_agent("simple", build_agent)
