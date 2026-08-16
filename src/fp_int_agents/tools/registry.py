from langchain_core.tools import BaseTool, tool

_REGISTRY: dict[str, BaseTool] = {}


def register(tool: BaseTool) -> None:
    _REGISTRY[tool.name] = tool


def get_tools(names: list[str]) -> list[BaseTool]:
    return [_REGISTRY[n] for n in names if n in _REGISTRY]


def get_tool(name: str) -> BaseTool:
    tool = _REGISTRY.get(name)
    if tool is None:
        return _unknown_tool(name)
    return tool


def list_tools() -> list[str]:
    return list(_REGISTRY.keys())


@tool
def _unknown_tool(name: str) -> str:
    """Return an error message for an unregistered tool."""
    return f"Error: tool '{name}' does not exist."
