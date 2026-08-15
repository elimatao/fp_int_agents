from langchain_core.tools import BaseTool

_REGISTRY: dict[str, BaseTool] = {}


def register(tool: BaseTool) -> None:
    _REGISTRY[tool.name] = tool


def get_tools(names: list[str]) -> list[BaseTool]:
    return [_REGISTRY[n] for n in names if n in _REGISTRY]


def list_tools() -> list[str]:
    return list(_REGISTRY.keys())
