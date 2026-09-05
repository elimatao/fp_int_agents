import importlib
import json
import subprocess
from pathlib import Path

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, field_validator

_REGISTRY: dict[str, BaseTool] = {}

_TOOLS_JSON = Path(__file__).parent / "tools.json"
_TOOLS_MODULE = "fp_int_agents.tools.tools"


class JsonToolParameter(BaseModel):
    type: str
    description: str = ""
    enum: list[str] | None = None


class JsonToolParameters(BaseModel):
    type: str = "object"
    properties: dict[str, JsonToolParameter] = {}
    required: list[str] = []


class JsonToolDefinition(BaseModel):
    name: str
    description: str
    command: str
    parameters: JsonToolParameters = JsonToolParameters()

    @field_validator("command")
    @classmethod
    def command_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("command must not be empty")
        return v


def _make_shell_tool(definition: JsonToolDefinition) -> BaseTool:
    command = definition.command

    def _run(**kwargs: object) -> str:
        result = subprocess.run(
            command,
            input=json.dumps(kwargs),
            capture_output=True,
            text=True,
            shell=True,
            check=False,
        )
        if result.returncode != 0:
            return f"Error (exit {result.returncode}): {result.stderr.strip()}"
        return result.stdout.strip()

    return StructuredTool.from_function(
        func=_run,
        name=definition.name,
        description=definition.description,
    )


def _load_json_tools() -> None:
    if not _TOOLS_JSON.exists():
        return
    raw = json.loads(_TOOLS_JSON.read_text())
    entries = raw if isinstance(raw, list) else [raw]
    for entry in entries:
        defn = JsonToolDefinition.model_validate(entry)
        t = _make_shell_tool(defn)
        _REGISTRY[t.name] = t


def _load_decorator_tools() -> None:
    module = importlib.import_module(_TOOLS_MODULE)
    importlib.reload(module)
    for attr in vars(module).values():
        if isinstance(attr, BaseTool):
            _REGISTRY[attr.name] = attr


def refresh() -> None:
    """Reload all tools from tools.py and tools.json."""
    _load_decorator_tools()
    _load_json_tools()


def register(t: BaseTool) -> None:
    _REGISTRY[t.name] = t


def get_tools(names: list[str]) -> list[BaseTool]:
    return [_REGISTRY[n] for n in names if n in _REGISTRY]


def list_tools() -> list[str]:
    return list(_REGISTRY.keys())


async def dispatch(tool_calls: list[dict]) -> list[ToolMessage]:
    """Execute a list of LLM tool calls and return ToolMessages."""
    results = []
    for tc in tool_calls:
        t = _REGISTRY.get(tc["name"])
        if t is None:
            content = f"Error: tool '{tc['name']}' is not available."
        else:
            content = await t.ainvoke(tc["args"])
        results.append(
            ToolMessage(content=content, tool_call_id=tc["id"], name=tc["name"])
        )
    return results


refresh()
