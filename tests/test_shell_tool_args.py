"""JSON-defined shell tools must forward args to the script, not nest them under 'kwargs'."""

import pytest

from fp_int_agents.tools.registry import (
    JsonToolDefinition,
    _make_shell_tool,
)


@pytest.mark.asyncio
async def test_shell_tool_forwards_named_args() -> None:
    definition = JsonToolDefinition.model_validate(
        {
            "name": "echo_location",
            "description": "Echo the location arg back.",
            # Reads JSON from stdin and prints the 'location' field.
            "command": (
                "python3 -c "
                "'import json,sys; print(json.load(sys.stdin)[\"location\"])'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "units": {"type": "string", "description": "Temperature unit"},
                },
                "required": ["location"],
            },
        }
    )
    tool = _make_shell_tool(definition)

    result = await tool.ainvoke({"location": "Potsdam, Germany"})

    assert result == "Potsdam, Germany"
