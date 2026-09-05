from textual.app import ComposeResult
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Input

from fp_int_agents.tools.registry import list_tools, refresh


class InputBar(Widget):
    """Text input bar with a submit button."""

    DEFAULT_CSS = """
    InputBar {
        height: auto;
        padding: 1 1 0 1;
        layout: horizontal;
    }
    InputBar > Input {
        width: 1fr;
    }
    InputBar > Button {
        width: auto;
        margin-left: 1;
    }
    """

    class Submitted(Message):
        """Posted when the user submits a message."""

        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    class ToolsRequested(Message):
        """Posted when the user wants to select tools."""

        def __init__(self, active_tools: list[str], all_tools: list[str]) -> None:
            super().__init__()
            self.active_tools = active_tools
            self.all_tools = all_tools

    def __init__(self, active_tools: list[str] | None = None) -> None:
        super().__init__()
        self._active_tools: list[str] = active_tools or []

    def compose(self) -> ComposeResult:
        yield Button(self._tools_label(), variant="default", id="chat-tools")
        yield Input(placeholder="Type a message…", id="chat-input")
        yield Button("Send", variant="primary", id="chat-send")

    def update_tools(self, active_tools: list[str]) -> None:
        self._active_tools = active_tools
        self.query_one("#chat-tools", Button).label = self._tools_label()

    def _tools_label(self) -> str:
        return f"Tools ({len(self._active_tools)}/{len(list_tools())})"

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._submit(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "chat-send":
            self._submit(self.query_one("#chat-input", Input).value)
        elif event.button.id == "chat-tools":
            refresh()
            all_tools = list_tools()
            self.post_message(self.ToolsRequested(self._active_tools, all_tools))

    def _submit(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self.query_one("#chat-input", Input).clear()
        self.post_message(self.Submitted(text))
