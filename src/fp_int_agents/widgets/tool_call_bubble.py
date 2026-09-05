import json

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static


class ToolCallBubble(Widget):
    """A distinct bubble summarizing a single tool call and its result."""

    DEFAULT_CSS = """
    ToolCallBubble {
        width: 100%;
        padding: 0 1;
        margin-bottom: 1;
        height: auto;
    }
    ToolCallBubble .bubble-content {
        padding: 0 1;
        max-width: 80%;
        height: auto;
        background: $panel;
        color: $text-muted;
        border-left: wide $accent;
    }
    ToolCallBubble .tool-header {
        text-style: bold;
        color: $accent;
    }
    """

    def __init__(self, name: str, args: dict) -> None:
        super().__init__()
        self._name = name
        self._args = args

    def compose(self) -> ComposeResult:
        with Widget(classes="bubble-content"):
            yield Static(f"⚙ {self._name}", classes="tool-header")
            yield Static(self._format_args(), classes="tool-args")
            yield Static("running…", classes="tool-result")

    def _format_args(self) -> str:
        try:
            return json.dumps(self._args, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(self._args)

    def set_result(self, result: str) -> None:
        self.query_one(".tool-result", Static).update(result)
