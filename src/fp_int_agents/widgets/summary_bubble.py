from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Markdown, Static


class SummaryBubble(Widget):
    """A non-persisted bubble showing the previous session's summary."""

    DEFAULT_CSS = """
    SummaryBubble {
        width: 100%;
        padding: 0 1;
        margin-bottom: 1;
        height: auto;
    }
    SummaryBubble .bubble-content {
        padding: 0 1;
        max-width: 80%;
        height: auto;
        background: $warning 15%;
        border-left: tall $warning 50%;
        color: $text;
    }
    SummaryBubble .bubble-role {
        text-style: bold dim;
        margin-bottom: 0;
    }
    """

    def __init__(self, summary: str) -> None:
        super().__init__()
        self._summary = summary

    def compose(self) -> ComposeResult:
        with Widget(classes="bubble-content"):
            yield Static("Session Summary", classes="bubble-role")
            yield Markdown(self._summary)
