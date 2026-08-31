from textual.app import ComposeResult
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Input


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

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Type a message…", id="chat-input")
        yield Button("Send", variant="primary", id="chat-send")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._submit(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "chat-send":
            self._submit(self.query_one("#chat-input", Input).value)

    def _submit(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self.query_one("#chat-input", Input).clear()
        self.post_message(self.Submitted(text))
