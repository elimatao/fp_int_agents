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

    class SettingsRequested(Message):
        """Posted when the user wants to open thread settings."""

        def __init__(
            self,
            current_model: str,
            active_tools: list[str],
            all_tools: list[str],
        ) -> None:
            super().__init__()
            self.current_model = current_model
            self.active_tools = active_tools
            self.all_tools = all_tools

    def __init__(
        self, active_tools: list[str] | None = None, chat_model: str = ""
    ) -> None:
        super().__init__()
        self._active_tools: list[str] = active_tools or []
        self._chat_model: str = chat_model

    def compose(self) -> ComposeResult:
        yield Button("Settings", variant="default", id="chat-settings")
        yield Input(placeholder="Type a message…", id="chat-input")
        yield Button("Send", variant="primary", id="chat-send")

    def update_settings(self, active_tools: list[str], chat_model: str) -> None:
        self._active_tools = active_tools
        self._chat_model = chat_model

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._submit(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "chat-send":
            self._submit(self.query_one("#chat-input", Input).value)
        elif event.button.id == "chat-settings":
            refresh()
            self.post_message(
                self.SettingsRequested(self._chat_model, self._active_tools, list_tools())
            )

    def _submit(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self.query_one("#chat-input", Input).clear()
        self.post_message(self.Submitted(text))
