from langchain_core.messages import AIMessage, HumanMessage
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Markdown, Static


class MessageBubble(Widget):
    """A single chat message bubble."""

    DEFAULT_CSS = """
    MessageBubble {
        width: 100%;
        padding: 0 1;
        margin-bottom: 1;
        height: auto;
    }
    MessageBubble.human {
        align: right middle;
    }
    MessageBubble .bubble-content {
        padding: 0 1;
        max-width: 80%;
        height: auto;

    }
    MessageBubble.human .bubble-content {
        background: $primary 30%;
        color: $text;
    }
    MessageBubble.ai .bubble-content {
        background: $surface;
        color: $text;
    }
    .bubble-role {
        text-style: bold dim;
        margin-bottom: 0;
    }
    """

    def __init__(self, message: HumanMessage | AIMessage) -> None:
        is_human = isinstance(message, HumanMessage)
        super().__init__(classes="human" if is_human else "ai")
        self._content = message.content if isinstance(message.content, str) else ""

    def compose(self) -> ComposeResult:
        label = "You" if "human" in self.classes else "Assistant"
        with Widget(classes="bubble-content"):
            yield Static(label, classes="bubble-role")
            yield Markdown(self._content)

    def append_token(self, token: str) -> None:
        self._content += token
        self.query_one(Markdown).update(self._content)
