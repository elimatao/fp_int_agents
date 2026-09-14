from fp_int_agents.agents.agent_caller import _RAG_NODE_LABELS
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static


class RagLogBubble(Widget):
    """A log bubble for RAG pipeline internal steps (rewriter, judge)."""

    DEFAULT_CSS = """
    RagLogBubble {
        width: 100%;
        padding: 0 1;
        margin-bottom: 1;
        height: auto;
    }
    RagLogBubble .bubble-content {
        padding: 0 1;
        max-width: 80%;
        height: auto;
        background: $panel;
        color: $text-muted;
        border-left: wide $warning;
    }
    RagLogBubble .rag-header {
        text-style: bold;
        color: $warning;
    }
    RagLogBubble .rag-result {
        color: $text-muted;
    }
    """

    def __init__(self, node: str) -> None:
        super().__init__()
        self._node = node
        self._label = _RAG_NODE_LABELS.get(node, node)

    def compose(self) -> ComposeResult:
        with Widget(classes="bubble-content"):
            yield Static(f"◈ {self._label}", classes="rag-header")
            yield Static("…", classes="rag-result")

    def set_result(self, result: str) -> None:
        self.query_one(".rag-result", Static).update(result)
