from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Button, SelectionList
from textual.widgets.selection_list import Selection


class ToolSelector(ModalScreen):
    """Modal screen for selecting active tools for the current thread."""

    DEFAULT_CSS = """
    ToolSelector {
        align: center middle;
    }
    ToolSelector > .dialog {
        width: 50;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: tall $primary;
        padding: 1 2;
        layout: vertical;
    }
    ToolSelector > .dialog > SelectionList {
        height: auto;
        max-height: 20;
        border: none;
    }
    ToolSelector > .dialog > .buttons {
        height: auto;
        layout: horizontal;
        align-horizontal: right;
        margin-top: 1;
    }
    ToolSelector > .dialog > .buttons > Button {
        margin-left: 1;
    }
    """

    def __init__(self, active_tools: list[str], all_tools: list[str]) -> None:
        super().__init__()
        self._active_tools = active_tools
        self._all_tools = all_tools

    def compose(self) -> ComposeResult:
        selections = [
            Selection(name, name, initial_state=name in self._active_tools)
            for name in self._all_tools
        ]
        with Container(classes="dialog"):
            yield SelectionList(*selections, id="tool-list")
            with Container(classes="buttons"):
                yield Button("Cancel", variant="default", id="cancel")
                yield Button("Confirm", variant="primary", id="confirm")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm":
            selected = self.query_one("#tool-list", SelectionList).selected
            self.dismiss(list(selected))
        elif event.button.id == "cancel":
            self.dismiss(None)
