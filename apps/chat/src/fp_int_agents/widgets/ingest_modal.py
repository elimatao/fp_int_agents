from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label


class IngestModal(ModalScreen[str | None]):
    """Modal to pick a file path to ingest into the current project."""

    DEFAULT_CSS = """
    IngestModal {
        align: center middle;
    }
    IngestModal > .dialog {
        width: 60;
        height: auto;
        background: $surface;
        border: tall $primary;
        padding: 1 2;
        layout: vertical;
    }
    IngestModal > .dialog > Label {
        margin-top: 1;
        text-style: bold;
    }
    IngestModal > .dialog > Label.first {
        margin-top: 0;
    }
    IngestModal > .dialog > .buttons {
        height: auto;
        layout: horizontal;
        align-horizontal: right;
        margin-top: 1;
    }
    IngestModal > .dialog > .buttons > Button {
        margin-left: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Container(classes="dialog"):
            yield Label("File or folder path (.pdf or text)", classes="first")
            yield Input(placeholder="/path/to/file.pdf or /path/to/folder", id="ingest-path")
            with Container(classes="buttons"):
                yield Button("Cancel", variant="default", id="cancel")
                yield Button("Ingest", variant="primary", id="confirm")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm":
            path = self.query_one("#ingest-path", Input).value.strip()
            if not path:
                return
            self.dismiss(path)
        elif event.button.id == "cancel":
            self.dismiss(None)
