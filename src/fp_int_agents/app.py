from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Label

from .config import Defaults, load_defaults
from .constants import APP_NAME


class FPIntAgentsApp(App):
    TITLE = APP_NAME

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label(f"Welcome to {self.TITLE}. No project loaded.")
        yield Footer()

    def on_mount(self) -> None:
        self.theme = "tokyo-night"
        self._defaults: Defaults = load_defaults()
