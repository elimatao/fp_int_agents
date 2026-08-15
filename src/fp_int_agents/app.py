import pathlib
import sys

from textual.app import App, ComposeResult
from textual.widgets import Footer, Header, Label

from .config import DATA_DIR, AppConfig, load_config
from .constants import APP_NAME


def _check_project_root() -> None:
    if not pathlib.Path("pyproject.toml").exists():
        print("Error: must be run from the project root (no pyproject.toml found).")
        sys.exit(1)
    DATA_DIR.mkdir(exist_ok=True)


class FPIntAgentsApp(App):
    TITLE = APP_NAME

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label(f"Welcome to {self.TITLE}. No project loaded.")
        yield Footer()

    def on_mount(self) -> None:
        self.theme = "tokyo-night"
        self._config: AppConfig = load_config()
