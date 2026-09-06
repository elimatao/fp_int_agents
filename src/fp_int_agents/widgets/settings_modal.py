from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Button, Label, RadioButton, RadioSet, SelectionList
from textual.widgets.selection_list import Selection


@dataclass
class SettingsResult:
    active_tools: list[str]
    chat_model: str | None  # None = keep project default


class SettingsModal(ModalScreen[SettingsResult | None]):
    """Modal for per-thread settings: model override and active tools."""

    DEFAULT_CSS = """
    SettingsModal {
        align: center middle;
    }
    SettingsModal > .dialog {
        width: 60;
        height: auto;
        max-height: 85%;
        background: $surface;
        border: tall $primary;
        padding: 1 2;
        layout: vertical;
    }
    SettingsModal > .dialog > Label {
        margin-top: 1;
        text-style: bold;
    }
    SettingsModal > .dialog > Label.first {
        margin-top: 0;
    }
    SettingsModal > .dialog > RadioSet {
        height: auto;
        max-height: 12;
        border: none;
        padding: 0;
    }
    SettingsModal > .dialog > SelectionList {
        height: auto;
        max-height: 12;
        border: none;
    }
    SettingsModal > .dialog > .buttons {
        height: auto;
        layout: horizontal;
        align-horizontal: right;
        margin-top: 1;
    }
    SettingsModal > .dialog > .buttons > Button {
        margin-left: 1;
    }
    """

    def __init__(
        self,
        current_model: str,
        available_models: list[str],
        active_tools: list[str],
        all_tools: list[str],
    ) -> None:
        super().__init__()
        self._current_model = current_model
        self._available_models = available_models
        self._active_tools = active_tools
        self._all_tools = all_tools

    def compose(self) -> ComposeResult:
        model_buttons = [
            RadioButton(m, value=(m == self._current_model), id=f"model-{i}")
            for i, m in enumerate(self._available_models)
        ]
        selections = [
            Selection(name, name, initial_state=name in self._active_tools)
            for name in self._all_tools
        ]
        with Container(classes="dialog"):
            yield Label("Model", classes="first")
            if model_buttons:
                yield RadioSet(*model_buttons, id="model-radio")
            else:
                yield Label("(no models available)")
            yield Label("Tools")
            yield SelectionList(*selections, id="tool-list")
            with Container(classes="buttons"):
                yield Button("Cancel", variant="default", id="cancel")
                yield Button("Confirm", variant="primary", id="confirm")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm":
            selected_tools = list(self.query_one("#tool-list", SelectionList).selected)
            model = self._current_model
            try:
                radio = self.query_one("#model-radio", RadioSet)
                if radio.pressed_button is not None:
                    label = str(radio.pressed_button.label)
                    if label in self._available_models:
                        model = label
            except Exception as exc:  # noqa: BLE001 - RadioSet may be absent when no models; fallback is safe
                self.log.warning(f"Could not read selected model: {exc}")
            self.dismiss(SettingsResult(active_tools=selected_tools, chat_model=model))
        elif event.button.id == "cancel":
            self.dismiss(None)
