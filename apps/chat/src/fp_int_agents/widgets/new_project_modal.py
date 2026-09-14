from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, RadioButton, RadioSet, TextArea

_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

_PROJECT_TYPES = ["simple", "BundesRAG"]


@dataclass
class NewProjectResult:
    name: str
    agent: str
    system_prompt: str


class NewProjectModal(ModalScreen[NewProjectResult | None]):
    """Modal to create a project: name, project type, and system prompt."""

    DEFAULT_CSS = """
    NewProjectModal {
        align: center middle;
    }
    NewProjectModal > .dialog {
        width: 64;
        height: auto;
        max-height: 90%;
        background: $surface;
        border: tall $primary;
        padding: 1 2;
        layout: vertical;
    }
    NewProjectModal > .dialog > Label {
        margin-top: 1;
        text-style: bold;
    }
    NewProjectModal > .dialog > Label.first {
        margin-top: 0;
    }
    NewProjectModal > .dialog > RadioSet {
        height: auto;
        max-height: 8;
        border: none;
        padding: 0;
    }
    NewProjectModal > .dialog > TextArea {
        height: 6;
    }
    NewProjectModal > .dialog > .buttons {
        height: auto;
        layout: horizontal;
        align-horizontal: right;
        margin-top: 1;
    }
    NewProjectModal > .dialog > .buttons > Button {
        margin-left: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Container(classes="dialog"):
            yield Label("Name", classes="first")
            yield Input(placeholder="New Project", id="project-name")
            yield Label("Project type")
            yield RadioSet(
                *[RadioButton(t, value=(t == _PROJECT_TYPES[0])) for t in _PROJECT_TYPES],
                id="type-radio",
            )
            yield Label("System prompt")
            yield TextArea(_DEFAULT_SYSTEM_PROMPT, id="system-prompt")
            with Container(classes="buttons"):
                yield Button("Cancel", variant="default", id="cancel")
                yield Button("Create", variant="primary", id="confirm")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm":
            name = self.query_one("#project-name", Input).value.strip() or "New Project"
            radio = self.query_one("#type-radio", RadioSet)
            agent = (
                str(radio.pressed_button.label)
                if radio.pressed_button is not None
                else _PROJECT_TYPES[0]
            )
            system_prompt = self.query_one("#system-prompt", TextArea).text
            self.dismiss(NewProjectResult(name=name, agent=agent, system_prompt=system_prompt))
        elif event.button.id == "cancel":
            self.dismiss(None)
