from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, RadioButton, RadioSet, TextArea

_NONE = "(none)"
_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


@dataclass
class NewProjectResult:
    name: str
    agent: str
    mem_agent: str | None
    ingestor: str | None
    system_prompt: str


class NewProjectModal(ModalScreen[NewProjectResult | None]):
    """Modal to create a project: name, agent pickers, and system prompt."""

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

    def __init__(
        self,
        conv_agents: list[str],
        mem_agents: list[str],
        ingestors: list[str],
    ) -> None:
        super().__init__()
        self._conv_agents = conv_agents or ["simple"]
        self._mem_agents = [_NONE, *mem_agents]
        self._ingestors = [_NONE, *ingestors]

    @staticmethod
    def _radio(options: list[str], default: str, radio_id: str) -> RadioSet:
        return RadioSet(
            *[RadioButton(o, value=(o == default)) for o in options],
            id=radio_id,
        )

    def compose(self) -> ComposeResult:
        with Container(classes="dialog"):
            yield Label("Name", classes="first")
            yield Input(placeholder="New Project", id="project-name")
            yield Label("Conversational agent")
            yield self._radio(self._conv_agents, self._conv_agents[0], "conv-radio")
            yield Label("Memory agent")
            yield self._radio(self._mem_agents, "simple"
                              if "simple" in self._mem_agents else _NONE, "mem-radio")
            yield Label("Ingestor")
            yield self._radio(self._ingestors, _NONE, "ingestor-radio")
            yield Label("System prompt")
            yield TextArea(_DEFAULT_SYSTEM_PROMPT, id="system-prompt")
            with Container(classes="buttons"):
                yield Button("Cancel", variant="default", id="cancel")
                yield Button("Create", variant="primary", id="confirm")

    def _selected(self, radio_id: str, fallback: str) -> str:
        radio = self.query_one(f"#{radio_id}", RadioSet)
        if radio.pressed_button is not None:
            return str(radio.pressed_button.label)
        return fallback

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm":
            name = self.query_one("#project-name", Input).value.strip() or "New Project"
            agent = self._selected("conv-radio", self._conv_agents[0])
            mem = self._selected("mem-radio", _NONE)
            ingestor = self._selected("ingestor-radio", _NONE)
            system_prompt = self.query_one("#system-prompt", TextArea).text
            self.dismiss(
                NewProjectResult(
                    name=name,
                    agent=agent,
                    mem_agent=None if mem == _NONE else mem,
                    ingestor=None if ingestor == _NONE else ingestor,
                    system_prompt=system_prompt,
                )
            )
        elif event.button.id == "cancel":
            self.dismiss(None)
