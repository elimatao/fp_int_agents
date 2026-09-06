from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.events import Click
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button

from fp_int_agents.config import Project, Thread
from fp_int_agents.widgets.thread_list import ThreadList


class _ProjectRow(Widget):
    """Project header row (toggle + name + delete) above a collapsible ThreadList."""

    collapsed: reactive[bool] = reactive(True)

    DEFAULT_CSS = """
    _ProjectRow {
        height: auto;
        layout: vertical;
        width: 100%;
    }
    _ProjectRow > Horizontal {
        height: 1;
        width: 100%;
    }
    _ProjectRow > ThreadList {
        display: none;
    }
    _ProjectRow.-expanded > ThreadList {
        display: block;
    }
    _ProjectRow.--active _NameLabel {
        color: $accent;
        text-style: bold;
    }
    _ProjectRow.--active _ToggleLabel {
        color: $accent;
    }
    """

    class _ToggleLabel(Widget):
        DEFAULT_CSS = """
        _ToggleLabel { width: 2; height: 1; color: $foreground; background: transparent; }
        _ToggleLabel:hover { color: $accent; }
        """

        def __init__(self, row: "_ProjectRow", **kwargs) -> None:
            super().__init__(**kwargs)
            self._row = row

        def render(self) -> str:
            return "▼" if not self._row.collapsed else "▶"

        def on_click(self, event: Click) -> None:
            event.stop()
            self._row.collapsed = not self._row.collapsed

    class _NameLabel(Widget):
        DEFAULT_CSS = """
        _NameLabel { width: 1fr; height: 1; padding: 0 1; color: $foreground; background: transparent; }
        _NameLabel:hover { color: $accent; }
        """

        def __init__(self, name: str, row: "_ProjectRow", **kwargs) -> None:
            super().__init__(**kwargs)
            self._name = name
            self._row = row

        def render(self) -> str:
            return self._name

        def on_click(self, event: Click) -> None:
            event.stop()
            self._row.collapsed = not self._row.collapsed

    class _DeleteLabel(Widget):
        DEFAULT_CSS = """
        _DeleteLabel { width: 2; height: 1; color: $error; background: transparent; }
        _DeleteLabel:hover { text-style: bold; }
        """

        def __init__(self, project: "Project", **kwargs) -> None:
            super().__init__(**kwargs)
            self._project_id = project.id

        def render(self) -> str:
            return "✕"

        def on_click(self, event: Click) -> None:
            event.stop()
            self.post_message(ProjectSidebar.DeleteProject(self._project_id))

    def __init__(self, project: Project, threads: list[Thread]) -> None:
        super().__init__(id=f"row-{project.id}")
        self._project = project
        self._threads = threads

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield self._ToggleLabel(self)
            yield self._NameLabel(
                self._project.name, self, id=f"name-project-{self._project.id}"
            )
            yield self._DeleteLabel(self._project, id=f"del-project-{self._project.id}")
        yield ThreadList(self._project, self._threads, id=f"tl-{self._project.id}")

    def watch_collapsed(self, collapsed: bool) -> None:
        self.set_class(not collapsed, "-expanded")
        self.query_one(self._ToggleLabel).refresh()

    def expand(self) -> None:
        self.collapsed = False

    @property
    def thread_list(self) -> ThreadList:
        return self.query_one(ThreadList)


class ProjectSidebar(Widget):
    """Sidebar showing all projects as collapsed sections, each with a ThreadList."""

    DEFAULT_CSS = """
    ProjectSidebar {
        width: 26;
        height: 100%;
        layout: vertical;
        border-right: solid $panel;
    }
    ProjectSidebar > Button {
        width: 100%;
        margin: 1 1 0 1;
    }
    ProjectSidebar > VerticalScroll {
        height: 1fr;
        margin-top: 1;
    }
    """

    class NewProject(Message):
        """User clicked '+ New Project'."""

    class NewThread(Message):
        def __init__(self, project_id: str) -> None:
            super().__init__()
            self.project_id = project_id

    class ThreadSelected(Message):
        def __init__(self, thread_id: str, project_id: str) -> None:
            super().__init__()
            self.thread_id = thread_id
            self.project_id = project_id

    class DeleteThread(Message):
        def __init__(self, thread_id: str, project_id: str) -> None:
            super().__init__()
            self.thread_id = thread_id
            self.project_id = project_id

    class DeleteProject(Message):
        def __init__(self, project_id: str) -> None:
            super().__init__()
            self.project_id = project_id

    def compose(self) -> ComposeResult:
        yield Button("+ New Project", id="new-project-btn", variant="primary")
        yield VerticalScroll(id="project-scroll")

    async def populate(self, data: list[tuple[Project, list[Thread]]]) -> None:
        scroll = self.query_one("#project-scroll", VerticalScroll)
        for project, threads in data:
            await scroll.mount(_ProjectRow(project, threads))

    async def add_project(
        self, project: Project, threads: list[Thread] | None = None
    ) -> None:
        scroll = self.query_one("#project-scroll", VerticalScroll)
        await scroll.mount(_ProjectRow(project, threads or []), before=0)

    async def add_thread(self, project: Project, thread: Thread) -> None:
        await self.query_one(f"#tl-{project.id}", ThreadList).add_thread(thread)

    async def remove_thread(self, project_id: str, thread_id: str) -> None:
        await self.query_one(f"#tl-{project_id}", ThreadList).remove_thread(thread_id)

    async def remove_project(self, project_id: str) -> None:
        await self.query_one(f"#row-{project_id}", _ProjectRow).remove()

    def set_active_thread(self, project_id: str, thread_id: str) -> None:
        for row in self.query(_ProjectRow):
            is_active = row._project.id == project_id
            row.set_class(is_active, "--active")
            if is_active:
                row.expand()
                row.thread_list.set_active(thread_id)
            else:
                row.thread_list.set_active("")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "new-project-btn":
            event.stop()
            self.post_message(self.NewProject())

    def on_thread_list_new_thread(self, event: ThreadList.NewThread) -> None:
        event.stop()
        self.post_message(self.NewThread(event.project_id))

    def on_thread_list_thread_selected(self, event: ThreadList.ThreadSelected) -> None:
        event.stop()
        self.post_message(self.ThreadSelected(event.thread_id, event.project_id))

    def on_thread_list_delete_thread(self, event: ThreadList.DeleteThread) -> None:
        event.stop()
        self.post_message(self.DeleteThread(event.thread_id, event.project_id))
