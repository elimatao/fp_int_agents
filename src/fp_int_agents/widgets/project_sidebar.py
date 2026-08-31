from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Collapsible

from fp_int_agents.config import Project, Thread
from fp_int_agents.widgets.thread_list import ThreadList


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
    ProjectSidebar Collapsible > CollapsibleTitle {
        padding: 0 1;
    }
    ProjectSidebar Collapsible > Contents {
        padding: 0;
    }
    ProjectSidebar Collapsible.--active > CollapsibleTitle {
        color: $accent;
        text-style: bold;
    }
    """

    class NewProject(Message):
        """User clicked '+ New Project'."""

    class NewThread(Message):
        def __init__(self, project: Project) -> None:
            super().__init__()
            self.project = project

    class ThreadSelected(Message):
        def __init__(self, thread: Thread, project: Project) -> None:
            super().__init__()
            self.thread = thread
            self.project = project

    def __init__(self) -> None:
        super().__init__()
        self._projects: list[Project] = []

    def compose(self) -> ComposeResult:
        yield Button("+ New Project", id="new-project-btn", variant="primary")
        yield VerticalScroll(id="project-scroll")

    async def populate(self, data: dict[Project, list[Thread]] | list[tuple[Project, list[Thread]]]) -> None:
        items = data.items() if isinstance(data, dict) else data
        scroll = self.query_one("#project-scroll", VerticalScroll)
        for project, threads in items:
            self._projects.append(project)
            await scroll.mount(self._make_collapsible(project, threads))

    def _make_collapsible(self, project: Project, threads: list[Thread]) -> Collapsible:
        return Collapsible(
            ThreadList(project, threads),
            title=project.name,
            collapsed=True,
            id=f"project-{project.id}",
        )

    async def add_project(self, project: Project, threads: list[Thread] | None = None) -> None:
        self._projects.insert(0, project)
        scroll = self.query_one("#project-scroll", VerticalScroll)
        await scroll.mount(self._make_collapsible(project, threads or []), before=0)

    async def add_thread(self, project: Project, thread: Thread) -> None:
        await self.query_one(f"#project-{project.id} ThreadList", ThreadList).add_thread(thread)

    def set_active_thread(self, project: Project, thread: Thread) -> None:
        """Expand+highlight the active project; highlight thread; clear others."""
        self.set_active_project(project)
        for tl in self.query(ThreadList):
            if tl._project.id == project.id:
                tl.set_active(thread.id)
            else:
                tl.set_active("")

    def set_active_project(self, project: Project) -> None:
        """Mark the collapsible for project as active and expand it."""
        for c in self.query(Collapsible):
            c.set_class(c.id == f"project-{project.id}", "--active")
        active = self.query_one(f"#project-{project.id}", Collapsible)
        active.collapsed = False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "new-project-btn":
            event.stop()
            self.post_message(self.NewProject())

    def on_thread_list_new_thread(self, event: ThreadList.NewThread) -> None:
        event.stop()
        self.post_message(self.NewThread(event.project))

    def on_thread_list_thread_selected(self, event: ThreadList.ThreadSelected) -> None:
        event.stop()
        self.post_message(self.ThreadSelected(event.thread, event.project))
