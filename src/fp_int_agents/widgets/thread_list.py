from textual.app import ComposeResult
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Label, ListItem, ListView

from fp_int_agents.config import Project, Thread


class ThreadList(Widget):
    """List of threads for a single project, with a '+ New Chat' button."""

    DEFAULT_CSS = """
    ThreadList {
        height: auto;
        layout: vertical;
    }
    ThreadList > Button {
        width: 100%;
        height: auto;
        margin: 0;
    }
    ThreadList > ListView {
        height: auto;
        border: none;
        background: transparent;
        padding: 0;
        margin: 0;
    }
    ThreadList ListView > ListItem {
        padding: 0 1 0 2;
    }
    """

    class NewThread(Message):
        def __init__(self, project: Project) -> None:
            super().__init__()
            self.project = project

    class ThreadSelected(Message):
        def __init__(self, thread: Thread, project: Project) -> None:
            super().__init__()
            self.thread = thread
            self.project = project

    def __init__(self, project: Project, threads: list[Thread]) -> None:
        super().__init__()
        self._project = project
        self._threads = list(threads)

    def compose(self) -> ComposeResult:
        yield Button("+ New Chat", id="new-thread-btn", variant="default")
        yield ListView(
            *[
                ListItem(Label(t.title or "Untitled"), id=f"thread-{t.id}")
                for t in self._threads
            ],
            id="thread-lv",
        )

    async def add_thread(self, thread: Thread) -> None:
        self._threads.insert(0, thread)
        item = ListItem(Label(thread.title or "Untitled"), id=f"thread-{thread.id}")
        lv = self.query_one("#thread-lv", ListView)
        await lv.mount(item, before=0)

    def set_active(self, thread_id: str) -> None:
        """Highlight the item matching thread_id, clear all highlights if not found."""
        lv = self.query_one("#thread-lv", ListView)
        for item in lv.query(ListItem):
            item.highlighted = False
        lv.set_reactive(ListView.index, None)
        for i, t in enumerate(self._threads):
            if t.id == thread_id:
                lv.index = i
                return

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.NewThread(self._project))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        event.stop()
        item_id = event.item.id or ""
        if not item_id.startswith("thread-"):
            return
        thread_id = item_id.removeprefix("thread-")
        thread = next((t for t in self._threads if t.id == thread_id), None)
        if thread:
            self.post_message(self.ThreadSelected(thread, self._project))
