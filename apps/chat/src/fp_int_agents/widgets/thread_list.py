from fp_int_agents.config import Project, Thread
from textual.app import ComposeResult
from textual.events import Click
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Label, ListItem, ListView


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
        padding: 0 0 0 2;
    }
    ThreadList ListView > ListItem > Label {
        width: 1fr;
        padding: 0 3 0 0;
    }
    """

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

    class _DeleteLabel(Widget):
        DEFAULT_CSS = """
        _DeleteLabel { dock: right; width: 2; height: 1; color: $error; background: transparent; }
        _DeleteLabel:hover { text-style: bold; }
        """

        def __init__(self, thread_id: str, project_id: str, **kwargs) -> None:
            super().__init__(**kwargs)
            self._del_thread_id = thread_id
            self._del_project_id = project_id

        def render(self) -> str:
            return "✕"

        def on_click(self, event: Click) -> None:
            event.stop()
            self.post_message(
                ThreadList.DeleteThread(self._del_thread_id, self._del_project_id)
            )

    def __init__(self, project: Project, threads: list[Thread], **kwargs) -> None:
        super().__init__(**kwargs)
        self._project = project
        self._threads = list(threads)

    def _make_item(self, thread: Thread) -> ListItem:
        return ListItem(
            self._DeleteLabel(
                thread_id=thread.id,
                project_id=self._project.id,
                id=f"del-thread-{thread.id}",
            ),
            Label(thread.title or "Untitled"),
            id=f"thread-{thread.id}",
        )

    def compose(self) -> ComposeResult:
        yield Button("+ New Chat", id="new-thread-btn", variant="default")
        yield ListView(
            *[self._make_item(t) for t in self._threads],
            id="thread-lv",
        )

    async def add_thread(self, thread: Thread) -> None:
        self._threads.insert(0, thread)
        lv = self.query_one("#thread-lv", ListView)
        await lv.mount(self._make_item(thread), before=0)

    async def remove_thread(self, thread_id: str) -> None:
        self._threads = [t for t in self._threads if t.id != thread_id]
        item = self.query_one(f"#thread-{thread_id}", ListItem)
        await item.remove()

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
        if event.button.id == "new-thread-btn":
            self.post_message(self.NewThread(self._project.id))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        event.stop()
        item_id = event.item.id or ""
        if not item_id.startswith("thread-"):
            return
        thread_id = item_id.removeprefix("thread-")
        self.post_message(self.ThreadSelected(thread_id, self._project.id))
