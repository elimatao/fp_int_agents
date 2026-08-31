from textual.app import ComposeResult
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Label, ListItem, ListView

from fp_int_agents.config import Thread


class ChatListSidebar(Widget):
    """Sidebar with a 'New Chat' button and a list of existing threads."""

    DEFAULT_CSS = """
    ChatListSidebar {
        width: 24;
        height: 100%;
        layout: vertical;
        border-right: solid $surface-darken-1;
    }
    ChatListSidebar > Button {
        width: 100%;
        margin: 1 1 0 1;
    }
    ChatListSidebar > ListView {
        width: 100%;
        height: 1fr;
        margin-top: 1;
        border: none;
        background: transparent;
    }
    ChatListSidebar ListView > ListItem {
        padding: 0 1;
    }
    """

    class NewChat(Message):
        """Posted when the user wants to start a new chat."""

    class ThreadSelected(Message):
        """Posted when the user selects a thread from the list."""

        def __init__(self, thread: Thread) -> None:
            super().__init__()
            self.thread = thread

    def __init__(self, threads: list[Thread]) -> None:
        super().__init__()
        self._threads = list(threads)

    def compose(self) -> ComposeResult:
        yield Button("+ New Chat", id="new-chat-btn", variant="primary")
        with ListView() as lv:
            self._list_view = lv
            for thread in self._threads:
                yield ListItem(
                    Label(thread.title or "Untitled"), id=f"thread-{thread.id}"
                )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "new-chat-btn":
            self.post_message(self.NewChat())

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item_id = event.item.id or ""
        if item_id.startswith("thread-"):
            thread_id = item_id.removeprefix("thread-")
            thread = next((t for t in self._threads if t.id == thread_id), None)
            if thread:
                self.post_message(self.ThreadSelected(thread))

    def add_thread(self, thread: Thread) -> None:
        """Prepend a newly created thread to the list."""
        self._threads.insert(0, thread)
        item = ListItem(Label(thread.title or "Untitled"), id=f"thread-{thread.id}")
        self.query_one(ListView).mount(item, before=0)
