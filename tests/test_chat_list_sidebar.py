"""Unit tests for ChatListSidebar widget."""

import pytest

from fp_int_agents.config import Thread
from fp_int_agents.widgets.chat_list_sidebar import ChatListSidebar


def _make_thread(project_id: str, title: str | None = None) -> Thread:
    return Thread(project_id=project_id, title=title)


@pytest.mark.asyncio
async def test_sidebar_shows_new_chat_button() -> None:
    """Sidebar composes with a 'New Chat' button."""
    from textual.app import App, ComposeResult

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield ChatListSidebar(threads=[])

    async with TestApp().run_test() as pilot:
        assert pilot.app.query_one("#new-chat-btn")


@pytest.mark.asyncio
async def test_sidebar_lists_threads() -> None:
    """Sidebar renders one ListView item per thread."""
    from textual.app import App, ComposeResult
    from textual.widgets import ListView

    threads = [_make_thread("p1", "Chat 1"), _make_thread("p1", "Chat 2")]

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield ChatListSidebar(threads=threads)

    async with TestApp().run_test() as pilot:
        items = pilot.app.query_one(ListView).children
        assert len(items) == 2


@pytest.mark.asyncio
async def test_sidebar_new_chat_button_posts_message() -> None:
    """Pressing 'New Chat' triggers ChatListSidebar.NewChat message."""
    from textual.app import App, ComposeResult

    received: list = []

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield ChatListSidebar(threads=[])

        def on_chat_list_sidebar_new_chat(
            self, event: ChatListSidebar.NewChat
        ) -> None:
            received.append(event)

    async with TestApp().run_test() as pilot:
        await pilot.click("#new-chat-btn")
        assert len(received) == 1
