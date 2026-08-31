"""Tests for ThreadList and ProjectSidebar widgets."""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Collapsible, ListView

from fp_int_agents.config import Project, Thread
from fp_int_agents.widgets.project_sidebar import ProjectSidebar
from fp_int_agents.widgets.thread_list import ThreadList


def _project(name: str = "P") -> Project:
    return Project(name=name, chat_model="llama3.2")


def _thread(project_id: str, title: str = "T") -> Thread:
    return Thread(project_id=project_id, title=title)


# ---------------------------------------------------------------------------
# ThreadList tests
# ---------------------------------------------------------------------------


class ThreadListApp(App):
    def __init__(self, project: Project, threads: list[Thread]) -> None:
        super().__init__()
        self._project = project
        self._threads = threads

    def compose(self) -> ComposeResult:
        yield ThreadList(self._project, self._threads)


@pytest.mark.asyncio
async def test_thread_list_has_new_chat_button() -> None:
    p = _project()
    async with ThreadListApp(p, []).run_test() as pilot:
        assert pilot.app.query_one("#new-thread-btn")


@pytest.mark.asyncio
async def test_thread_list_shows_threads() -> None:
    p = _project()
    threads = [_thread(p.id, "A"), _thread(p.id, "B")]
    async with ThreadListApp(p, threads).run_test() as pilot:
        items = pilot.app.query_one(ListView).children
        assert len(items) == 2


@pytest.mark.asyncio
async def test_thread_list_new_chat_posts_new_thread() -> None:
    received: list[ThreadList.NewThread] = []
    p = _project()

    class App_(App):
        def compose(self) -> ComposeResult:
            yield ThreadList(p, [])

        def on_thread_list_new_thread(self, event: ThreadList.NewThread) -> None:
            received.append(event)

    async with App_().run_test() as pilot:
        await pilot.click("#new-thread-btn")
        assert len(received) == 1
        assert received[0].project.id == p.id


@pytest.mark.asyncio
async def test_thread_list_item_click_posts_thread_selected() -> None:
    received: list[ThreadList.ThreadSelected] = []
    p = _project()
    t = _thread(p.id, "Chat 1")

    class App_(App):
        def compose(self) -> ComposeResult:
            yield ThreadList(p, [t])

        def on_thread_list_thread_selected(
            self, event: ThreadList.ThreadSelected
        ) -> None:
            received.append(event)

    async with App_().run_test() as pilot:
        await pilot.click(f"#thread-{t.id}")
        assert len(received) == 1
        assert received[0].thread.id == t.id
        assert received[0].project.id == p.id


@pytest.mark.asyncio
async def test_thread_list_add_thread_prepends() -> None:
    p = _project()
    t1 = _thread(p.id, "First")
    t2 = _thread(p.id, "Second")
    async with ThreadListApp(p, [t1]).run_test() as pilot:
        tl = pilot.app.query_one(ThreadList)
        await tl.add_thread(t2)
        await pilot.pause()
        items = list(pilot.app.query_one(ListView).children)
        assert len(items) == 2
        assert items[0].id == f"thread-{t2.id}"


# ---------------------------------------------------------------------------
# ProjectSidebar tests
# ---------------------------------------------------------------------------


class SidebarApp(App):
    def compose(self) -> ComposeResult:
        yield ProjectSidebar()


@pytest.mark.asyncio
async def test_sidebar_has_new_project_button() -> None:
    async with SidebarApp().run_test() as pilot:
        assert pilot.app.query_one("#new-project-btn")


@pytest.mark.asyncio
async def test_populate_creates_collapsible_per_project() -> None:
    p1, p2 = _project("A"), _project("B")
    async with SidebarApp().run_test() as pilot:
        sidebar = pilot.app.query_one(ProjectSidebar)
        await sidebar.populate([(p1, []), (p2, [])])
        await pilot.pause()
        assert len(pilot.app.query(Collapsible)) == 2


@pytest.mark.asyncio
async def test_projects_collapsed_by_default() -> None:
    p1, p2 = _project("A"), _project("B")
    async with SidebarApp().run_test() as pilot:
        sidebar = pilot.app.query_one(ProjectSidebar)
        await sidebar.populate([(p1, []), (p2, [])])
        await pilot.pause()
        for c in pilot.app.query(Collapsible):
            assert c.collapsed is True


@pytest.mark.asyncio
async def test_new_project_button_posts_message() -> None:
    received: list[ProjectSidebar.NewProject] = []

    class App_(App):
        def compose(self) -> ComposeResult:
            yield ProjectSidebar()

        def on_project_sidebar_new_project(
            self, event: ProjectSidebar.NewProject
        ) -> None:
            received.append(event)

    async with App_().run_test() as pilot:
        await pilot.click("#new-project-btn")
        assert len(received) == 1


@pytest.mark.asyncio
async def test_add_project_adds_collapsible() -> None:
    p = _project("Solo")
    async with SidebarApp().run_test() as pilot:
        sidebar = pilot.app.query_one(ProjectSidebar)
        await sidebar.populate({})
        await sidebar.add_project(p, threads=[])
        await pilot.pause()
        assert len(pilot.app.query(Collapsible)) == 1
        assert pilot.app.query_one(Collapsible).collapsed is True
