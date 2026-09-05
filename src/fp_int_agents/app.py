import asyncio

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Footer, Header

from .config import AppConfig, Project, QueryConfig, Thread
from .constants import APP_NAME
from .storage.db import (
    DB,
    create_project,
    create_thread,
    delete_project,
    delete_thread,
    list_projects,
    list_threads,
)
from .widgets.chat import Chat
from .widgets.project_sidebar import ProjectSidebar


class FPIntAgentsApp(App):
    TITLE = APP_NAME

    CSS = """
    #main-layout {
        width: 100%;
        height: 1fr;
    }
    Chat {
        width: 1fr;
    }
    """

    def __init__(self, db: DB, config: AppConfig) -> None:
        self.db = db
        self.config = config
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-layout"):
            yield ProjectSidebar()
        yield Footer()

    async def on_mount(self) -> None:
        self.theme = "tokyo-night"
        projects = await list_projects(self.db.conn)
        if not projects:
            project = await create_project(
                self.db.conn, "Default Project", self.config.chat_model
            )
            projects = [project]

        threads_per_project = await asyncio.gather(
            *[list_threads(self.db.conn, p.id) for p in projects]
        )
        data: list[tuple[Project, list[Thread]]] = []
        for p, threads in zip(projects, threads_per_project):
            if not threads:
                t = await create_thread(self.db.conn, p.id, "Default Thread Title")
                threads = [t]
            data.append((p, threads))

        sidebar = self.query_one(ProjectSidebar)
        await sidebar.populate(data)

        first_project, first_threads = data[0]
        first_thread = first_threads[0]
        query_config = QueryConfig(
            thread_id=first_thread.id,
            project_id=first_project.id,
            chat_model=first_project.chat_model,
            active_tools=first_thread.active_tools,
        )
        await self.query_one("#main-layout", Horizontal).mount(Chat(query_config))
        sidebar.set_active_thread(first_project, first_thread)

    async def _activate_thread(self, project: Project, thread: Thread) -> None:
        await self._swap_chat(
            QueryConfig(
                thread_id=thread.id,
                project_id=project.id,
                chat_model=project.chat_model,
                active_tools=thread.active_tools,
            )
        )
        self.query_one(ProjectSidebar).set_active_thread(project, thread)

    async def _swap_chat(self, query_config: QueryConfig) -> None:
        layout = self.query_one("#main-layout", Horizontal)
        await layout.query_one(Chat).remove()
        await layout.mount(Chat(query_config))

    async def on_project_sidebar_thread_selected(
        self, event: ProjectSidebar.ThreadSelected
    ) -> None:
        await self._activate_thread(event.project, event.thread)

    async def on_project_sidebar_new_project(
        self, _: ProjectSidebar.NewProject
    ) -> None:
        project = await create_project(
            self.db.conn, "New Project", self.config.chat_model
        )
        thread: Thread = await create_thread(self.db.conn, project.id)
        sidebar = self.query_one(ProjectSidebar)
        await sidebar.add_project(project, threads=[thread])
        await self._activate_thread(project, thread)

    async def on_project_sidebar_new_thread(
        self, event: ProjectSidebar.NewThread
    ) -> None:
        thread: Thread = await create_thread(self.db.conn, event.project.id)
        sidebar = self.query_one(ProjectSidebar)
        await sidebar.add_thread(event.project, thread)
        await self._activate_thread(event.project, thread)

    async def on_project_sidebar_delete_thread(
        self, event: ProjectSidebar.DeleteThread
    ) -> None:
        sidebar = self.query_one(ProjectSidebar)
        active_thread_id = self.query_one(Chat)._query_config.thread_id
        await delete_thread(self.db.conn, self.db.checkpointer, event.thread.id)
        await sidebar.remove_thread(event.project.id, event.thread.id)
        if event.thread.id == active_thread_id:
            threads = await list_threads(self.db.conn, event.project.id)
            if threads:
                await self._activate_thread(event.project, threads[0])
            else:
                new_thread: Thread = await create_thread(self.db.conn, event.project.id)
                await sidebar.add_thread(event.project, new_thread)
                await self._activate_thread(event.project, new_thread)

    async def on_project_sidebar_delete_project(
        self, event: ProjectSidebar.DeleteProject
    ) -> None:
        thread_ids = [t.id for t in await list_threads(self.db.conn, event.project.id)]
        await delete_project(
            self.db.conn, self.db.checkpointer, event.project.id, thread_ids
        )
        sidebar = self.query_one(ProjectSidebar)
        await sidebar.remove_project(event.project.id)
        if event.project.id == self.query_one(Chat)._query_config.project_id:
            projects = await list_projects(self.db.conn)
            if not projects:
                project = await create_project(
                    self.db.conn, "Default Project", self.config.chat_model
                )
                thread: Thread = await create_thread(self.db.conn, project.id)
                await sidebar.add_project(project, threads=[thread])
                await self._activate_thread(project, thread)
            else:
                p = projects[0]
                threads = await list_threads(self.db.conn, p.id)
                if not threads:
                    threads = [await create_thread(self.db.conn, p.id)]
                    await sidebar.add_thread(p, threads[0])
                await self._activate_thread(p, threads[0])
