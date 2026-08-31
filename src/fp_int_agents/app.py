from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Footer, Header

from .config import AppConfig, QueryConfig, Thread
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
        self._active_thread_id: str | None = None
        self._active_project_id: str | None = None
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

        data: list[tuple] = []
        for p in projects:
            threads = await list_threads(self.db.conn, p.id)
            if not threads:
                t: Thread = await create_thread(
                    self.db.conn, p.id, "Default Thread Title"
                )
                threads = [t]
            data.append((p, threads))

        sidebar = self.query_one(ProjectSidebar)
        await sidebar.populate(data)

        first_project, first_threads = data[0]
        first_thread = first_threads[0]
        sidebar.set_active_thread(first_project, first_thread)
        query_config = QueryConfig(
            thread_id=first_thread.id,
            project_id=first_project.id,
            chat_model=first_project.chat_model,
        )
        self._active_thread_id = first_thread.id
        self._active_project_id = first_project.id
        await self.query_one("#main-layout", Horizontal).mount(Chat(query_config))

    async def _swap_chat(self, query_config: QueryConfig) -> None:
        self._active_thread_id = query_config.thread_id
        self._active_project_id = query_config.project_id
        layout = self.query_one("#main-layout", Horizontal)
        await layout.query_one(Chat).remove()
        await layout.mount(Chat(query_config))

    async def on_project_sidebar_thread_selected(
        self, event: ProjectSidebar.ThreadSelected
    ) -> None:
        self.query_one(ProjectSidebar).set_active_thread(event.project, event.thread)
        await self._swap_chat(
            QueryConfig(
                thread_id=event.thread.id,
                project_id=event.project.id,
                chat_model=event.project.chat_model,
            )
        )

    async def on_project_sidebar_new_project(
        self, _: ProjectSidebar.NewProject
    ) -> None:
        project = await create_project(
            self.db.conn, "New Project", self.config.chat_model
        )
        thread: Thread = await create_thread(self.db.conn, project.id)
        sidebar = self.query_one(ProjectSidebar)
        await sidebar.add_project(project, threads=[thread])
        sidebar.set_active_thread(project, thread)
        await self._swap_chat(
            QueryConfig(
                thread_id=thread.id,
                project_id=project.id,
                chat_model=project.chat_model,
            )
        )

    async def on_project_sidebar_new_thread(
        self, event: ProjectSidebar.NewThread
    ) -> None:
        thread: Thread = await create_thread(self.db.conn, event.project.id)
        sidebar = self.query_one(ProjectSidebar)
        await sidebar.add_thread(event.project, thread)
        sidebar.set_active_thread(event.project, thread)
        await self._swap_chat(
            QueryConfig(
                thread_id=thread.id,
                project_id=event.project.id,
                chat_model=event.project.chat_model,
            )
        )

    async def on_project_sidebar_delete_thread(
        self, event: ProjectSidebar.DeleteThread
    ) -> None:
        sidebar = self.query_one(ProjectSidebar)
        await delete_thread(self.db.conn, self.db.checkpointer, event.thread.id)
        await sidebar.remove_thread(event.project.id, event.thread.id)
        if event.thread.id == self._active_thread_id:
            # Switch to first remaining thread in the same project, or first of any project.
            threads = await list_threads(self.db.conn, event.project.id)
            if threads:
                sidebar.set_active_thread(event.project, threads[0])
                await self._swap_chat(
                    QueryConfig(
                        thread_id=threads[0].id,
                        project_id=event.project.id,
                        chat_model=event.project.chat_model,
                    )
                )
            else:
                # No threads left — create a fresh one.
                new_thread: Thread = await create_thread(self.db.conn, event.project.id)
                await sidebar.add_thread(event.project, new_thread)
                sidebar.set_active_thread(event.project, new_thread)
                await self._swap_chat(
                    QueryConfig(
                        thread_id=new_thread.id,
                        project_id=event.project.id,
                        chat_model=event.project.chat_model,
                    )
                )

    async def on_project_sidebar_delete_project(
        self, event: ProjectSidebar.DeleteProject
    ) -> None:
        thread_ids = [t.id for t in await list_threads(self.db.conn, event.project.id)]
        await delete_project(self.db.conn, self.db.checkpointer, event.project.id, thread_ids)
        sidebar = self.query_one(ProjectSidebar)
        await sidebar.remove_project(event.project.id)
        if event.project.id == self._active_project_id:
            # Switch to first remaining project, creating one if none left.
            projects = await list_projects(self.db.conn)
            if not projects:
                project = await create_project(
                    self.db.conn, "Default Project", self.config.chat_model
                )
                thread: Thread = await create_thread(self.db.conn, project.id)
                await sidebar.add_project(project, threads=[thread])
                sidebar.set_active_thread(project, thread)
                await self._swap_chat(
                    QueryConfig(
                        thread_id=thread.id,
                        project_id=project.id,
                        chat_model=project.chat_model,
                    )
                )
            else:
                p = projects[0]
                threads = await list_threads(self.db.conn, p.id)
                if not threads:
                    threads = [await create_thread(self.db.conn, p.id)]
                    await sidebar.add_thread(p, threads[0])
                sidebar.set_active_thread(p, threads[0])
                await self._swap_chat(
                    QueryConfig(
                        thread_id=threads[0].id,
                        project_id=p.id,
                        chat_model=p.chat_model,
                    )
                )
