from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Footer, Header

from .config import AppConfig, QueryConfig, Thread
from .constants import APP_NAME
from .storage.db import DB, create_project, create_thread, list_projects, list_threads
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
        await self.query_one("#main-layout", Horizontal).mount(Chat(query_config))

    async def _swap_chat(self, query_config: QueryConfig) -> None:
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
