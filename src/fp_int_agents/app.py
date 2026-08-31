from textual.app import App, ComposeResult
from textual.widgets import Footer, Header

from .config import AppConfig, QueryConfig, Thread
from .constants import APP_NAME
from .storage.db import DB, create_project, create_thread
from .widgets.chat import Chat


class FPIntAgentsApp(App):
    TITLE = APP_NAME

    def __init__(self, db: DB, config: AppConfig) -> None:
        self.db = db
        self.config = config
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()

    async def on_mount(self) -> None:
        self.theme = "tokyo-night"
        project = await create_project(self.db.conn, "Test Project", self.config.chat_model)
        thread: Thread = await create_thread(self.db.conn, project.id)
        query_config = QueryConfig(
            thread_id=thread.id,
            project_id=project.id,
            chat_model=self.config.chat_model,
        )
        await self.mount(Chat(query_config), before=self.query_one(Footer))
