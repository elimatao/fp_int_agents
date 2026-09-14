import asyncio
import pathlib

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Footer, Header

from .agents.agent_caller import summarize_thread
from .agents.registry import INGESTOR_AGENTS
from .config import AppConfig, Project, QueryConfig, Thread
from .constants import APP_NAME
from .storage.db import (
    DB,
    create_project,
    create_thread,
    delete_project,
    delete_thread,
    get_project,
    get_thread,
    list_projects,
    list_threads,
    update_thread_memory_count,
    update_thread_summary,
)
from .widgets.chat import Chat
from .widgets.ingest_modal import IngestModal
from .widgets.new_project_modal import NewProjectModal, NewProjectResult
from .widgets.project_sidebar import ProjectSidebar

_AGENT_BUNDLES: dict[str, dict[str, str]] = {
    "simple": {"mem_agent": "simple", "ingestor": "simple"},
    "BundesRAG": {"mem_agent": "BundesRAG", "ingestor": "BundesRAG"},
}


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

        if not data:
            return

        first_project, first_threads = data[0]
        first_thread = first_threads[0]
        query_config = QueryConfig(
            thread_id=first_thread.id,
            project_id=first_project.id,
            chat_model=first_thread.chat_model or first_project.chat_model,
            active_tools=first_thread.active_tools,
        )
        await self.query_one("#main-layout", Horizontal).mount(Chat(query_config))
        sidebar.set_active_thread(first_project.id, first_thread.id)

    async def _activate_thread(self, project: Project, thread: Thread) -> None:
        await self._swap_chat(
            QueryConfig(
                thread_id=thread.id,
                project_id=project.id,
                chat_model=thread.chat_model or project.chat_model,
                active_tools=thread.active_tools,
            )
        )
        self.query_one(ProjectSidebar).set_active_thread(project.id, thread.id)

    async def _activate_thread_by_id(self, project_id: str, thread_id: str) -> None:
        project = await get_project(self.db.conn, project_id)
        thread = await get_thread(self.db.conn, thread_id)
        if project and thread:
            await self._activate_thread(project, thread)

    async def action_quit(self) -> None:
        try:
            chat = self.query_one(Chat)
            await self._summarize_on_leave(chat.query_config)
        except Exception as exc:  # noqa: BLE001
            self.log.warning(f"Summarization on quit failed: {exc}")
        await super().action_quit()

    async def _swap_chat(self, query_config: QueryConfig) -> None:
        layout = self.query_one("#main-layout", Horizontal)
        outgoing = layout.query("Chat").first(Chat) if layout.query("Chat") else None
        if outgoing is not None:
            asyncio.create_task(self._summarize_on_leave(outgoing.query_config))
            await outgoing.remove()
        await layout.mount(Chat(query_config))

    async def _summarize_on_leave(self, query_config: QueryConfig) -> None:
        thread, snapshot = await asyncio.gather(
            get_thread(self.db.conn, query_config.thread_id),
            self.db.checkpointer.aget(query_config.to_runnable_config()),
        )
        if not snapshot:
            return
        messages = snapshot["channel_values"].get("messages", [])
        if not messages:
            return
        if thread and thread.summary_message_count == len(messages) and thread.memory_message_count == len(messages):
            return
        project = await get_project(self.db.conn, query_config.project_id)
        if project is None or not project.mem_agent:
            return
        try:
            summary, new_memory_count = await summarize_thread(
                project=project,
                thread_id=query_config.thread_id,
                messages=messages,
                llm_config=self.config.llm,
                conn=self.db.conn,
                memory_message_count=(thread.memory_message_count or 0) if thread else 0,
            )
            await update_thread_summary(
                self.db.conn, query_config.thread_id, summary, len(messages)
            )
            if new_memory_count != ((thread.memory_message_count or 0) if thread else 0):
                await update_thread_memory_count(
                    self.db.conn, query_config.thread_id, new_memory_count
                )
        except Exception as exc:  # noqa: BLE001 - summarization is best-effort
            self.log.warning(f"Thread summarization failed: {exc}")

    async def on_project_sidebar_thread_selected(
        self, event: ProjectSidebar.ThreadSelected
    ) -> None:
        await self._activate_thread_by_id(event.project_id, event.thread_id)

    async def on_project_sidebar_new_project(
        self, _: ProjectSidebar.NewProject
    ) -> None:
        def _done(result: NewProjectResult | None) -> None:
            if result is not None:
                self.run_worker(self._create_project(result), exclusive=False)

        self.push_screen(NewProjectModal(), _done)

    async def _create_project(self, result: NewProjectResult) -> None:
        bundle = _AGENT_BUNDLES.get(result.agent, _AGENT_BUNDLES["simple"])
        init_config = (
            {"embedding_model": self.config.embedding_model}
            if result.agent == "BundesRAG"
            else {}
        )
        project = await create_project(
            self.db.conn,
            result.name,
            self.config.chat_model,
            agent=result.agent,
            mem_agent=bundle["mem_agent"],
            ingestor=bundle["ingestor"],
            system_prompt=result.system_prompt or None,
            init_config=init_config,
        )
        thread: Thread = await create_thread(self.db.conn, project.id)
        sidebar = self.query_one(ProjectSidebar)
        await sidebar.add_project(project, threads=[thread])
        await self._activate_thread(project, thread)

    async def on_project_sidebar_ingest_document(
        self, event: ProjectSidebar.IngestDocument
    ) -> None:
        project = await get_project(self.db.conn, event.project_id)
        if project is None:
            return
        if not project.ingestor:
            self.notify("Ingestion is disabled for this project.", severity="warning")
            return

        def _done(path: str | None) -> None:
            if path:
                self.run_worker(self._run_ingest(project, path), exclusive=False)

        self.push_screen(IngestModal(), _done)

    _INGESTIBLE_SUFFIXES: frozenset[str] = frozenset(
        {".pdf", ".txt", ".md", ".py", ".rst", ".html", ".xml", ".json"}
    )

    async def _run_ingest(self, project: Project, path: str) -> None:
        assert project.ingestor is not None
        files = self._collect_ingest_paths(path)
        if not files:
            self.notify(f"No supported files found in {path}", severity="warning")
            return

        async def _ingest_one(p: pathlib.Path) -> None:
            try:
                text = await asyncio.to_thread(self._read_file, str(p))
                await INGESTOR_AGENTS[project.ingestor](  # type: ignore[index]
                    project, text, self.config.llm, self.db.conn, p.name
                )
                self.notify(f"Ingested {p.name}")
            except Exception as exc:  # noqa: BLE001
                self.log.warning(f"Ingest failed for {p}: {exc}")
                self.notify(f"Ingest failed: {p.name} — {exc}", severity="error")

        await asyncio.gather(*(_ingest_one(f) for f in files))

    @classmethod
    def _collect_ingest_paths(cls, path: str) -> list[pathlib.Path]:
        p = pathlib.Path(path)
        if p.is_dir():
            return sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in cls._INGESTIBLE_SUFFIXES)
        return [p]

    @staticmethod
    def _read_file(path: str) -> str:
        p = pathlib.Path(path)
        if p.suffix.lower() == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(p))
            return "\n\n".join((page.extract_text() or "") for page in reader.pages)
        return p.read_text(encoding="utf-8")

    async def on_project_sidebar_new_thread(
        self, event: ProjectSidebar.NewThread
    ) -> None:
        project = await get_project(self.db.conn, event.project_id)
        if project is None:
            return
        thread: Thread = await create_thread(self.db.conn, project.id)
        sidebar = self.query_one(ProjectSidebar)
        await sidebar.add_thread(project, thread)
        await self._activate_thread(project, thread)

    async def on_project_sidebar_delete_thread(
        self, event: ProjectSidebar.DeleteThread
    ) -> None:
        sidebar = self.query_one(ProjectSidebar)
        active_thread_id = self.query_one(Chat).query_config.thread_id
        await delete_thread(self.db.conn, self.db.checkpointer, event.thread_id)
        await sidebar.remove_thread(event.project_id, event.thread_id)
        if event.thread_id == active_thread_id:
            project = await get_project(self.db.conn, event.project_id)
            if project is None:
                return
            threads = await list_threads(self.db.conn, project.id)
            if threads:
                await self._activate_thread(project, threads[0])
            else:
                new_thread: Thread = await create_thread(self.db.conn, project.id)
                await sidebar.add_thread(project, new_thread)
                await self._activate_thread(project, new_thread)

    async def on_project_sidebar_delete_project(
        self, event: ProjectSidebar.DeleteProject
    ) -> None:
        thread_ids = [t.id for t in await list_threads(self.db.conn, event.project_id)]
        await delete_project(
            self.db.conn, self.db.checkpointer, event.project_id, thread_ids
        )
        sidebar = self.query_one(ProjectSidebar)
        await sidebar.remove_project(event.project_id)
        if event.project_id == self.query_one(Chat).query_config.project_id:
            projects = await list_projects(self.db.conn)
            if not projects:
                layout = self.query_one("#main-layout", Horizontal)
                await layout.query_one(Chat).remove()
            else:
                p = projects[0]
                threads = await list_threads(self.db.conn, p.id)
                if not threads:
                    threads = [await create_thread(self.db.conn, p.id)]
                    await sidebar.add_thread(p, threads[0])
                await self._activate_thread(p, threads[0])
