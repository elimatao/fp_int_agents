import asyncio
from typing import TYPE_CHECKING, cast

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widget import Widget

from fp_int_agents.agents.agent_caller import (
    RagStepFinished,
    RagStepStarted,
    RerankerFinished,
    TextToken,
    ToolCallFinished,
    ToolCallStarted,
    call_agent,
    preview_result,
)
from fp_int_agents.config import QueryConfig
from fp_int_agents.storage.db import get_project, get_thread, update_thread_settings

from .input_bar import InputBar
from .message_bubble import MessageBubble
from .rag_log_bubble import RagLogBubble
from .settings_modal import SettingsModal, SettingsResult
from .summary_bubble import SummaryBubble
from .tool_call_bubble import ToolCallBubble

if TYPE_CHECKING:
    from fp_int_agents.app import FPIntAgentsApp


class Chat(Widget):
    """Display a multi-turn conversation from a LangGraph thread."""

    DEFAULT_CSS = """
    Chat {
        width: 100%;
        height: 1fr;
        layout: vertical;
    }
    Chat > VerticalScroll {
        height: 1fr;
    }
    """

    def __init__(self, query_config: QueryConfig) -> None:
        super().__init__()
        self._query_config = query_config
        self._agent_task: asyncio.Task | None = None

    @property
    def query_config(self) -> QueryConfig:
        return self._query_config

    @property
    def _fp_app(self) -> "FPIntAgentsApp":
        from fp_int_agents.app import FPIntAgentsApp

        return cast(FPIntAgentsApp, self.app)

    def compose(self) -> ComposeResult:
        with VerticalScroll() as scroll:
            self._scroll = scroll
        yield InputBar(self._query_config.active_tools, self._query_config.chat_model)

    async def on_mount(self) -> None:
        thread, snapshot = await asyncio.gather(
            get_thread(self._fp_app.db.conn, self._query_config.thread_id),
            self._fp_app.db.checkpointer.aget(self._query_config.to_runnable_config()),
        )
        if not snapshot:
            return
        messages = snapshot["channel_values"].get("messages", [])
        results = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}
        for msg in messages:
            if isinstance(msg, HumanMessage):
                await self._mount_message_bubble(msg)
            elif isinstance(msg, AIMessage):
                if isinstance(msg.content, str) and msg.content:
                    await self._mount_message_bubble(msg)
                for tc in msg.tool_calls:
                    await self._mount_tool_call_bubble(tc, results.get(tc["id"]))
        if thread and thread.summary:
            await self._mount_summary_bubble(thread.summary)

    def on_unmount(self) -> None:
        if self._agent_task and not self._agent_task.done():
            self._agent_task.cancel()

    # --- Event handlers ---

    async def on_input_bar_submitted(self, event: InputBar.Submitted) -> None:
        await self._mount_message_bubble(HumanMessage(content=event.text))
        project = await get_project(self._fp_app.db.conn, self._query_config.project_id)
        if project is None:
            return
        self._agent_task = asyncio.create_task(
            self._stream_agent_response(event.text, project)
        )

    async def on_input_bar_settings_requested(
        self, event: InputBar.SettingsRequested
    ) -> None:
        available_models = [event.current_model]
        try:
            from fp_int_agents.llm.models import list_models

            infos = await list_models(
                self._fp_app.config.llm.base_url, self._fp_app.config.llm.api_key
            )
            fetched = [m.id for m in infos]
            if fetched:
                available_models = fetched
                if event.current_model not in available_models:
                    available_models.insert(0, event.current_model)
        except Exception as exc:  # noqa: BLE001 - model list fetch is best-effort; fallback to current model
            self.log.warning(f"Could not fetch model list: {exc}")

        async def _on_dismiss(result: SettingsResult | None) -> None:
            if result is None:
                return
            model = result.chat_model or self._query_config.chat_model
            self._query_config = self._query_config.model_copy(
                update={"active_tools": result.active_tools, "chat_model": model}
            )
            await update_thread_settings(
                self._fp_app.db.conn,
                self._query_config.thread_id,
                result.active_tools,
                result.chat_model,
            )
            self.query_one(InputBar).update_settings(result.active_tools, model)

        await self.app.push_screen(
            SettingsModal(
                event.current_model,
                available_models,
                event.active_tools,
                event.all_tools,
            ),
            _on_dismiss,
        )

    async def _stream_agent_response(self, text: str, project) -> None:
        text_bubble: MessageBubble | None = None
        tool_bubbles: dict[str, ToolCallBubble] = {}
        rag_bubbles: dict[str, RagLogBubble] = {}
        try:
            async for ev in call_agent(
                project=project,
                query_config=self._query_config,
                message=HumanMessage(content=text),
                checkpointer=self._fp_app.db.checkpointer,
                llm_config=self._fp_app.config.llm,
                conn=self._fp_app.db.conn,
            ):
                if isinstance(ev, TextToken):
                    if text_bubble is None:
                        text_bubble = MessageBubble(AIMessage(content=""))
                        await self._scroll.mount(text_bubble)
                    text_bubble.append_token(ev.text)
                elif isinstance(ev, ToolCallStarted):
                    bubble = ToolCallBubble(ev.name, ev.args)
                    tool_bubbles[ev.run_id] = bubble
                    await self._scroll.mount(bubble)
                    text_bubble = None
                elif isinstance(ev, ToolCallFinished):
                    bubble = tool_bubbles.get(ev.run_id)
                    if bubble is not None:
                        bubble.set_result(ev.result)
                elif isinstance(ev, RagStepStarted):
                    rag_bubble = RagLogBubble(ev.node)
                    rag_bubbles[ev.node] = rag_bubble
                    await self._scroll.mount(rag_bubble)
                    text_bubble = None
                elif isinstance(ev, RagStepFinished):
                    rag_bubble = rag_bubbles.get(ev.node)
                    if rag_bubble is not None:
                        rag_bubble.set_result(ev.result)
                elif isinstance(ev, RerankerFinished):
                    rag_bubble = RagLogBubble("reranker")
                    await self._scroll.mount(rag_bubble)
                    rag_bubble.set_result("\n".join(ev.docs) if ev.docs else "no documents")
                self._scroll.scroll_end(animate=False)
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001 - top-level guard so agent errors surface instead of crashing the task
            self.notify(str(exc), title="Agent error", severity="error", timeout=10)

    async def _mount_message_bubble(self, message: HumanMessage | AIMessage) -> None:
        await self._scroll.mount(MessageBubble(message))
        self._scroll.scroll_end(animate=False)

    async def _mount_tool_call_bubble(
        self, tool_call: dict, result: ToolMessage | None
    ) -> None:
        bubble = ToolCallBubble(tool_call["name"], tool_call.get("args", {}))
        await self._scroll.mount(bubble)
        if result is not None:
            bubble.set_result(preview_result(result.content))
        self._scroll.scroll_end(animate=False)

    async def _mount_summary_bubble(self, summary: str) -> None:
        for existing in self._scroll.query(SummaryBubble):
            await existing.remove()
        await self._scroll.mount(SummaryBubble(summary))
        self._scroll.scroll_end(animate=False)
