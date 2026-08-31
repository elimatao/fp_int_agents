import asyncio
from typing import TYPE_CHECKING, cast

from langchain_core.messages import AIMessage, HumanMessage
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widget import Widget

from fp_int_agents.agents.agent_caller import call_agent
from fp_int_agents.config import QueryConfig
from fp_int_agents.storage.db import get_project

from .input_bar import InputBar
from .message_bubble import MessageBubble

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
    def _fp_app(self) -> "FPIntAgentsApp":
        from fp_int_agents.app import FPIntAgentsApp

        return cast(FPIntAgentsApp, self.app)

    def compose(self) -> ComposeResult:
        with VerticalScroll() as scroll:
            self._scroll = scroll
        yield InputBar()

    async def on_mount(self) -> None:
        snapshot = await self._fp_app.db.checkpointer.aget(
            self._query_config.to_runnable_config()
        )
        if snapshot:
            for msg in snapshot["channel_values"].get("messages", []):
                if isinstance(msg, (HumanMessage, AIMessage)):
                    await self._append(msg)

    def on_unmount(self) -> None:
        if self._agent_task and not self._agent_task.done():
            self._agent_task.cancel()

    async def on_input_bar_submitted(self, event: InputBar.Submitted) -> None:
        project = await get_project(self._fp_app.db.conn, self._query_config.project_id)
        if project is None:
            return
        await self._append(HumanMessage(content=event.text))
        ai_bubble = MessageBubble(AIMessage(content=""))
        await self._scroll.mount(ai_bubble)
        self._scroll.scroll_end(animate=False)
        self._agent_task = asyncio.create_task(
            self._stream_response(ai_bubble, event.text, project)
        )

    async def _stream_response(
        self, ai_bubble: "MessageBubble", text: str, project
    ) -> None:
        try:
            async for token in call_agent(
                project=project,
                query_config=self._query_config,
                message=HumanMessage(content=text),
                checkpointer=self._fp_app.db.checkpointer,
            ):
                ai_bubble.append_token(token)
                if ai_bubble.virtual_region.y >= self._scroll.scroll_offset.y:
                    self._scroll.scroll_end(animate=False)
        except asyncio.CancelledError:
            pass

    async def _append(self, message: HumanMessage | AIMessage) -> None:
        bubble = MessageBubble(message)
        await self._scroll.mount(bubble)
        self._scroll.scroll_end(animate=False)
