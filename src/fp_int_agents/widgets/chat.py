import asyncio
from typing import TYPE_CHECKING, cast

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widget import Widget

from fp_int_agents.agents.agent_caller import (
    TextToken,
    ToolCallFinished,
    ToolCallStarted,
    call_agent,
    preview_result,
)
from fp_int_agents.config import QueryConfig
from fp_int_agents.storage.db import get_project, update_thread_tools

from .input_bar import InputBar
from .message_bubble import MessageBubble
from .tool_call_bubble import ToolCallBubble
from .tool_selector import ToolSelector

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
        yield InputBar(self._query_config.active_tools)

    async def on_mount(self) -> None:
        snapshot = await self._fp_app.db.checkpointer.aget(
            self._query_config.to_runnable_config()
        )
        if not snapshot:
            return
        messages = snapshot["channel_values"].get("messages", [])
        results = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}
        for msg in messages:
            if isinstance(msg, HumanMessage):
                await self._append(msg)
            elif isinstance(msg, AIMessage):
                if isinstance(msg.content, str) and msg.content:
                    await self._append(msg)
                for tc in msg.tool_calls:
                    await self._append_tool_call(tc, results.get(tc["id"]))

    async def _append_tool_call(
        self, tool_call: dict, result: ToolMessage | None
    ) -> None:
        bubble = ToolCallBubble(tool_call["name"], tool_call.get("args", {}))
        await self._scroll.mount(bubble)
        if result is not None:
            bubble.set_result(preview_result(result.content))
        self._scroll.scroll_end(animate=False)

    def on_unmount(self) -> None:
        if self._agent_task and not self._agent_task.done():
            self._agent_task.cancel()

    async def on_input_bar_tools_requested(
        self, event: InputBar.ToolsRequested
    ) -> None:
        async def _on_dismiss(tools: list[str] | None) -> None:
            if tools is None:
                return
            self._query_config = self._query_config.model_copy(
                update={"active_tools": tools}
            )
            await update_thread_tools(
                self._fp_app.db.conn, self._query_config.thread_id, tools
            )
            self.query_one(InputBar).update_tools(tools)

        await self.app.push_screen(
            ToolSelector(event.active_tools, event.all_tools), _on_dismiss
        )

    async def on_input_bar_submitted(self, event: InputBar.Submitted) -> None:
        project = await get_project(self._fp_app.db.conn, self._query_config.project_id)
        if project is None:
            return
        await self._append(HumanMessage(content=event.text))
        self._agent_task = asyncio.create_task(
            self._stream_response(event.text, project)
        )

    async def _stream_response(self, text: str, project) -> None:
        text_bubble: MessageBubble | None = None
        tool_bubbles: dict[str, ToolCallBubble] = {}
        try:
            async for ev in call_agent(
                project=project,
                query_config=self._query_config,
                message=HumanMessage(content=text),
                checkpointer=self._fp_app.db.checkpointer,
                llm_config=self._fp_app.config.llm,
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
                self._scroll.scroll_end(animate=False)
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # noqa: BLE001 - top-level guard so agent errors surface instead of crashing the task
            self.notify(str(exc), title="Agent error", severity="error", timeout=10)

    async def _append(self, message: HumanMessage | AIMessage) -> None:
        bubble = MessageBubble(message)
        await self._scroll.mount(bubble)
        self._scroll.scroll_end(animate=False)
