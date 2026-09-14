import aiosqlite
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage

from fp_int_agents.config import LlmConfig, Project
from fp_int_agents.llm.client import get_chat_model

_SYSTEM_PROMPT = (
    "You are a memory assistant. Given a conversation history, produce a concise summary "
    "of the key facts, decisions, and topics discussed. Be brief and factual."
)

_SUMMARIZE_PROMPT = "Summarize the conversation above into a short paragraph."


async def memorize_thread(
    project: Project,
    messages: list[AnyMessage],
    llm_config: LlmConfig,
    conn: aiosqlite.Connection | None,
    memory_message_count: int,
    thread_id: str,
) -> tuple[str, int]:
    model = get_chat_model(
        base_url=llm_config.base_url,
        model=project.chat_model,
        api_key=llm_config.api_key,
    )
    response = await model.ainvoke(
        [SystemMessage(content=_SYSTEM_PROMPT)]
        + messages
        + [HumanMessage(content=_SUMMARIZE_PROMPT)]
    )
    return str(response.content), memory_message_count
