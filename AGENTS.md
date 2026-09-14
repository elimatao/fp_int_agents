# fp-int-agents

A Textual TUI chat app with uv + LangGraph + litellm.

## Conversations
- Be very concise

## Code Style
- Always use type hints
- Always check context7 for up-to-date docs before using any library/framework
- Run ruff after finishing a task
- Whenever you plan a new feature, write a test for it first.

## Stack
- Runtime: uv
- TUI: Textual
- Conversation Graph: LangGraph
- LLM client: litellm, connects with local models via custom servers such as mlx-lm over a litellm proxy, but also has support for remote models (any openai-compatible, also ollama)
- Checkpointer: AsyncSqliteSaver (for now)
- Deployment: Docker Compose

## Model Selection
- Global default model set via config file
- Per-conversation override stored in LangGraph thread metadata
- Mid-conversation switches persist and take effect on the next message

