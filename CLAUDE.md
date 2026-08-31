# fp-int-agents

A Textual TUI chat app with LangGraph + Ollama + litellm, bu

## Conversations
- Be very concise

## Code Style
- Always use type hints
- Always check context7 for up-to-date docs before using any library/framework
- run ruff after finishing a task
- Whenever you plan a new feature, write a test for it first.

## Stack
- TUI: Textual
- Conversation Graph: LangGraph
- LLM client: litellm, connects with local models via ollama, but also has support for remote models
- Checkpointer: AsyncSqliteSaver (for now)
- Deployment: Docker Compose

## Model Selection
- Global default model set via config file
- Per-conversation override stored in LangGraph thread metadata
- Mid-conversation switches persist and take effect on the next message