# FP Intelligent Agents

A Textual TUI chat app with LangGraph, hybrid Qdrant retrieval, and local Apple Silicon model serving (MLX + BGE Reranker + LiteLLM).

## Quickstart
Optionally, set your huggingface API key (copy the root `config.toml.example` to `config.toml` for that and edit the latter one).
```bash
uv run python launch.py
```
Starts all background model servers, waits for health checks, launches the TUI, and shuts down servers on exit:


## Running Components Individually

- **Model Services (MLX fine-tuned & base, Reranker, Embeddings, LiteLLM proxy):**
  ```bash
  bash services/model_server/serve.sh
  ```
  *Default ports: MLX fine-tuned `8000`, MLX base `8003`, Reranker `8001`, Embeddings `8002`, LiteLLM proxy `4000`.*

- **Chat App Only** (connects to configured `llm.base_url`):
  ```bash
  uv run --package fp-int-agents fp-int-agents
  ```

- **Testing:**
  ```bash
  uv run pytest apps/chat/tests/ -m "not network"   # Offline unit tests
  
  bash services/model_server/serve.sh
  uv run pytest apps/chat/tests/                    # Full suite requires live model endpoints
  ```

---

## Configuration

Configuration is split between endpoint connections and application defaults:

- **Root Endpoints (`config.toml`):**
  - `llm.base_url`: LLM proxy or server URL (e.g. `"http://localhost:4000/v1"`).
  - `llm.api_key`: API key (`"none"` for local proxy).
  - `llm.reranker_url`: Cross-encoder reranker endpoint (default: `"http://127.0.0.1:8001/v1/rerank"`).
  - `huggingface.hf_key`: Optional HF token for model downloads.

- **App Defaults (`apps/chat/config.toml`):**
  - `defaults.chat_model`: Active model name (e.g. `"qwen-kleine-anfragen"`).
  - `defaults.embedding_model`: Embedding model name (e.g. `"multilingual-e5-small"`).
  - `defaults.agent`: Default agent (`"simple"` or `"BundesRAG"`).
  - `defaults.mem_agent`: Memory agent type (`"simple"` or `"BundesRAG"`).
  - `defaults.db_path`: SQLite database path (default: `"data/app.db"`).

---

## Further Documentation
See finetuning information and the RAG agent diagram in the [`docs/`](docs/) directory.
