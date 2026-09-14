# FP Intelligent Agents

A Textual TUI chat app with LangGraph, hybrid Qdrant retrieval, and local Apple Silicon model serving (MLX + BGE Reranker + LiteLLM).

## Running the Project

### 1. Unified Launch (Recommended)
Starts all background model servers, waits for health checks, launches the TUI, and shuts down servers on exit:
```bash
uv run python launch.py
```

### 2. Run Components Individually

- **Model Servers (MLX + Reranker + LiteLLM):**
  ```bash
  # Uses ports: MLX=8000, Reranker=8001, LiteLLM=4000 (or custom LITELLM_PORT)
  bash services/model_server/serve.sh
  ```
  *Or run them separately with custom ports:*
  - **MLX Chat Model:** `uv run --project services/model_server python -m mlx_lm.server --model mlx-community/Qwen2.5-3B-Instruct-bf16 --adapter-path scripts/kleine_anfragen/adapters_lr_e_-4 --port 8000`
  - **BGE Reranker:** `uv run --project services/model_server python services/model_server/serve_reranker.py --port 8001`
  - **LiteLLM Proxy:** `uv run --project services/model_server litellm --config services/model_server/litellm_config.yaml --port <PORT>`

- **Chat App Only (connects to whatever `llm.base_url` is configured in `config.toml`):**
  ```bash
  uv run --package fp-int-agents fp-int-agents
  ```

- **Testing:**
  ```bash
  uv run pytest -m "not network"   # Unit tests (offline)
  uv run pytest                    # Full suite (requires live model endpoints)
  ```

---

## Configuration

The configuration is cleanly split into **Service Endpoints (Root)** and **Application Defaults (App)**:

### 1. Service & Endpoint Configuration (`config.toml` at root)
Defines where models and external services are hosted (`cp config.toml.example config.toml`):
- `llm.base_url`: Target LLM endpoint (e.g. `"http://localhost:4000/v1"`, your existing proxy, or remote API).
- `llm.api_key`: Authentication key for the endpoint (`"none"` for local proxies).
- `reranker.url`: Cross-encoder reranker endpoint (default: `"http://127.0.0.1:8001/v1/rerank"`).

*Note: `launch.py` automatically reads this file to determine ports and health check targets.*

### 2. App Defaults (`apps/chat/config.toml`)
Defines chat application behavior (`cp apps/chat/config.toml.example apps/chat/config.toml`):
- `defaults.chat_model`: Active model name (e.g. `"qwen-kleine-anfragen"`).
- `defaults.embedding_model`: Text embedding model.
- `defaults.agent`: Default agent type (`"simple"` or `"rag"`).
- `defaults.db_path`: SQLite database path (default: `"data/app.db"`).

### 2. Optional Local LiteLLM Proxy (`services/model_server/litellm_config.yaml`)
If you choose to run the included local proxy:
- Default port is set via `$LITELLM_PORT` (defaults to `4000`).
- Routes `qwen-kleine-anfragen` to local `mlx_lm.server` on `:8000`.
- Catch-all `*` can forward unmapped models to an upstream proxy via `UPSTREAM_LITELLM_URL` and `UPSTREAM_LITELLM_KEY`.

---

## Documentation
See architecture notes and diagrams in the [`docs/`](docs/) directory.
