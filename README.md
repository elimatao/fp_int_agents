# FP Intelligent Agents

A Textual TUI chat app with LangGraph, hybrid Qdrant retrieval, and local Apple Silicon model serving (MLX + BGE Reranker + LiteLLM).

## Prerequisites

- **Apple Silicon Mac** (M1 or later) — MLX model serving requires it
- **Python 3.13+** — install via [pyenv](https://github.com/pyenv/pyenv) or the [official installer](https://www.python.org/downloads/)
- **uv** — install with `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Setup

1. Clone the repo and enter it.

2. **Optional:** to set a HuggingFace token (recommended to avoid download rate limits), create `config.toml` from the template:
   ```bash
   cp config.toml.example config.toml
   # then uncomment and set huggingface.hf_key
   ```
   Get a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Without it, all defaults still work — model downloads may just be slower or rate-limited.

3. That's it — `uv` manages all Python environments automatically.

> **First run**: ~3.5 GB of models will be downloaded from HuggingFace Hub (`Qwen2.5-3B-Instruct-4bit` and `multilingual-e5-small`). This can take 15–30 minutes depending on your connection. Progress is logged to `data/logs/`.

## Quickstart

```bash
uv run python launch.py
```

Starts all background model servers, waits for health checks (up to 2 minutes), launches the TUI, and shuts down servers on exit.

To skip local model serving and connect to an existing external proxy:
```bash
START_MODELS=0 uv run python launch.py
```

## Running Components Individually

- **Model services** (MLX fine-tuned & base, Reranker, Embeddings, LiteLLM proxy):
  ```bash
  bash services/model_server/serve.sh
  ```
  Default ports: MLX fine-tuned `8000`, MLX base `8003`, Reranker `8001`, Embeddings `8002`, LiteLLM proxy `4000`.
  Override any port via environment variables: `MLX_PORT`, `RERANKER_PORT`, `EMBEDDINGS_PORT`, `EMBEDDINGS_PORT`, `LITELLM_PORT`, `BASE_PORT`.

- **Chat app only** (connects to configured `llm.base_url`):
  ```bash
  uv run --package fp-int-agents fp-int-agents
  ```

- **Tests:**
  ```bash
  uv run pytest apps/chat/tests/ -m "not network"   # Offline unit tests

  bash services/model_server/serve.sh
  uv run pytest apps/chat/tests/                    # Full suite (requires live endpoints)
  ```

---

## Configuration

Configuration is split between endpoint connections and application defaults:

- **Root endpoints (`config.toml`, copy from `config.toml.example`):**
  - `llm.base_url`: LLM proxy URL (default: `"http://localhost:4000/v1"`)
  - `llm.api_key`: API key (`"none"` for local proxy)
  - `llm.reranker_url`: Cross-encoder reranker endpoint (default: `"http://127.0.0.1:8001/v1/rerank"`)
  - `huggingface.hf_key`: HF token for model downloads (recommended)

- **App defaults (`apps/chat/config.toml`):**
  - `defaults.chat_model`: Active model (e.g. `"qwen-kleine-anfragen"`)
  - `defaults.embedding_model`: Embedding model (e.g. `"multilingual-e5-small"`)
  - `defaults.agent`: Default agent (`"simple"` or `"BundesRAG"`)
  - `defaults.mem_agent`: Memory agent type (`"simple"` or `"BundesRAG"`)
  - `defaults.db_path`: SQLite database path (default: `"data/app.db"`)

---

## Repository Structure

```
fp_int_agents/
├── apps/
│   └── chat/                   # Textual TUI chat application (uv workspace member)
│       ├── src/fp_int_agents/  # App source: TUI, LangGraph agents, RAG, config
│       ├── tests/              # Unit and integration tests
│       └── config.toml         # App-level defaults (committed)
│
├── services/
│   └── model_server/           # Separate uv environment for model serving
│       ├── serve.sh            # Starts all model services in background
│       ├── serve_embeddings.py # FastAPI server: multilingual-e5-small
│       ├── serve_reranker.py   # FastAPI server: BGE reranker (fine-tuned)
│       ├── litellm_config.yaml # LiteLLM proxy model routing config
│       └── pyproject.toml      # Dependencies: mlx-lm, torch, sentence-transformers
│
├── scripts/
│   └── kleine_anfragen/        # Fine-tuning scripts and trained adapter checkpoints
│       ├── adapters_lr_e_-4/   # LoRA adapter for Qwen (conversation model)
│       └── adapters_reranker/  # Fine-tuned BGE reranker weights
│
├── docs/                       # Architecture diagrams, finetuning notes
├── data/                       # Runtime data: SQLite DB, Qdrant vectors, logs (gitignored)
├── launch.py                   # Orchestrator: starts servers + TUI, handles shutdown
├── config.toml.example         # Root config template — copy to config.toml
└── pyproject.toml              # Root uv workspace definition
```

---

## Further Documentation

See finetuning information and the RAG agent diagram in the [`docs/`](docs/) directory.
