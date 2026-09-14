#!/usr/bin/env bash
set -e

# Change to project root if running from within services/model_server
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

trap 'kill $(jobs -p) 2>/dev/null || true' EXIT INT TERM

LOG_DIR="${LOG_DIR:-data/logs}"
mkdir -p "$LOG_DIR"

MLX_PORT="${MLX_PORT:-8000}"
RERANKER_PORT="${RERANKER_PORT:-8001}"
EMBEDDINGS_PORT="${EMBEDDINGS_PORT:-8002}"
LITELLM_PORT="${LITELLM_PORT:-4000}"

echo "==> Starting MLX Model Server on :${MLX_PORT}..."
uv run --project services/model_server python -m mlx_lm.server \
  --model mlx-community/Qwen2.5-3B-Instruct-bf16 \
  --adapter-path scripts/kleine_anfragen/adapters_lr_e_-4 \
  --port "$MLX_PORT" >"$LOG_DIR/mlx.log" 2>&1 &

echo "==> Starting BGE Reranker Service on :${RERANKER_PORT}..."
uv run --project services/model_server python services/model_server/serve_reranker.py \
  --port "$RERANKER_PORT" >"$LOG_DIR/reranker.log" 2>&1 &

echo "==> Starting Embeddings Service on :${EMBEDDINGS_PORT}..."
uv run --project services/model_server python services/model_server/serve_embeddings.py \
  --port "$EMBEDDINGS_PORT" >"$LOG_DIR/embeddings.log" 2>&1 &

echo "==> Starting LiteLLM Proxy on :${LITELLM_PORT}..."
uv run --project services/model_server litellm \
  --config services/model_server/litellm_config.yaml \
  --port "$LITELLM_PORT" >"$LOG_DIR/litellm.log" 2>&1 &

echo "==> All model services started. Press Ctrl+C to terminate."
wait
