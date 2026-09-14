#!/usr/bin/env bash
set -e

# Change to project root if running from within services/model_server
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

trap 'kill $(jobs -p) 2>/dev/null || true' EXIT INT TERM

echo "==> Starting MLX Model Server on :8000..."
uv run --project services/model_server python -m mlx_lm.server \
  --model mlx-community/Qwen2.5-3B-Instruct-bf16 \
  --adapter-path scripts/kleine_anfragen/adapters_lr_e_-4 \
  --port 8000 &

echo "==> Starting BGE Reranker Service on :8001..."
uv run --project services/model_server python services/model_server/serve_reranker.py \
  --port 8001 &

echo "==> Starting LiteLLM Proxy on :6655..."
uv run --project services/model_server litellm \
  --config services/model_server/litellm_config.yaml \
  --port 6655 &

echo "==> All model services started. Press Ctrl+C to terminate."
wait
