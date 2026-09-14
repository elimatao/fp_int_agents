import os
import signal
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import NoReturn
from urllib.parse import urlparse

import httpx

SERVE_SCRIPT = "services/model_server/serve.sh"


def get_config_endpoints() -> tuple[str, str, str, str, str | None, int]:
    """Read LLM base URL and Reranker URL from config.toml, or use defaults."""
    config_file = Path("config.toml")
    base_url = "http://localhost:4000/v1"
    api_key = "none"
    reranker_url = "http://127.0.0.1:8001/v1/rerank"
    embeddings_url = "http://127.0.0.1:8002/v1/embeddings"
    hf_key: str | None = None

    if config_file.exists():
        try:
            with config_file.open("rb") as f:
                cfg = tomllib.load(f)
                base_url = cfg.get("llm", {}).get("base_url", base_url)
                api_key = cfg.get("llm", {}).get("api_key", api_key)
                if "reranker_url" in cfg.get("llm", {}):
                    reranker_url = cfg["llm"]["reranker_url"]
                embeddings_sec = cfg.get("embeddings", {})
                if isinstance(embeddings_sec, dict) and "url" in embeddings_sec:
                    embeddings_url = embeddings_sec["url"]
                hf_key = cfg.get("huggingface", {}).get("hf_key")
        except Exception:
            pass

    parsed = urlparse(base_url)
    proxy_port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return base_url, api_key, reranker_url, embeddings_url, hf_key, proxy_port


def wait_for_services(
    base_url: str,
    api_key: str,
    reranker_url: str,
    embeddings_url: str,
    timeout_seconds: int = 40,
) -> bool:
    print("Waiting for model endpoints to become ready...")
    start_time = time.time()

    # Generic check endpoints for any OpenAI-compatible server + reranker
    models_url = base_url.rstrip("/") + "/models"
    reranker_health = reranker_url.rsplit("/v1", 1)[0].rstrip("/") + "/health"
    embeddings_health = embeddings_url.rsplit("/v1", 1)[0].rstrip("/") + "/health"

    check_targets = [
        (reranker_health, {"headers": {}}, "Reranker"),
        (embeddings_health, {"headers": {}}, "Embeddings"),
        (
            models_url,
            {"headers": {"Authorization": f"Bearer {api_key}"}},
            f"LLM Endpoint ({base_url})",
        ),
    ]

    ready: dict[str, bool] = {name: False for _, _, name in check_targets}

    while time.time() - start_time < timeout_seconds:
        for url, kwargs, name in check_targets:
            if not ready[name]:
                try:
                    resp = httpx.get(url, timeout=1.0, **kwargs)
                    if resp.status_code in (200, 401, 403):
                        ready[name] = True
                        print(f"  [+] {name} is online.")
                except Exception:
                    pass

        if all(ready.values()):
            print("All services are ready!")
            return True
        time.sleep(1)

    print("Warning: Timed out waiting for all services. Starting app anyway...")
    return False


def main() -> NoReturn:
    server_process: subprocess.Popen | None = None

    def _cleanup(_signum=None, _frame=None) -> None:
        if server_process and server_process.poll() is None:
            print("\nShutting down model servers...")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    try:
        base_url, api_key, reranker_url, embeddings_url, hf_key, proxy_port = (
            get_config_endpoints()
        )

        # START_MODELS=0 skips local background model serving (useful if using an existing external proxy)
        start_models = os.getenv("START_MODELS", "1").lower() not in (
            "0",
            "false",
            "no",
        )

        if start_models and os.path.exists(SERVE_SCRIPT):
            print(
                f"Starting model servers via {SERVE_SCRIPT} (LiteLLM port {proxy_port})..."
            )
            env = {**os.environ, "LITELLM_PORT": str(proxy_port)}
            if hf_key:
                env["HF_TOKEN"] = hf_key
            server_process = subprocess.Popen(["bash", SERVE_SCRIPT], env=env)
            wait_for_services(base_url, api_key, reranker_url, embeddings_url)

        print("Launching Textual Chat App...")
        exit_code = subprocess.call(
            ["uv", "run", "--package", "fp-int-agents", "fp-int-agents"]
        )
        sys.exit(exit_code)
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
