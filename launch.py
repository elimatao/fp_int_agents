import os
import signal
import subprocess
import sys
import time
from typing import NoReturn

import httpx

SERVE_SCRIPT = "services/model_server/serve.sh"
CHECK_URLS = [
    ("http://127.0.0.1:8001/health", "Reranker (:8001)"),
    ("http://127.0.0.1:6655/health/liveliness", "LiteLLM Proxy (:6655)"),
]


def wait_for_services(timeout_seconds: int = 40) -> bool:
    print("Waiting for model endpoints to become ready...")
    start_time = time.time()
    ready: dict[str, bool] = {name: False for _, name in CHECK_URLS}

    while time.time() - start_time < timeout_seconds:
        for url, name in CHECK_URLS:
            if not ready[name]:
                try:
                    resp = httpx.get(url, timeout=1.0)
                    if resp.status_code in (
                        200,
                        401,
                    ):  # 401 means service is up and auth required
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
        if os.path.exists(SERVE_SCRIPT):
            print(f"Starting model servers via {SERVE_SCRIPT}...")
            server_process = subprocess.Popen(["bash", SERVE_SCRIPT])
            wait_for_services()

        print("Launching Textual Chat App...")
        exit_code = subprocess.call(
            ["uv", "run", "--package", "fp-int-agents", "fp-int-agents"]
        )
        sys.exit(exit_code)
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
