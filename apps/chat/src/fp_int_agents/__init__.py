import asyncio
import logging
import os
import pathlib
import sys

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from fp_int_agents.app import FPIntAgentsApp
from fp_int_agents.config import load_config
from fp_int_agents.storage.db import open_db

from .config import DATA_DIR

__all__ = ["FPIntAgentsApp", "main", "setup_asyncio_logging"]


def setup_asyncio_logging(
    log_file: pathlib.Path = DATA_DIR / "asyncio.log",
    level: int = logging.DEBUG,
) -> logging.FileHandler:
    """Route asyncio event loop logs and slow-callback warnings to a dedicated file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("asyncio")
    logger.setLevel(level)
    logger.propagate = False

    resolved_path = log_file.resolve()
    for h in logger.handlers:
        if (
            isinstance(h, logging.FileHandler)
            and pathlib.Path(h.baseFilename) == resolved_path
        ):
            return h

    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return handler


def _check_project_root() -> None:
    if not pathlib.Path("pyproject.toml").exists():
        print("Error: must be run from the project root (no pyproject.toml found).")
        sys.exit(1)
    DATA_DIR.mkdir(exist_ok=True)


def main() -> None:
    _check_project_root()
    setup_asyncio_logging()
    debug = os.getenv("ASYNCIO_DEBUG", "1").lower() not in ("0", "false", "no")
    asyncio.run(_run(), debug=debug)


async def _run() -> None:
    cfg = await load_config()
    async with open_db(cfg.db_path) as db:
        await FPIntAgentsApp(db, cfg).run_async()
