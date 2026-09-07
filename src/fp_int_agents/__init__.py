import asyncio
import pathlib
import sys

from fp_int_agents.app import FPIntAgentsApp
from fp_int_agents.config import load_config
from fp_int_agents.storage.db import open_db

from .config import DATA_DIR


def _check_project_root() -> None:
    if not pathlib.Path("pyproject.toml").exists():
        print("Error: must be run from the project root (no pyproject.toml found).")
        sys.exit(1)
    DATA_DIR.mkdir(exist_ok=True)


def main() -> None:
    _check_project_root()
    asyncio.run(_run())


async def _run() -> None:
    cfg = await load_config()
    async with open_db(cfg.db_path) as db:
        await FPIntAgentsApp(db, cfg).run_async()
