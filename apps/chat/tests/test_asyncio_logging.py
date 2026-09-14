import asyncio
import logging
import pathlib
import time
from collections.abc import Generator

import pytest

from fp_int_agents import setup_asyncio_logging


@pytest.fixture
def clean_asyncio_logger() -> Generator[logging.Logger]:
    logger = logging.getLogger("asyncio")
    orig_handlers = list(logger.handlers)
    orig_level = logger.level
    orig_propagate = logger.propagate
    try:
        yield logger
    finally:
        for h in list(logger.handlers):
            if h not in orig_handlers:
                logger.removeHandler(h)
                h.close()
        logger.handlers = orig_handlers
        logger.setLevel(orig_level)
        logger.propagate = orig_propagate


def test_setup_asyncio_logging_creates_file_and_logs(
    tmp_path: pathlib.Path, clean_asyncio_logger: logging.Logger
) -> None:
    log_file = tmp_path / "asyncio.log"
    handler = setup_asyncio_logging(log_file=log_file)

    assert log_file.exists()
    assert clean_asyncio_logger.propagate is False
    assert handler in clean_asyncio_logger.handlers

    clean_asyncio_logger.warning("Test warning from asyncio logger")
    handler.flush()

    content = log_file.read_text(encoding="utf-8")
    assert "Test warning from asyncio logger" in content
    assert "[WARNING] asyncio:" in content


def test_asyncio_debug_slow_callback_is_logged(
    tmp_path: pathlib.Path, clean_asyncio_logger: logging.Logger
) -> None:
    log_file = tmp_path / "slow_callback.log"
    handler = setup_asyncio_logging(log_file=log_file)

    async def _runner() -> None:
        loop = asyncio.get_running_loop()
        loop.set_debug(True)
        loop.slow_callback_duration = 0.01

        def _blocking_call() -> None:
            time.sleep(0.03)

        loop.call_soon(_blocking_call)
        await asyncio.sleep(0.05)

    asyncio.run(_runner())
    handler.flush()

    content = log_file.read_text(encoding="utf-8")
    assert "Executing" in content
    assert "took" in content
