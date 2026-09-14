"""Tests for folder-based document ingestion."""

import pathlib
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from fp_int_agents.app import FPIntAgentsApp

TEXT_EXTENSIONS = [".txt", ".md", ".py"]
SUPPORTED = TEXT_EXTENSIONS + [".pdf"]
UNSUPPORTED = [".jpg", ".bin", ".csv"]


class TestCollectIngestPaths:
    def test_single_file_returns_itself(self, tmp_path: pathlib.Path) -> None:
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"")
        result = FPIntAgentsApp._collect_ingest_paths(str(f))
        assert result == [f]

    def test_folder_returns_pdf_and_text_files(self, tmp_path: pathlib.Path) -> None:
        (tmp_path / "a.pdf").write_bytes(b"")
        (tmp_path / "b.txt").write_bytes(b"")
        (tmp_path / "c.md").write_bytes(b"")
        (tmp_path / "d.jpg").write_bytes(b"")  # should be excluded
        result = FPIntAgentsApp._collect_ingest_paths(str(tmp_path))
        names = {p.name for p in result}
        assert names == {"a.pdf", "b.txt", "c.md"}

    def test_folder_skips_unsupported_extensions(self, tmp_path: pathlib.Path) -> None:
        for ext in UNSUPPORTED:
            (tmp_path / f"file{ext}").write_bytes(b"")
        result = FPIntAgentsApp._collect_ingest_paths(str(tmp_path))
        assert result == []

    def test_folder_includes_all_supported_extensions(
        self, tmp_path: pathlib.Path
    ) -> None:
        for ext in SUPPORTED:
            (tmp_path / f"file{ext}").write_bytes(b"")
        result = FPIntAgentsApp._collect_ingest_paths(str(tmp_path))
        assert len(result) == len(SUPPORTED)

    def test_empty_folder_returns_empty(self, tmp_path: pathlib.Path) -> None:
        result = FPIntAgentsApp._collect_ingest_paths(str(tmp_path))
        assert result == []


def _make_app() -> FPIntAgentsApp:
    """Create a bare FPIntAgentsApp instance without calling __init__."""
    app = FPIntAgentsApp.__new__(FPIntAgentsApp)
    app.notify = MagicMock()
    app.config = MagicMock()
    app.db = MagicMock()
    return app


class TestRunIngestFolder:
    @pytest.mark.asyncio
    async def test_folder_ingests_all_files(self, tmp_path: pathlib.Path) -> None:
        (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
        (tmp_path / "b.txt").write_text("world", encoding="utf-8")

        mock_project = MagicMock()
        mock_project.ingestor = "simple"

        ingested_titles: list[str] = []

        async def fake_ingestor(project, text, llm_config, conn, title):
            ingested_titles.append(title)

        app = _make_app()
        with (
            patch.dict(
                "fp_int_agents.agents.registry.INGESTOR_AGENTS",
                {"simple": fake_ingestor},
            ),
            patch.object(
                type(app), "log", new_callable=PropertyMock, return_value=MagicMock()
            ),
        ):
            await app._run_ingest(mock_project, str(tmp_path))

        assert set(ingested_titles) == {"a.txt", "b.txt"}

    @pytest.mark.asyncio
    async def test_empty_folder_notifies_warning(self, tmp_path: pathlib.Path) -> None:
        mock_project = MagicMock()
        mock_project.ingestor = "simple"

        app = _make_app()
        with patch.object(
            type(app), "log", new_callable=PropertyMock, return_value=MagicMock()
        ):
            await app._run_ingest(mock_project, str(tmp_path))

        app.notify.assert_called_once()
        call_kwargs = app.notify.call_args
        assert call_kwargs.kwargs.get("severity") == "warning"
