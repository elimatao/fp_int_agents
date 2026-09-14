"""Ingestors register themselves on package import."""

import inspect


def test_ingestors_registered() -> None:
    import fp_int_agents.agents  # noqa: F401 - triggers registration side effects
    from fp_int_agents.agents.registry import INGESTOR_AGENTS

    assert {"simple", "BundesRAG"} <= set(INGESTOR_AGENTS)
    assert all(inspect.iscoroutinefunction(fn) for fn in INGESTOR_AGENTS.values())
