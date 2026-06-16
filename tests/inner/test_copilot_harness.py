"""Tests for ``omnigent/inner/copilot_harness.py`` — the ``harness: copilot`` wrap.

The wrap reads ``HARNESS_COPILOT_*`` env vars and constructs a
:class:`CopilotExecutor` lazily. Constructing the executor does NOT import the
``github-copilot-sdk`` package (that import is deferred to the first turn), so
these tests run without the optional SDK installed. Mirrors
``test_cursor_harness.py``.
"""

from __future__ import annotations

import json

import pytest

from omnigent.inner import copilot_harness as ch
from omnigent.inner.copilot_executor import CopilotExecutor


@pytest.fixture(autouse=True)
def _clear_harness_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "HARNESS_COPILOT_MODEL",
        "HARNESS_COPILOT_CWD",
        "HARNESS_COPILOT_GITHUB_TOKEN",
        "HARNESS_COPILOT_OS_ENV",
        "HARNESS_COPILOT_SKILLS_FILTER",
        "HARNESS_COPILOT_BUNDLE_DIR",
        "HARNESS_COPILOT_AGENT_NAME",
        "COPILOT_GITHUB_TOKEN",
        "GH_TOKEN",
        "GITHUB_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)


def test_resolve_os_env_default() -> None:
    os_env = ch._resolve_os_env()
    assert os_env.type == "caller_process"
    assert os_env.sandbox is not None and os_env.sandbox.type == "none"


def test_resolve_os_env_from_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "HARNESS_COPILOT_OS_ENV",
        json.dumps({"type": "caller_process", "cwd": "/tmp/x", "sandbox": {"type": "none"}}),
    )
    os_env = ch._resolve_os_env()
    assert os_env.cwd == "/tmp/x"


def test_resolve_skills_filter_defaults_all(monkeypatch: pytest.MonkeyPatch) -> None:
    assert ch._resolve_skills_filter() == "all"
    monkeypatch.setenv("HARNESS_COPILOT_SKILLS_FILTER", json.dumps(["a", "b"]))
    assert ch._resolve_skills_filter() == ["a", "b"]
    monkeypatch.setenv("HARNESS_COPILOT_SKILLS_FILTER", "not-json")
    assert ch._resolve_skills_filter() == "all"


def test_build_executor_threads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARNESS_COPILOT_MODEL", "claude-haiku-4.5")
    monkeypatch.setenv("HARNESS_COPILOT_GITHUB_TOKEN", "gho_test")
    monkeypatch.setenv("HARNESS_COPILOT_CWD", "/tmp/work")
    monkeypatch.setenv("HARNESS_COPILOT_AGENT_NAME", "cop")
    executor = ch._build_copilot_executor()
    assert isinstance(executor, CopilotExecutor)
    assert executor._model_override == "claude-haiku-4.5"
    assert executor._github_token == "gho_test"
    assert executor._cwd == "/tmp/work"
    assert executor._agent_name == "cop"


def test_create_app_builds_fastapi() -> None:
    app = ch.create_app()
    # The ExecutorAdapter builds a FastAPI app; just assert it has routes.
    assert hasattr(app, "routes")
