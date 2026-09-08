"""Shared test bootstrap.

The unit suite exercises the rules branch only -- entity classifiers, the
framework and the bypass preprocessors. It deliberately runs without torch,
without spaCy and without model weights, so a contributor can clone, install
``.[dev]`` and get a green run in seconds.

To make that possible a minimal ``presidio_analyzer`` stub is installed when the
real package is absent: the classifiers only need ``Pattern``,
``PatternRecognizer``, ``EntityRecognizer`` and ``RecognizerResult`` as data
holders. When presidio *is* installed the real package wins and the stub is
never created.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

# Works both for an editable install and for a bare `pytest` in a fresh clone.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _install_presidio_stub_if_missing() -> None:
    """Install a minimal presidio_analyzer stub for lightweight unit tests."""
    try:
        import presidio_analyzer  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    stub = ModuleType("presidio_analyzer")

    class AnalyzerEngine:  # pragma: no cover - compatibility shim
        pass

    class EntityRecognizer:  # pragma: no cover - compatibility shim
        def __init__(self, supported_entities=None, name="", supported_language="ru"):
            self.supported_entities = supported_entities or []
            self.name = name
            self.supported_language = supported_language

    @dataclass
    class RecognizerResult:  # pragma: no cover - compatibility shim
        entity_type: str
        start: int
        end: int
        score: float

    class Pattern:  # pragma: no cover - compatibility shim
        def __init__(self, name: str, regex: str, score: float) -> None:
            self.name = name
            self.regex = regex
            self.score = score

    class PatternRecognizer:  # pragma: no cover - compatibility shim
        def __init__(
            self,
            supported_entity: str,
            name: str,
            patterns: list[Pattern],
            supported_language: str = "ru",
            context: list[str] | None = None,
        ) -> None:
            self.supported_entity = supported_entity
            self.name = name
            self.patterns = patterns
            self.supported_language = supported_language
            self.context = context or []

    stub.AnalyzerEngine = AnalyzerEngine
    stub.EntityRecognizer = EntityRecognizer
    stub.RecognizerResult = RecognizerResult
    stub.Pattern = Pattern
    stub.PatternRecognizer = PatternRecognizer
    sys.modules["presidio_analyzer"] = stub


_install_presidio_stub_if_missing()


QUALITY_SUMMARIES_KEY = pytest.StashKey[list[str]]()


@pytest.fixture(scope="session")
def quality_summary_sink(pytestconfig: pytest.Config):
    """Store quality-gate terminal summary for `pytest_terminal_summary` hook."""

    def _save(summary: str) -> None:
        summaries = list(pytestconfig.stash.get(QUALITY_SUMMARIES_KEY, []))
        summaries.append(summary)
        pytestconfig.stash[QUALITY_SUMMARIES_KEY] = summaries

    return _save


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    """Always print quality-gate metrics block if it was produced in tests."""
    summaries = config.stash.get(QUALITY_SUMMARIES_KEY, [])
    if not summaries:
        return
    terminalreporter.ensure_newline()
    terminalreporter.section("Quality Gate Summaries", sep="=")
    for index, summary in enumerate(summaries):
        if index:
            terminalreporter.ensure_newline()
        terminalreporter.write_line(summary)
