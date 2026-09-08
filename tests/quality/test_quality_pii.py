"""Entity-level quality gate tests for supported PII types."""

from __future__ import annotations

import urllib.error
from collections.abc import Callable
from dataclasses import dataclass

import pytest

from .utils import (
    CORE_CATEGORIES,
    GateConfig,
    GateResult,
    ReportPaths,
    create_client,
    evaluate_baseline_regression,
    evaluate_entity_quality,
    evaluate_gates,
    load_gate_config,
    render_terminal_summary,
    transport_name,
    write_reports,
)
from .utils.metrics import EvaluationResult

try:  # torch is optional; without it the weights error type does not exist
    from pii_guard.ner.recognizer import ModelUnavailableError
except ImportError:  # pragma: no cover -- rules-only install
    class ModelUnavailableError(RuntimeError):  # type: ignore[no-redef]
        """Placeholder so the except clause below stays valid without torch."""

pytestmark = pytest.mark.quality


@dataclass
class GateRunResult:
    config: GateConfig
    evaluation: EvaluationResult
    gates: GateResult
    report_paths: ReportPaths


@pytest.fixture(scope="module")
def gate_run(
    pytestconfig: pytest.Config,
    quality_summary_sink: Callable[[str], None],
) -> GateRunResult:
    config = load_gate_config()
    terminal_reporter = pytestconfig.pluginmanager.get_plugin("terminalreporter")
    capture_manager = pytestconfig.pluginmanager.get_plugin("capturemanager")
    progress_bar: object | None = None
    last_current = 0

    def progress_sink(message: str, final: bool, current: int, total: int) -> None:
        nonlocal progress_bar, last_current

        def _emit() -> None:
            nonlocal progress_bar, last_current

            if progress_bar is None:
                try:
                    from tqdm import tqdm  # type: ignore[import-untyped]
                except Exception:  # pragma: no cover - optional dependency
                    if terminal_reporter is None:
                        print(message, flush=True)
                    else:
                        terminal_reporter.write_line(message)
                    return

                progress_bar = tqdm(
                    total=total,
                    desc="pii-gate",
                    unit="case",
                    dynamic_ncols=True,
                    leave=True,
                )

            update_step = max(current - last_current, 0)
            if update_step:
                progress_bar.update(update_step)
                last_current = current

            if final:
                progress_bar.close()
                progress_bar = None

        if capture_manager is None:
            _emit()
            return

        with capture_manager.global_and_fixture_disabled():
            _emit()

    try:
        guard_api = create_client(config)
    except urllib.error.URLError as exc:
        pytest.skip(
            f"Guard API is unavailable at {config.api_url} "
            f"(PII_GATE_TRANSPORT={transport_name()}). Details: {exc}"
        )
    except PermissionError as exc:
        pytest.skip(str(exc))
    except ImportError as exc:
        # Local transport needs the `ner` extra; there is nothing to measure
        # without the model, so skip rather than report zeroes as a regression.
        pytest.skip(f"engine unavailable for local transport: {exc}")
    except ModelUnavailableError as exc:
        # Weights absent with torch installed. A RuntimeError, so the ImportError
        # handler above never caught it and a fresh clone hit a hard failure on a
        # command CONTRIBUTING tells contributors to run.
        pytest.skip(str(exc))

    try:
        evaluation = evaluate_entity_quality(
            config,
            guard_api,
            progress_sink=progress_sink,
        )
    except FileNotFoundError as exc:
        # The quality dataset is deliberately untracked (see .gitignore): 57 of its
        # rows come from production logs. A fresh clone therefore has no dataset,
        # which is a reason to skip, not to fail.
        pytest.skip(f"quality dataset unavailable: {exc}")
    regression = evaluate_baseline_regression(
        current_micro_f1=float(evaluation.overall.f1),
        current_dataset_sha256=evaluation.dataset_meta.sha256,
        baseline_path=config.baseline_result_path,
        max_drop=config.max_f1_drop,
        require_same_dataset=config.require_same_dataset_for_regression,
    )
    gates = evaluate_gates(config, evaluation, regression)
    report_paths = write_reports(config, evaluation, gates)

    summary = render_terminal_summary(
        config,
        evaluation,
        gates,
        report_json_path=report_paths.report_json,
        result_json_path=report_paths.result_json,
    )
    print(summary)
    quality_summary_sink(summary)

    return GateRunResult(
        config=config,
        evaluation=evaluation,
        gates=gates,
        report_paths=report_paths,
    )


def test_dataset_coverage_gate(gate_run: GateRunResult) -> None:
    report = gate_run.report_paths.report_json
    assert gate_run.gates.missing_categories_passed, (
        "Dataset does not cover all categories: "
        f"{gate_run.gates.missing_categories}. "
        f"Report: {report}"
    )


def test_micro_f1_gate(gate_run: GateRunResult) -> None:
    report = gate_run.report_paths.report_json
    assert gate_run.gates.micro_f1_passed, (
        f"Micro-F1 {gate_run.evaluation.overall.f1:.3f} < threshold "
        f"{gate_run.gates.micro_f1_threshold:.2f}. "
        f"Report: {report}"
    )


def test_regression_gate(gate_run: GateRunResult) -> None:
    report = gate_run.report_paths.report_json
    regression = gate_run.gates.regression
    assert regression.passed, (
        "Regression gate failed: "
        f"baseline={regression.baseline_micro_f1}, "
        f"current={gate_run.evaluation.overall.f1:.3f}, "
        f"drop={regression.drop}, "
        f"max_drop={regression.max_drop}, "
        f"same_dataset={regression.same_dataset}. "
        f"Report: {report}"
    )


def test_api_error_gate(gate_run: GateRunResult) -> None:
    report = gate_run.report_paths.report_json
    assert gate_run.gates.api_error_passed, (
        "API error gate failed: "
        f"api_error_count={gate_run.gates.api_error_count}, "
        f"api_error_total={gate_run.gates.api_error_total}, "
        f"api_error_rate={gate_run.gates.api_error_rate:.2%}, "
        f"api_error_rate_threshold={gate_run.gates.api_error_rate_threshold:.2%}, "
        f"timeout_policy={gate_run.gates.timeout_policy}. "
        f"Report: {report}"
    )


@pytest.mark.parametrize("category", sorted(CORE_CATEGORIES))
def test_category_floor(category: str, gate_run: GateRunResult) -> None:
    report = gate_run.report_paths.report_json

    floor = float(gate_run.config.category_floors.get(category, 0.0))
    value = float(gate_run.evaluation.per_category[category].f1)
    assert value >= floor, (
        f"{category} F1 {value:.3f} < floor {floor:.2f}. Report: {report}"
    )
