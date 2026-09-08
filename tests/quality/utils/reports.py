"""Генерация отчетов quality gate в JSON/CSV/PNG."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from .config import GateConfig
from .evaluate import GateResult
from .io import atomic_write_text
from .metrics import (
    ApiError,
    CategoryMetrics,
    EntityError,
    EvaluationResult,
)


@dataclass
class ReportPaths:
    run_dir: Path
    report_json: Path
    result_json: Path
    per_category_csv: Path
    errors_csv: Path
    api_errors_csv: Path
    metrics_png: Path | None = None
    counts_png: Path | None = None
    graphs_note: Path | None = None


@dataclass
class GraphArtifacts:
    metrics_png: Path | None = None
    counts_png: Path | None = None
    graphs_note: Path | None = None


def write_reports(
    config: GateConfig,
    evaluation: EvaluationResult,
    gates: GateResult,
) -> ReportPaths:
    """Сохраняет отчеты прогона в файловую систему.

    Parameters:
        config: Конфигурация quality gate.
        evaluation: Расчетные метрики и ошибки текущего прогона.
        gates: Статусы прохождения gate-проверок.

    Returns:
        ReportPaths: Пути к артефактам отчета (JSON/CSV/PNG).
    """
    day = evaluation.generated_at_utc.strftime("%Y-%m-%d")
    ts = evaluation.generated_at_utc.strftime("%Y%m%dT%H%M%SZ")
    run_dir = config.report_root / day / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    per_category_csv = run_dir / "per_category_metrics.csv"
    errors_csv = run_dir / "entity_errors.csv"
    api_errors_csv = run_dir / "api_errors.csv"
    report_json = run_dir / "entity_quality_report.json"
    result_json = run_dir / "result.json"

    _write_per_category_csv(per_category_csv, evaluation.per_category)
    _write_errors_csv(errors_csv, evaluation.entity_errors)
    _write_api_errors_csv(api_errors_csv, evaluation.api_errors)

    graph_paths = _write_graphs(
        out_dir=run_dir,
        per_category=evaluation.per_category,
        run_label=evaluation.generated_at_utc.isoformat(),
        enable_plots=config.enable_plots,
    )

    artifacts: dict[str, str] = {
        "per_category_csv": str(per_category_csv.resolve()),
        "errors_csv": str(errors_csv.resolve()),
        "api_errors_csv": str(api_errors_csv.resolve()),
    }
    if graph_paths.metrics_png is not None:
        artifacts["metrics_png"] = str(graph_paths.metrics_png.resolve())
    if graph_paths.counts_png is not None:
        artifacts["counts_png"] = str(graph_paths.counts_png.resolve())
    if graph_paths.graphs_note is not None:
        artifacts["graphs_note"] = str(graph_paths.graphs_note.resolve())

    report_payload: dict[str, object] = {
        "generated_at_utc": evaluation.generated_at_utc.isoformat(),
        "generated_at_local": evaluation.generated_at_utc.astimezone().isoformat(),
        "dataset": evaluation.dataset_meta.as_dict(),
        "run_config": {
            "api_url": config.api_url,
            "api_timeout": config.api_timeout,
            "api_retries": config.api_retries,
            "api_error_rate_threshold": config.api_error_rate_threshold,
            "micro_f1_threshold": config.micro_f1_threshold,
            "category_floors": config.category_floors,
            "timeout_policy": config.timeout_policy,
            "max_f1_drop": config.max_f1_drop,
            "require_same_dataset_for_regression": config.require_same_dataset_for_regression,
            "baseline_result_path": (
                str(config.baseline_result_path.resolve())
                if config.baseline_result_path is not None
                else None
            ),
        },
        "summary": {
            "cases_total": evaluation.cases_total,
            "unique_sentences": evaluation.unique_sentences,
            "unmapped_gold_labels": evaluation.unmapped_gold_labels,
            "missing_categories": evaluation.missing_categories,
            "entity_errors_count": len(evaluation.entity_errors),
            "api_errors_count": len(evaluation.api_errors),
            "warnings": gates.warnings,
        },
        "overall": evaluation.overall.as_dict(),
        "per_category": {
            category: metrics.as_dict()
            for category, metrics in evaluation.per_category.items()
        },
        "gates": gates.as_dict(),
        "artifacts": artifacts,
    }

    report_payload_text = json.dumps(report_payload, ensure_ascii=False, indent=2)
    atomic_write_text(report_json, report_payload_text)

    result_payload: dict[str, object] = {
        "generated_at_utc": evaluation.generated_at_utc.isoformat(),
        "passed": gates.overall_passed,
        "micro_f1": evaluation.overall.f1,
        "threshold_micro_f1": config.micro_f1_threshold,
        "dataset_sha256": evaluation.dataset_meta.sha256,
        "dataset": evaluation.dataset_meta.as_dict(),
        "report_json": str(report_json.resolve()),
        "gates": gates.as_dict(),
        "warnings": gates.warnings,
    }
    result_payload_text = json.dumps(result_payload, ensure_ascii=False, indent=2)
    atomic_write_text(result_json, result_payload_text)

    latest = config.report_root / "latest_result.json"
    latest.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(latest, result_payload_text)

    return ReportPaths(
        run_dir=run_dir,
        report_json=report_json,
        result_json=result_json,
        per_category_csv=per_category_csv,
        errors_csv=errors_csv,
        api_errors_csv=api_errors_csv,
        metrics_png=graph_paths.metrics_png,
        counts_png=graph_paths.counts_png,
        graphs_note=graph_paths.graphs_note,
    )


def _write_per_category_csv(
    path: Path, per_category: dict[str, CategoryMetrics]
) -> None:
    fieldnames = ["category", "support", "tp", "fp", "fn", "precision", "recall", "f1"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for category in sorted(per_category):
            row: dict[str, float | int | str] = {"category": category}
            row.update(per_category[category].as_dict())
            writer.writerow(row)


def _write_errors_csv(path: Path, rows: list[EntityError]) -> None:
    fieldnames = [
        "error_type",
        "row_id",
        "source",
        "text",
        "label",
        "gold_label",
        "pred_label",
        "start",
        "end",
        "score",
        "span_text",
        "overlap_gold_labels",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def _write_api_errors_csv(path: Path, rows: list[ApiError]) -> None:
    fieldnames = ["sentence", "error"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def _write_graphs(
    out_dir: Path,
    per_category: dict[str, CategoryMetrics],
    run_label: str,
    enable_plots: bool,
) -> GraphArtifacts:
    artifacts = GraphArtifacts()

    if not enable_plots:
        note = out_dir / "graphs_disabled.txt"
        note.write_text(
            "Plot generation disabled by PII_ENABLE_PLOTS.\n",
            encoding="utf-8",
        )
        artifacts.graphs_note = note
        return artifacts

    categories = sorted(per_category)
    if not categories:
        return artifacts

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        note = out_dir / "graphs_unavailable.txt"
        note.write_text(
            f"Matplotlib is unavailable, graphs were not generated: {exc}\n",
            encoding="utf-8",
        )
        artifacts.graphs_note = note
        return artifacts

    precision = [float(per_category[c].precision) for c in categories]
    recall = [float(per_category[c].recall) for c in categories]
    f1 = [float(per_category[c].f1) for c in categories]
    tp = [int(per_category[c].tp) for c in categories]
    fp = [int(per_category[c].fp) for c in categories]
    fn = [int(per_category[c].fn) for c in categories]

    fig, ax = plt.subplots(figsize=(14, 6))
    _draw_bar_chart(
        ax=ax,
        categories=categories,
        series=[
            ("Precision", precision),
            ("Recall", recall),
            ("F1", f1),
        ],
        title=f"Entity-Level Metrics by Category ({run_label})",
        y_lim=(0.0, 1.02),
    )
    fig.tight_layout()
    metrics_png = out_dir / "metrics_by_category.png"
    fig.savefig(metrics_png, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 6))
    _draw_bar_chart(
        ax=ax,
        categories=categories,
        series=[
            ("TP", tp),
            ("FP", fp),
            ("FN", fn),
        ],
        title=f"Entity Counts by Category ({run_label})",
        y_lim=None,
    )
    fig.tight_layout()
    counts_png = out_dir / "counts_by_category.png"
    fig.savefig(counts_png, dpi=160)
    plt.close(fig)

    artifacts.metrics_png = metrics_png
    artifacts.counts_png = counts_png
    return artifacts


def _draw_bar_chart(
    ax,
    categories: list[str],
    series: list[tuple[str, list[int] | list[float]]],
    title: str,
    y_lim: tuple[float, float] | None,
) -> None:
    x = list(range(len(categories)))
    width = 0.8 / max(1, len(series))
    start_offset = -((len(series) - 1) / 2) * width

    for idx, (label, values) in enumerate(series):
        offset = start_offset + idx * width
        ax.bar([v + offset for v in x], values, width=width, label=label)

    if y_lim is not None:
        ax.set_ylim(*y_lim)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=35, ha="right")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
