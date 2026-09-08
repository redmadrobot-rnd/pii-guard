"""Оркестрация оценки качества PII-маскирования."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Literal

from .api_client import (
    GuardApiClient,
    TimeoutOrGatewayError,
    anonymize_with_retry,
)
from .config import GateConfig
from .iob2_parser import (
    API_TO_GOLD,
    DatasetCase,
    load_dataset_cases,
)
from .metrics import (
    ApiError,
    CategoryMetrics,
    CategoryStats,
    EntityError,
    EvaluationResult,
    OverallMetrics,
    PredEntity,
    build_dataset_meta,
    match_entities,
    score_from_stats,
    spans_overlap,
)

_PREDICTED_LABEL_ALIASES: dict[str, str] = {
    "IP_PORT": "IP_ADDRESS",
}

ProgressSink = Callable[[str, bool, int, int], None]


@dataclass
class RegressionResult:
    checked: bool
    baseline_path: str
    baseline_micro_f1: float | None
    baseline_dataset_sha256: str | None
    current_dataset_sha256: str
    same_dataset: bool | None
    max_drop: float
    drop: float | None
    passed: bool
    warning: str | None

    def as_dict(self) -> dict[str, str | float | bool | None]:
        return {
            "checked": self.checked,
            "baseline_path": self.baseline_path,
            "baseline_micro_f1": self.baseline_micro_f1,
            "baseline_dataset_sha256": self.baseline_dataset_sha256,
            "current_dataset_sha256": self.current_dataset_sha256,
            "same_dataset": self.same_dataset,
            "max_drop": self.max_drop,
            "drop": self.drop,
            "passed": self.passed,
            "warning": self.warning,
        }


@dataclass
class CategoryFailure:
    category: str
    metric: Literal["f1"]
    value: float
    floor: float

    def as_dict(self) -> dict[str, str | float]:
        return {
            "category": self.category,
            "metric": self.metric,
            "value": self.value,
            "floor": self.floor,
        }


@dataclass
class GateResult:
    overall_passed: bool
    missing_categories_passed: bool
    missing_categories: list[str]
    micro_f1_passed: bool
    micro_f1_threshold: float
    category_floor_passed: bool
    category_failures: list[CategoryFailure]
    api_error_passed: bool
    api_error_count: int
    api_error_rate: float
    api_error_rate_threshold: float
    api_error_total: int
    timeout_policy: str
    regression: RegressionResult
    warnings: list[str]

    def as_dict(self) -> dict[str, object]:
        return {
            "overall_passed": self.overall_passed,
            "missing_categories_passed": self.missing_categories_passed,
            "missing_categories": self.missing_categories,
            "micro_f1_passed": self.micro_f1_passed,
            "micro_f1_threshold": self.micro_f1_threshold,
            "category_floor_passed": self.category_floor_passed,
            "category_failures": [item.as_dict() for item in self.category_failures],
            "api_error_passed": self.api_error_passed,
            "api_error_count": self.api_error_count,
            "api_error_rate": self.api_error_rate,
            "api_error_rate_threshold": self.api_error_rate_threshold,
            "api_error_total": self.api_error_total,
            "timeout_policy": self.timeout_policy,
            "regression": self.regression.as_dict(),
            "warnings": self.warnings,
        }


def evaluate_entity_quality(
    config: GateConfig,
    guard_api: GuardApiClient,
    progress_sink: ProgressSink | None = None,
) -> EvaluationResult:
    """Собирает предсказания API и считает entity-level метрики по датасету.

    Parameters:
        config: Конфигурация quality gate.
        guard_api: Инициализированный клиент Guard API.

    Returns:
        EvaluationResult: Итог метрик, ошибок и предупреждений.
    """
    generated_at_utc = datetime.now(timezone.utc)
    dataset_meta = build_dataset_meta(config.dataset_path)

    cases, observed, unmapped_gold_labels = load_dataset_cases(config.dataset_path)
    if not cases:
        raise FileNotFoundError(f"No dataset cases found in {config.dataset_path}")

    missing_categories = sorted(set(config.categories) - observed)
    unique_sentences = sorted({case.text for case in cases})

    predictions, api_errors = _collect_predictions(
        config=config,
        guard_api=guard_api,
        sentences=unique_sentences,
        progress_sink=progress_sink,
    )
    overall, per_category, entity_errors, report_lines = _compute_category_metrics(
        config=config,
        cases=cases,
        predictions=predictions,
    )

    warnings: list[str] = []
    if api_errors:
        warnings.append(
            "Во время запроса к Guard API были сетевые ошибки: "
            f"{len(api_errors)} предложений обработаны как пустой ответ, "
            "из-за чего FN и итоговый F1 могли ухудшиться."
        )

    return EvaluationResult(
        generated_at_utc=generated_at_utc,
        dataset_meta=dataset_meta,
        overall=overall,
        per_category=per_category,
        missing_categories=missing_categories,
        unmapped_gold_labels=sorted(unmapped_gold_labels),
        entity_errors=entity_errors,
        api_errors=api_errors,
        cases_total=len(cases),
        unique_sentences=len(unique_sentences),
        report_lines=report_lines,
        warnings=warnings,
    )


def _collect_predictions(
    config: GateConfig,
    guard_api: GuardApiClient,
    sentences: list[str],
    progress_sink: ProgressSink | None,
) -> tuple[dict[str, list[PredEntity]], list[ApiError]]:
    predictions: dict[str, list[PredEntity]] = {sentence: [] for sentence in sentences}
    api_errors: list[ApiError] = []

    for sentence in _iter_sentences_with_progress(
        sentences=sentences,
        enabled=config.log_progress,
        progress_sink=progress_sink,
    ):
        try:
            response = anonymize_with_retry(
                guard_api, sentence, retries=config.api_retries
            )
            norm_text = response.get("normalized_text")
            norm_text = norm_text if isinstance(norm_text, str) else sentence
            entities = _extract_predicted_entities(response, norm_text, config.categories)
            predictions[sentence] = _map_predictions_to_orig(sentence, norm_text, entities)
        except TimeoutOrGatewayError as exc:
            api_errors.append(ApiError(sentence=sentence, error=str(exc)))
            if config.timeout_policy == "fail":
                raise
            predictions[sentence] = []

    return predictions, api_errors


def _iter_sentences_with_progress(
    sentences: list[str],
    enabled: bool,
    progress_sink: ProgressSink | None = None,
) -> Iterator[str]:
    if not enabled:
        yield from sentences
        return

    total = len(sentences)

    if progress_sink is not None:
        for idx, sentence in enumerate(sentences, start=1):
            progress_sink(f"[pii-gate] case {idx}/{total}", False, idx, total)
            yield sentence
        progress_sink(
            f"[pii-gate] processed {total}/{total} cases",
            True,
            total,
            total,
        )
        return

    # In non-interactive runs (common under pytest), tqdm can render as an
    # empty carriage-return line. Use plain textual progress instead.
    if not sys.stdout.isatty() or not sys.stderr.isatty():
        for idx, sentence in enumerate(sentences, start=1):
            print(f"[pii-gate] case {idx}/{total}", flush=True)
            yield sentence
        return

    try:
        from tqdm import tqdm  # type: ignore[import-untyped]
    except Exception:  # pragma: no cover - optional dependency
        for idx, sentence in enumerate(sentences, start=1):
            print(f"[pii-gate] case {idx}/{total}", flush=True)
            yield sentence
        return

    progress = tqdm(
        sentences,
        total=total,
        desc="pii",
        unit="case",
        dynamic_ncols=True,
        leave=True,
        file=sys.stdout,
    )
    try:
        for sentence in progress:
            yield sentence
    finally:
        progress.close()



def _extract_predicted_entities(
    response: dict[str, object],
    norm_text: str,
    allowed_categories: tuple[str, ...],
) -> list[PredEntity]:
    entities_raw = response.get("entities")
    if not isinstance(entities_raw, list):
        return []

    allowed_set = set(allowed_categories)
    entities: list[PredEntity] = []
    for item in entities_raw:
        if not isinstance(item, dict):
            continue

        entity_type = item.get("entity_type")
        start = item.get("start")
        end = item.get("end")
        score = item.get("score")

        if not isinstance(entity_type, str):
            continue
        entity_type = _normalize_predicted_label(entity_type)
        if entity_type not in allowed_set:
            continue
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if start < 0 or end <= start or end > len(norm_text):
            continue

        parsed_score = float(score) if isinstance(score, (int, float)) else None
        entities.append(
            PredEntity(
                entity_type=entity_type,
                start=start,
                end=end,
                score=parsed_score,
            )
        )

    return entities


def _normalize_predicted_label(entity_type: str) -> str:
    return _PREDICTED_LABEL_ALIASES.get(entity_type, entity_type)


def _map_span_via_opcodes(
    opcodes: list[tuple[str, int, int, int, int]],
    norm_start: int,
    norm_end: int,
) -> tuple[int | None, int | None]:
    """Переводит спан из norm_text в orig_text через opcodes SequenceMatcher."""
    orig_start = orig_end = None
    for tag, i1, i2, j1, j2 in opcodes:
        if j2 <= norm_start or j1 >= norm_end:
            continue
        if tag == "equal":
            os = i1 + (max(norm_start, j1) - j1)
            oe = i1 + (min(norm_end, j2) - j1)
        else:
            os, oe = i1, i2
        orig_start = os if orig_start is None else orig_start
        orig_end = oe
    return orig_start, orig_end


def _map_predictions_to_orig(
    orig_text: str,
    norm_text: str,
    entities: list[PredEntity],
) -> list[PredEntity]:
    """Переводит координаты предсказаний из norm_text в orig_text.

    Нужно потому что normalize_text() меняет длину текста (слова-числа → цифры,
    удаление soft-hyphen и zero-width символов), а API возвращает координаты
    относительно нормализованного текста.
    """
    if orig_text == norm_text:
        return entities
    opcodes = SequenceMatcher(None, orig_text, norm_text, autojunk=False).get_opcodes()
    mapped = []
    for pred in entities:
        os, oe = _map_span_via_opcodes(opcodes, pred.start, pred.end)
        if os is not None and oe is not None:
            mapped.append(PredEntity(entity_type=pred.entity_type, start=os, end=oe, score=pred.score))
    return mapped


def _compute_category_metrics(
    config: GateConfig,
    cases: list[DatasetCase],
    predictions: dict[str, list[PredEntity]],
) -> tuple[OverallMetrics, dict[str, CategoryMetrics], list[EntityError], list[str]]:
    gold_counts = {category: CategoryStats() for category in config.categories}
    fp_counts = dict.fromkeys(config.categories, 0)
    entity_errors: list[EntityError] = []

    for case in cases:
        gold_entities = case.gold_entities
        predicted_entities = predictions.get(case.text, [])
        gold_matched, pred_matched = match_entities(
            gold_entities=gold_entities,
            predicted_entities=predicted_entities,
            api_to_gold=API_TO_GOLD,
        )

        for gold_idx, gold in enumerate(gold_entities):
            if gold_matched[gold_idx]:
                gold_counts[gold.api_label].tp += 1
            else:
                gold_counts[gold.api_label].fn += 1
                entity_errors.append(
                    EntityError(
                        error_type="FN",
                        row_id=case.row_id,
                        source=case.source,
                        text=case.text,
                        label=gold.api_label,
                        gold_label=gold.gold_label,
                        pred_label="",
                        start=gold.start,
                        end=gold.end,
                        score=None,
                        span_text=gold.text,
                        overlap_gold_labels="",
                    )
                )

        for pred_idx, pred in enumerate(predicted_entities):
            if pred_matched[pred_idx]:
                continue
            fp_counts[pred.entity_type] += 1
            overlap_gold_labels = sorted(
                {
                    gold.gold_label
                    for gold in gold_entities
                    if spans_overlap(pred.start, pred.end, gold.start, gold.end)
                }
            )
            entity_errors.append(
                EntityError(
                    error_type="FP",
                    row_id=case.row_id,
                    source=case.source,
                    text=case.text,
                    label=pred.entity_type,
                    gold_label="",
                    pred_label=pred.entity_type,
                    start=pred.start,
                    end=pred.end,
                    score=pred.score,
                    span_text=case.text[pred.start : pred.end],
                    overlap_gold_labels="|".join(overlap_gold_labels),
                )
            )

    total_counts = CategoryStats()
    per_category: dict[str, CategoryMetrics] = {}
    report_lines: list[str] = []

    for category in config.categories:
        tp = gold_counts[category].tp
        fn = gold_counts[category].fn
        fp = fp_counts[category]
        precision, recall, f1 = score_from_stats(CategoryStats(tp=tp, fp=fp, fn=fn))

        support = tp + fn
        total_counts.tp += tp
        total_counts.fp += fp
        total_counts.fn += fn

        per_category[category] = CategoryMetrics(
            support=support,
            tp=tp,
            fp=fp,
            fn=fn,
            precision=round(precision, 6),
            recall=round(recall, 6),
            f1=round(f1, 6),
        )
        report_lines.append(
            f"{category:<18} support={support:4d} tp={tp:4d} fp={fp:4d} fn={fn:4d} "
            f"precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}"
        )

    overall_precision, overall_recall, overall_f1 = score_from_stats(total_counts)
    overall = OverallMetrics(
        tp=total_counts.tp,
        fp=total_counts.fp,
        fn=total_counts.fn,
        precision=round(overall_precision, 6),
        recall=round(overall_recall, 6),
        f1=round(overall_f1, 6),
    )

    return overall, per_category, entity_errors, report_lines


def evaluate_baseline_regression(
    current_micro_f1: float,
    current_dataset_sha256: str,
    baseline_path: Path | None,
    max_drop: float,
    require_same_dataset: bool,
) -> RegressionResult:
    """Сравнивает текущий micro-F1 с baseline и учитывает совпадение датасета."""
    baseline_path_str = str(baseline_path.resolve()) if baseline_path else ""
    if baseline_path is None:
        return RegressionResult(
            checked=False,
            baseline_path=baseline_path_str,
            baseline_micro_f1=None,
            baseline_dataset_sha256=None,
            current_dataset_sha256=current_dataset_sha256,
            same_dataset=None,
            max_drop=max_drop,
            drop=None,
            passed=True,
            warning="Regression check не выполнен: baseline path не задан.",
        )

    if not baseline_path.exists():
        return RegressionResult(
            checked=False,
            baseline_path=baseline_path_str,
            baseline_micro_f1=None,
            baseline_dataset_sha256=None,
            current_dataset_sha256=current_dataset_sha256,
            same_dataset=None,
            max_drop=max_drop,
            drop=None,
            passed=True,
            warning=(
                "Regression check не выполнен: baseline file отсутствует "
                f"({baseline_path_str})."
            ),
        )

    try:
        payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return RegressionResult(
            checked=False,
            baseline_path=baseline_path_str,
            baseline_micro_f1=None,
            baseline_dataset_sha256=None,
            current_dataset_sha256=current_dataset_sha256,
            same_dataset=None,
            max_drop=max_drop,
            drop=None,
            passed=True,
            warning=(
                f"Regression check не выполнен: baseline JSON не читается ({exc})."
            ),
        )

    if not isinstance(payload, dict):
        return RegressionResult(
            checked=False,
            baseline_path=baseline_path_str,
            baseline_micro_f1=None,
            baseline_dataset_sha256=None,
            current_dataset_sha256=current_dataset_sha256,
            same_dataset=None,
            max_drop=max_drop,
            drop=None,
            passed=True,
            warning="Regression check не выполнен: baseline имеет неожиданный формат.",
        )

    baseline_micro_f1 = _extract_baseline_f1(payload)
    baseline_dataset_sha256 = _extract_baseline_dataset_sha(payload)

    if baseline_micro_f1 is None:
        return RegressionResult(
            checked=False,
            baseline_path=baseline_path_str,
            baseline_micro_f1=None,
            baseline_dataset_sha256=baseline_dataset_sha256,
            current_dataset_sha256=current_dataset_sha256,
            same_dataset=(
                baseline_dataset_sha256 == current_dataset_sha256
                if baseline_dataset_sha256
                else None
            ),
            max_drop=max_drop,
            drop=None,
            passed=True,
            warning="Regression check не выполнен: baseline не содержит micro_f1.",
        )

    same_dataset: bool | None = None
    warning: str | None = None
    if baseline_dataset_sha256:
        same_dataset = baseline_dataset_sha256 == current_dataset_sha256
        if require_same_dataset and not same_dataset:
            warning = (
                "Regression check пропущен: baseline и текущий запуск используют разные "
                "dataset sha256."
            )
            return RegressionResult(
                checked=False,
                baseline_path=baseline_path_str,
                baseline_micro_f1=baseline_micro_f1,
                baseline_dataset_sha256=baseline_dataset_sha256,
                current_dataset_sha256=current_dataset_sha256,
                same_dataset=same_dataset,
                max_drop=max_drop,
                drop=None,
                passed=True,
                warning=warning,
            )
        if not same_dataset:
            warning = (
                "Baseline и текущий запуск используют разные dataset sha256; "
                "сравнение выполнено, но интерпретируйте осторожно."
            )

    drop = baseline_micro_f1 - current_micro_f1
    return RegressionResult(
        checked=True,
        baseline_path=baseline_path_str,
        baseline_micro_f1=baseline_micro_f1,
        baseline_dataset_sha256=baseline_dataset_sha256,
        current_dataset_sha256=current_dataset_sha256,
        same_dataset=same_dataset,
        max_drop=max_drop,
        drop=round(drop, 6),
        passed=drop <= max_drop,
        warning=warning,
    )


def evaluate_gates(
    config: GateConfig,
    evaluation: EvaluationResult,
    regression: RegressionResult,
) -> GateResult:
    """Собирает итоговые gate-статусы по quality run.

    Parameters:
        config: Конфигурация quality gate.
        evaluation: Результат расчета entity-level метрик.
        regression: Результат regression-check относительно baseline.

    Returns:
        GateResult: Статусы прохождения всех gate-проверок.
    """
    category_failures: list[CategoryFailure] = []
    for category in config.categories:
        floor = float(config.category_floors.get(category, 0.0))
        value = float(evaluation.per_category[category].f1)
        if value < floor:
            category_failures.append(
                CategoryFailure(
                    category=category,
                    metric="f1",
                    value=round(value, 6),
                    floor=floor,
                )
            )

    missing_passed = len(evaluation.missing_categories) == 0
    micro_passed = float(evaluation.overall.f1) >= config.micro_f1_threshold
    category_passed = len(category_failures) == 0
    api_error_count = len(evaluation.api_errors)
    api_error_total = evaluation.unique_sentences
    api_error_rate = (
        (api_error_count / api_error_total) if api_error_total > 0 else 0.0
    )
    if config.timeout_policy == "continue":
        api_error_passed = api_error_rate <= config.api_error_rate_threshold
    else:
        api_error_passed = api_error_count == 0
    regression_passed = regression.passed

    warnings = list(evaluation.warnings)
    if (
        config.timeout_policy == "continue"
        and api_error_count > 0
        and not api_error_passed
    ):
        warnings.append(
            "API error-rate gate failed: "
            f"{api_error_count}/{api_error_total} "
            f"({api_error_rate:.2%}) > "
            f"{config.api_error_rate_threshold:.2%}."
        )
    if regression.warning:
        warnings.append(regression.warning)

    return GateResult(
        overall_passed=all(
            [
                missing_passed,
                micro_passed,
                category_passed,
                api_error_passed,
                regression_passed,
            ]
        ),
        missing_categories_passed=missing_passed,
        missing_categories=evaluation.missing_categories,
        micro_f1_passed=micro_passed,
        micro_f1_threshold=config.micro_f1_threshold,
        category_floor_passed=category_passed,
        category_failures=category_failures,
        api_error_passed=api_error_passed,
        api_error_count=api_error_count,
        api_error_rate=round(api_error_rate, 6),
        api_error_rate_threshold=config.api_error_rate_threshold,
        api_error_total=api_error_total,
        timeout_policy=config.timeout_policy,
        regression=regression,
        warnings=warnings,
    )


def render_terminal_summary(
    config: GateConfig,
    evaluation: EvaluationResult,
    gates: GateResult,
    report_json_path: Path,
    result_json_path: Path,
) -> str:
    """Готовит текстовый summary для терминала pytest."""
    lines = [
        "",
        "Entity-Level Quality Gate",
        f"Dataset: {evaluation.dataset_meta.path}",
        f"Dataset sha256: {evaluation.dataset_meta.sha256}",
        f"Cases: {evaluation.cases_total}, unique_texts: {evaluation.unique_sentences}",
        (
            "OVERALL: "
            f"tp={evaluation.overall.tp} fp={evaluation.overall.fp} fn={evaluation.overall.fn} "
            f"precision={evaluation.overall.precision:.3f} "
            f"recall={evaluation.overall.recall:.3f} f1={evaluation.overall.f1:.3f} "
            f"(threshold={config.micro_f1_threshold:.2f})"
        ),
        (
            "API errors: "
            f"{gates.api_error_count}/{gates.api_error_total} "
            f"({gates.api_error_rate:.2%}) "
            f"(policy={config.timeout_policy}, "
            f"threshold={config.api_error_rate_threshold:.2%})"
        ),
        f"Gates passed: {gates.overall_passed}",
    ]

    if gates.warnings:
        lines.append("Warnings:")
        lines.extend(f"- {item}" for item in gates.warnings)

    lines.extend(["Per-category:", *evaluation.report_lines])
    lines.append(f"Report: {report_json_path.resolve()}")
    lines.append(f"Result: {result_json_path.resolve()}")
    lines.append("")
    return "\n".join(lines)


def _extract_baseline_f1(payload: dict[str, object]) -> float | None:
    raw = payload.get("micro_f1")
    if isinstance(raw, (int, float)):
        return float(raw)

    overall_raw = payload.get("overall")
    if isinstance(overall_raw, dict):
        f1_raw = overall_raw.get("f1")
        if isinstance(f1_raw, (int, float)):
            return float(f1_raw)

    return None


def _extract_baseline_dataset_sha(payload: dict[str, object]) -> str | None:
    dataset_sha = payload.get("dataset_sha256")
    if isinstance(dataset_sha, str) and dataset_sha:
        return dataset_sha

    dataset = payload.get("dataset")
    if isinstance(dataset, dict):
        sha = dataset.get("sha256")
        if isinstance(sha, str) and sha:
            return sha

    return None
