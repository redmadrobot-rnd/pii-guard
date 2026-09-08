"""Типы и метрики entity-level quality gate."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .iob2_parser import GoldEntity


@dataclass
class CategoryStats:
    tp: int = 0
    fp: int = 0
    fn: int = 0


@dataclass
class CategoryMetrics:
    support: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "support": self.support,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass
class OverallMetrics:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass
class DatasetMeta:
    path: str
    sha256: str
    size_bytes: int

    def as_dict(self) -> dict[str, str | int]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass
class PredEntity:
    entity_type: str
    start: int
    end: int
    score: float | None


@dataclass
class EntityError:
    error_type: str
    row_id: str
    source: str
    text: str
    label: str
    gold_label: str
    pred_label: str
    start: int
    end: int
    score: float | None
    span_text: str
    overlap_gold_labels: str

    def as_dict(self) -> dict[str, str | int | float | None]:
        return {
            "error_type": self.error_type,
            "row_id": self.row_id,
            "source": self.source,
            "text": self.text,
            "label": self.label,
            "gold_label": self.gold_label,
            "pred_label": self.pred_label,
            "start": self.start,
            "end": self.end,
            "score": self.score,
            "span_text": self.span_text,
            "overlap_gold_labels": self.overlap_gold_labels,
        }


@dataclass
class ApiError:
    sentence: str
    error: str

    def as_dict(self) -> dict[str, str]:
        return {
            "sentence": self.sentence,
            "error": self.error,
        }


@dataclass
class EvaluationResult:
    generated_at_utc: datetime
    dataset_meta: DatasetMeta
    overall: OverallMetrics
    per_category: dict[str, CategoryMetrics]
    missing_categories: list[str]
    unmapped_gold_labels: list[str]
    entity_errors: list[EntityError]
    api_errors: list[ApiError]
    cases_total: int
    unique_sentences: int
    report_lines: list[str]
    warnings: list[str]


def spans_overlap(start1: int, end1: int, start2: int, end2: int) -> bool:
    """Проверяет пересечение двух спанов."""
    return start1 < end2 and start2 < end1


def match_entities(
    gold_entities: list[GoldEntity],
    predicted_entities: list[PredEntity],
    api_to_gold: dict[str, set[str]],
) -> tuple[list[bool], list[bool]]:
    """Матчит предсказания и gold по label+пересечению.

    Одно предсказание может покрывать несколько gold-сущностей той же категории
    (например, когда в gold документ разбит на части, а API вернул единый спан).
    """
    gold_matched = [False] * len(gold_entities)
    pred_matched = [False] * len(predicted_entities)

    for pred_idx, pred in enumerate(predicted_entities):
        pred_gold_labels = api_to_gold.get(pred.entity_type, {pred.entity_type})
        for gold_idx, gold in enumerate(gold_entities):
            if gold_matched[gold_idx]:
                continue
            if gold.gold_label not in pred_gold_labels:
                continue
            if spans_overlap(pred.start, pred.end, gold.start, gold.end):
                gold_matched[gold_idx] = True
                pred_matched[pred_idx] = True

    return gold_matched, pred_matched


def score_from_stats(stats: CategoryStats) -> tuple[float, float, float]:
    """Считает precision/recall/f1 для категории."""
    precision = stats.tp / (stats.tp + stats.fp) if (stats.tp + stats.fp) else 0.0
    recall = stats.tp / (stats.tp + stats.fn) if (stats.tp + stats.fn) else 0.0
    f1 = (
        (2.0 * precision * recall / (precision + recall))
        if (precision + recall)
        else 0.0
    )
    return precision, recall, f1


def build_dataset_meta(path: Path) -> DatasetMeta:
    """Формирует метаинформацию о датасете для отчета и regression-check."""
    stat = path.stat()
    return DatasetMeta(
        path=str(path.resolve()),
        sha256=_sha256_file(path),
        size_bytes=stat.st_size,
    )


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        if hasattr(hashlib, "file_digest"):
            return hashlib.file_digest(handle, "sha256").hexdigest()  # type: ignore[attr-defined]
        digest = hashlib.sha256()
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        return digest.hexdigest()
