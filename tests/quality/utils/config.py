"""Конфигурация entity-level quality gate для PII."""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

TimeoutPolicy = Literal["continue", "fail"]

CORE_CATEGORIES: tuple[str, ...] = tuple(
    sorted(
        {
            "PERSON",
            "PHONE_NUMBER",
            "EMAIL_ADDRESS",
            "LOCATION",
            "DATE_TIME",
            "SNILS",
            "INN",
            "PASSPORT",
            "DRIVER_LICENSE",
            "CREDIT_CARD",
            "OMS",
            "URL",
            "IP_ADDRESS",
            "MILITARY_ID",
            "BIRTH_CERTIFICATE",
            "BANK_ACCOUNT",
            "POSTAL_CODE",
            "TELEGRAM",
            "BIK",
        }
    )
)


@dataclass
class GateConfig:
    api_url: str
    api_key: str | None
    api_timeout: float
    api_retries: int
    api_error_rate_threshold: float
    micro_f1_threshold: float
    category_floors: dict[str, float]
    categories: tuple[str, ...]
    log_progress: bool
    timeout_policy: TimeoutPolicy
    report_root: Path
    enable_plots: bool
    baseline_result_path: Path | None
    max_f1_drop: float
    dataset_path: Path
    require_same_dataset_for_regression: bool


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_api_key() -> str | None:
    value = os.getenv("GUARD_API_KEY", "").strip()
    return value or None


def _parse_category_floors(categories: tuple[str, ...]) -> dict[str, float]:
    # Все категории равноправны и используют единый высокий порог по умолчанию.
    default_floor = float(os.getenv("PII_DEFAULT_CATEGORY_FLOOR", "0.85"))
    floors = dict.fromkeys(categories, default_floor)

    raw_json = os.getenv("PII_CATEGORY_FLOOR_JSON", "")
    if raw_json:
        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            warnings.warn(
                "PII_CATEGORY_FLOOR_JSON содержит невалидный JSON, "
                f"используются default floors. Ошибка: {exc}",
                stacklevel=2,
            )
            parsed = {}
        if isinstance(parsed, dict):
            for key, value in parsed.items():
                if key in floors:
                    try:
                        floors[key] = float(value)
                    except (TypeError, ValueError):
                        continue
        else:
            warnings.warn(
                "PII_CATEGORY_FLOOR_JSON должен быть JSON-объектом "
                "(dict), используются default floors.",
                stacklevel=2,
            )

    return floors


def _parse_timeout_policy() -> TimeoutPolicy:
    raw = os.getenv("PII_TIMEOUT_POLICY", "continue").strip().lower()
    if raw == "fail":
        return "fail"
    return "continue"


def load_gate_config() -> GateConfig:
    """Загружает конфигурацию quality gate из переменных окружения.

    Parameters:
        Отсутствуют. Все параметры читаются из env-переменных.

    Returns:
        GateConfig: Полная конфигурация запуска quality gate.
    """
    base_dir = Path(__file__).resolve().parent.parent

    dataset_path = Path(
        os.getenv(
            "PII_DATASET",
            str(base_dir / "test_data" / "test.xlsx"),
        )
    )
    report_root = Path(
        os.getenv(
            "PII_REPORT_DIR",
            str(base_dir / "reports"),
        )
    )
    baseline_env = os.getenv("PII_BASELINE_RESULT", "").strip()
    # Default to the committed baseline rather than a file under reports/: that
    # directory is gitignored, so a baseline living there would never reach CI and
    # the regression gate would silently do nothing on every fresh clone.
    baseline_path: Path | None = (
        Path(baseline_env) if baseline_env else base_dir / "baseline_result.json"
    )

    categories = CORE_CATEGORIES
    return GateConfig(
        api_url=os.getenv("GUARD_API_URL", "http://localhost:8080").rstrip("/"),
        api_key=_parse_api_key(),
        api_timeout=float(os.getenv("GUARD_API_TIMEOUT", "20")),
        api_retries=int(os.getenv("GUARD_API_RETRIES", "1")),
        api_error_rate_threshold=float(os.getenv("PII_API_ERROR_RATE_THRESHOLD", "0.05")),
        micro_f1_threshold=float(os.getenv("PII_GATE_THRESHOLD", "0.95")),
        category_floors=_parse_category_floors(categories),
        categories=categories,
        log_progress=_env_bool("PII_LOG_PROGRESS", True),
        timeout_policy=_parse_timeout_policy(),
        report_root=report_root,
        enable_plots=_env_bool("PII_ENABLE_PLOTS", True),
        baseline_result_path=baseline_path,
        max_f1_drop=float(os.getenv("PII_MAX_F1_DROP", "0.02")),
        dataset_path=dataset_path,
        require_same_dataset_for_regression=_env_bool("PII_REQUIRE_SAME_DATASET", True),
    )
