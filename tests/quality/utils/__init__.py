"""Публичное API quality gate для PII."""

from .api_client import (
    GuardApiClient,
    TimeoutOrGatewayError,
    create_guard_api_client,
)
from .config import CORE_CATEGORIES, GateConfig, load_gate_config
from .evaluate import (
    GateResult,
    RegressionResult,
    evaluate_baseline_regression,
    evaluate_entity_quality,
    evaluate_gates,
    render_terminal_summary,
)
from .local_client import LocalEngineClient, create_local_client
from .metrics import (
    ApiError,
    CategoryMetrics,
    DatasetMeta,
    EntityError,
    EvaluationResult,
    OverallMetrics,
)
from .reports import ReportPaths, write_reports
from .transport import create_client, transport_name

__all__ = [
    "CORE_CATEGORIES",
    "GateConfig",
    "GuardApiClient",
    "TimeoutOrGatewayError",
    "EvaluationResult",
    "OverallMetrics",
    "CategoryMetrics",
    "DatasetMeta",
    "ApiError",
    "EntityError",
    "GateResult",
    "RegressionResult",
    "ReportPaths",
    "LocalEngineClient",
    "load_gate_config",
    "create_client",
    "create_guard_api_client",
    "create_local_client",
    "transport_name",
    "evaluate_entity_quality",
    "evaluate_baseline_regression",
    "evaluate_gates",
    "render_terminal_summary",
    "write_reports",
]
