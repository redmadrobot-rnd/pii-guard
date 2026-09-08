"""HTTP-клиент pii-guard для quality gate.

Говорит с `pii_guard_server`: `POST /anonymize` со списком `texts`. До переноса
клиент обращался к другим путям и посылал одно поле `text`, поэтому транспорт
`PII_GATE_TRANSPORT=http` не работал вообще — сервер отвечал 404.

Ответ приводится к той же форме, что отдаёт `LocalEngineClient`, поэтому
`utils.evaluate` одинаково работает с обоими транспортами.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from .config import GateConfig


class TimeoutOrGatewayError(RuntimeError):
    """Ошибка таймаута или gateway-класса при запросе к Guard API."""


ANONYMIZE_PATH = "/anonymize"


@dataclass
class GuardApiClient:
    base_url: str
    api_key: str | None
    timeout: float
    anonymize_path: str = ANONYMIZE_PATH

    def __repr__(self) -> str:
        masked_key = "***" if self.api_key else None
        return (
            "GuardApiClient("
            f"base_url={self.base_url!r}, "
            f"api_key={masked_key!r}, "
            f"timeout={self.timeout!r}, "
            f"anonymize_path={self.anonymize_path!r})"
        )

    def anonymize(self, text: str) -> dict[str, object]:
        # mode="tag" because the gate scores spans, not the rendered output, and
        # tag mode is the only one that needs no mapping bookkeeping.
        response = request_json(
            f"{self.base_url}{self.anonymize_path}",
            payload={"texts": [text], "mode": "tag"},
            timeout=self.timeout,
            api_key=self.api_key,
        )
        items = response.get("items")
        if not isinstance(items, list) or not items:
            raise RuntimeError(f"pii-guard returned no items for a single text: {response!r}")
        item = items[0]
        return {
            "status": True,
            "anonymized_text": item.get("text"),
            "anonymized_text_tags": item.get("text"),
            "normalized_text": item.get("normalized_text"),
            "entities": item.get("entities") or [],
        }


def request_json(
    url: str,
    payload: dict[str, object] | None,
    timeout: float,
    api_key: str | None = None,
) -> dict[str, object]:
    """Выполняет HTTP-запрос и возвращает JSON как словарь."""
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        url=url,
        data=body,
        headers=headers,
        method="POST" if body is not None else "GET",
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        if not raw:
            return {}
        decoded = json.loads(raw)
        if isinstance(decoded, dict):
            return decoded
        return {}


def create_guard_api_client(config: GateConfig) -> GuardApiClient:
    """Проверяет доступность API и валидность ключа `GUARD_API_KEY`."""
    request_json(f"{config.api_url}/health", payload=None, timeout=config.api_timeout)

    anonymize_path = _resolve_anonymize_path(config)

    return GuardApiClient(
        base_url=config.api_url,
        api_key=config.api_key,
        timeout=config.api_timeout,
        anonymize_path=anonymize_path,
    )


def anonymize_with_retry(
    client: GuardApiClient, text: str, retries: int
) -> dict[str, object]:
    """Вызывает PII endpoint с повторными попытками при сетевых ошибках."""
    attempts = max(1, retries + 1)
    last_error: Exception | None = None

    for _ in range(attempts):
        try:
            return client.anonymize(text)
        except urllib.error.HTTPError as exc:
            if not _is_retriable_http_error(exc):
                raise
            last_error = exc
        except Exception as exc:  # pragma: no cover - runtime network behavior
            if not _is_timeout_error(exc):
                raise
            last_error = exc

    raise TimeoutOrGatewayError(
        "Guard API request failed with timeout/gateway while processing dataset text "
        f"after {attempts} attempts. Last error: {last_error}"
    )


def _is_retriable_http_error(exc: urllib.error.HTTPError) -> bool:
    return exc.code in {429, 502, 503, 504}


def _is_timeout_error(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, urllib.error.URLError):
        reason = getattr(exc, "reason", None)
        return isinstance(reason, TimeoutError) or "timed out" in str(reason).lower()
    return False


def _resolve_anonymize_path(config: GateConfig) -> str:
    """Probe the endpoint so a misconfigured URL or key fails before the run.

    Only one path is tried: `pii_guard_server` exposes exactly `/anonymize`. The
    previous version probed two endpoints of the closed product, which meant the
    real failure mode -- "you pointed me at the wrong service" -- surfaced as
    "no supported endpoint" after two 404s.
    """
    path = ANONYMIZE_PATH
    try:
        request_json(
            f"{config.api_url}{path}",
            payload={"texts": ["пробный текст"], "mode": "tag"},
            timeout=config.api_timeout,
            api_key=config.api_key,
        )
        return path
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            if config.api_key:
                raise PermissionError(
                    f"pii-guard отклонил GUARD_API_KEY для {path}. "
                    "Проверьте значение ключа."
                ) from exc
            raise PermissionError(
                f"pii-guard требует GUARD_API_KEY для {path} "
                "(на сервере задан PII_GUARD_API_KEYS)."
            ) from exc
        if exc.code == 404:
            raise RuntimeError(
                f"{config.api_url}{path} отвечает 404 — по этому адресу не "
                "pii-guard. Проверьте GUARD_API_URL (порт по умолчанию 8080)."
            ) from exc
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"pii-guard вернул HTTP {exc.code} для {path}: {detail}"
        ) from exc
