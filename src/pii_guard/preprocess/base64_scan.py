"""Поиск и декодирование Base64-подстрок с PII (перед PII-анализом).

Найти подстроки, похожие на Base64,
декодировать их, и подать декодированный текст в PII-распознаватели.
Условия:
* длина ≥ ``min_len`` и кратна 4 (валидный паддинг);
* алфавит строго ``[A-Za-z0-9+/]`` + ``=`` в конце;
* декодируется в валидный UTF-8;
* доля печатаемых символов в декодированном ≥ ``min_printable_ratio``.
Эти условия отсекают обычные слова/идентификаторы, которые лишь внешне похожи
на Base64.
"""

from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass

_B64_RE = re.compile(r"[A-Za-z0-9+/]{12,}={0,2}")


@dataclass
class Base64Span:
    start: int
    end: int
    decoded: str


def _printable_ratio(text: str) -> float:
    if not text:
        return 0.0
    printable = sum(1 for ch in text if ch.isprintable() or ch in " \t\n")
    return printable / len(text)


def find_base64_spans(
    text: str,
    *,
    min_len: int = 16,
    min_printable_ratio: float = 0.8,
) -> list[Base64Span]:
    """Находит Base64-подстроки, декодируемые в печатаемый UTF-8.

    Возвращает список :class:`Base64Span` со спаном в координатах *text* и
    декодированной строкой.
    """
    spans: list[Base64Span] = []
    for m in _B64_RE.finditer(text):
        token = m.group(0)
        if len(token) < min_len or len(token) % 4 != 0:
            continue
        try:
            raw = base64.b64decode(token, validate=True)
        except (binascii.Error, ValueError):
            continue
        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if not decoded.strip():
            continue
        if _printable_ratio(decoded) < min_printable_ratio:
            continue
        spans.append(Base64Span(start=m.start(), end=m.end(), decoded=decoded))
    return spans
