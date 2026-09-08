"""Препроцессоры обхода PII-детекции (включаемые/отключаемые модули).

Каждый препроцессор реализует один способ «раскрытия» скрытой PII перед
NER-анализом и управляется отдельным флагом в :class:`~pii_guard.config.Config`:

* ``translit``     — обратная транслитерация русского текста, записанного
                     латиницей (флаг ``PII_GUARD_ENABLE_TRANSLIT``).
* ``en_numbers``   — нормализация английских числительных словами → цифры
                     (флаг ``PII_GUARD_ENABLE_EN_NUMBERS``).
* ``base64_scan``  — поиск и декодирование Base64-подстрок с PII
                     (флаг ``PII_GUARD_ENABLE_BASE64``).

Препроцессоры самодостаточны и не зависят от модели NER, поэтому покрываются
оффлайн юнит-тестами.
"""

from __future__ import annotations

from .base64_scan import Base64Span, find_base64_spans
from .en_numbers import normalize_english_numbers
from .lang_guard import is_confident_foreign
from .lang_guard import warmup as langguard_warmup
from .translit import (
    SpanMap,
    looks_like_translit,
    transliterate_to_cyrillic,
)

__all__ = [
    "SpanMap",
    "looks_like_translit",
    "transliterate_to_cyrillic",
    "normalize_english_numbers",
    "Base64Span",
    "find_base64_spans",
    "is_confident_foreign",
    "langguard_warmup",
]
