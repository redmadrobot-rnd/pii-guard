"""Языковой фильтр для транслит-препроцессора (на базе lingua).

Назначение
Эвристическая функция ``looks_like_translit`` иногда ложно срабатывает на
иностранном латинском тексте (немецкий, французский и т.п.), который случайно
содержит «русские» диграфы. Этот модуль — **вторичное вето**: если детектор
языка уверенно (≥ ``threshold``) относит текст к конкретному НЕ-русскому языку,
транслит-проход отменяется.
"""

from __future__ import annotations

import threading

_LANG_NAMES = (
    "RUSSIAN", "ENGLISH", "GERMAN", "SPANISH", "FRENCH",
    "ITALIAN", "TURKISH", "POLISH", "CZECH",
)

_detector = None
_lock = threading.Lock()


def _build_detector():
    global _detector
    if _detector is not None:
        return _detector
    with _lock:
        if _detector is not None:
            return _detector
        try:
            from lingua import Language, LanguageDetectorBuilder
        except ImportError:
            return None
        languages = [getattr(Language, name) for name in _LANG_NAMES
                     if hasattr(Language, name)]
        _detector = LanguageDetectorBuilder.from_languages(*languages).build()
        return _detector


def warmup() -> None:
    """Прогревает модели языков (разовая загрузка ~1.4 с). Вызывать на старте."""
    det = _build_detector()
    if det is not None:
        det.compute_language_confidence_values("warmup ok")


def is_confident_foreign(text: str, threshold: float = 0.90) -> bool:
    """``True``, если *text* уверенно (≥ *threshold*) определён как НЕ-русский язык.

    Если библиотека недоступна — возвращает ``False`` (вето отключено), чтобы не
    ломать пайплайн.
    """
    det = _build_detector()
    if det is None:
        return False
    try:
        from lingua import Language
    except ImportError:
        return False
    values = det.compute_language_confidence_values(text)
    if not values:
        return False
    top = values[0]
    return top.language != Language.RUSSIAN and top.value >= threshold
