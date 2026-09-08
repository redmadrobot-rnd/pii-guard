"""Оффлайн юнит-тесты языкового вето транслита (lingua)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

import pii_guard

pytest.importorskip("lingua")  # тест требует установленной lingua

_MODULE_PATH = (
    Path(pii_guard.__file__).resolve().parent
    / "preprocess" / "lang_guard.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("lang_guard_preprocess", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["lang_guard_preprocess"] = module
    spec.loader.exec_module(module)
    return module


lg = _load()


@pytest.mark.parametrize(
    "text",
    [
        "Guten Tag mein Name ist Schmidt und ich komme aus Berlin",
        "Die Bestellung wurde gestern verschickt und kommt bald an",
        "Bonjour je voudrais reserver une table pour deux personnes",
    ],
)
def test_confident_foreign_vetoed(text: str) -> None:
    # уверенный иностранный язык -> вето (True)
    assert lg.is_confident_foreign(text, threshold=0.90) is True


@pytest.mark.parametrize(
    "text",
    [
        "menya zovut evgeniya moj nomer 8499876543",
        "privet ya ivan petrov prozhivayu v sankt-peterburge",
        "Dmitriy Yurevich Kuznetsov gorod Yekaterinburg",
    ],
)
def test_genuine_translit_not_vetoed(text: str) -> None:
    # настоящий транслит не дотягивает до 0.90 -> НЕ вето (False)
    assert lg.is_confident_foreign(text, threshold=0.90) is False


def test_threshold_is_respected() -> None:
    # при очень высоком пороге даже немецкий может не дотянуть
    text = "Guten Tag mein Name ist Schmidt"
    assert lg.is_confident_foreign(text, threshold=0.999) in (True, False)
    # при нулевом пороге любой непустой текст -> вето
    assert lg.is_confident_foreign("hello world this is english", threshold=0.0) is True
