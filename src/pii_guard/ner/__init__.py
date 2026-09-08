"""Transformer NER branch — detects the types no regex can find in free text.

Requires the optional extra ``ner`` (torch + transformers) and model weights.
Without it the engine runs the rules branch alone; see
:class:`pii_guard.engine.Engine`.

Nothing is imported eagerly here: ``import pii_guard.ner`` must stay cheap so
that a rules-only installation never touches torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

_LAZY: dict[str, tuple[str, str]] = {
    "NERAdapter": ("pii_guard.ner.adapter", "NERAdapter"),
    "TransformerNERConfig": ("pii_guard.ner.recognizer", "TransformerNERConfig"),
    "TransformerNERRecognizer": ("pii_guard.ner.recognizer", "TransformerNERRecognizer"),
    "ModelUnavailableError": ("pii_guard.ner.recognizer", "ModelUnavailableError"),
    "resolve_device": ("pii_guard.ner.recognizer", "resolve_device"),
}

__all__ = list(_LAZY)

if TYPE_CHECKING:  # pragma: no cover -- import-time typing only
    from pii_guard.ner.adapter import NERAdapter
    from pii_guard.ner.recognizer import (
        ModelUnavailableError,
        TransformerNERConfig,
        TransformerNERRecognizer,
        resolve_device,
    )


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
