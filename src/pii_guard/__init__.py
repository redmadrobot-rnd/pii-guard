"""
pii-guard — PII detection, anonymisation and reversible pseudonymisation for
Russian text.

Layout
------
``framework/``   generic infrastructure, no entity knowledge
``entities/``    one self-contained file per entity type, auto-discovered
``preprocess/``  bypass preprocessors (translit, base64, English numerals)
``ner/``         transformer NER recogniser (optional extra ``ner``)
``detect.py``    rules-branch entry point
``engine.py``    both branches + conflict resolution
``pseudonymize.py``  XML-tag pseudonymisation and de-anonymisation
``operators.py``     output modes: mask / tag / pseudonymize

The heavy names (:class:`Anonymizer`, :class:`Engine`) are resolved lazily, so
``import pii_guard`` stays cheap and pulls neither spaCy nor torch.  Importing a
submodule directly -- ``from pii_guard.entities.inn import InnClassifier`` -- costs
nothing beyond that submodule's own imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

_LAZY: dict[str, tuple[str, str]] = {
    "Anonymizer": ("pii_guard.anonymizer", "Anonymizer"),
    "AnonymizeResult": ("pii_guard.anonymizer", "AnonymizeResult"),
    "Config": ("pii_guard.config", "Config"),
    "Engine": ("pii_guard.engine", "Engine"),
    "Mode": ("pii_guard.operators", "Mode"),
    "ModelUnavailableError": ("pii_guard.ner.recognizer", "ModelUnavailableError"),
}

__all__ = [
    "Anonymizer",
    "AnonymizeResult",
    "Config",
    "Engine",
    "Mode",
    "ModelUnavailableError",
    "__version__",
]

if TYPE_CHECKING:  # pragma: no cover -- import-time typing only
    from pii_guard.anonymizer import Anonymizer, AnonymizeResult
    from pii_guard.config import Config
    from pii_guard.engine import Engine
    from pii_guard.ner.recognizer import ModelUnavailableError
    from pii_guard.operators import Mode


def __getattr__(name: str) -> Any:
    """Resolve heavy public names on first attribute access (PEP 562)."""
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value  # cache, so the next access is a plain lookup
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
