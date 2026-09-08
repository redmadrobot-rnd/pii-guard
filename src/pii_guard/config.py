"""Configuration for the detection engine.

Resolution order, highest priority first:

1. an explicit value passed to :class:`Config`
2. an environment variable ``PII_GUARD_*``
3. a YAML file (``PII_GUARD_CONFIG`` or ``./config.yaml``)
4. the built-in default

The YAML layer is inherited from the service this code was extracted from, where
operators shipped a mounted ``config.yaml``. It stays because that is still the
convenient way to tune thresholds in a container without rebuilding the image.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from pii_guard.models import DEFAULT_NER_MODEL, DEFAULT_NER_REVISION, DEFAULT_SPACY_MODEL

logger = logging.getLogger("pii_guard.config")

_ENV_PREFIX = "PII_GUARD_"

# Entity types masked when the caller supplies no allowlist.
DEFAULT_ENTITIES: tuple[str, ...] = (
    "PERSON", "LOCATION", "EMAIL_ADDRESS", "PHONE_NUMBER", "URL",
    "IP_ADDRESS", "IP_PORT", "DATE_TIME", "PASSPORT", "CREDIT_CARD",
    "DRIVER_LICENSE", "INN", "SNILS", "MILITARY_ID", "BIRTH_CERTIFICATE",
    "OMS", "BANK_ACCOUNT", "POSTAL_CODE", "TELEGRAM", "BIK",
)

# Entity types that only the transformer NER branch can find: DEFAULT_ENTITIES
# minus everything registered under entities/. Listed so the engine can warn
# instead of silently returning nothing for them.
NER_ONLY_ENTITIES: frozenset[str] = frozenset(
    {"PERSON", "LOCATION", "PHONE_NUMBER", "URL"}
)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_yaml() -> dict[str, Any]:
    explicit = os.getenv(f"{_ENV_PREFIX}CONFIG", "")
    path = Path(explicit) if explicit else Path("config.yaml")
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


@dataclass(slots=True)
class Config:
    """Engine settings. Every field can be overridden by ``PII_GUARD_<FIELD>``."""

    # ── Language / models ─────────────────────────────────────────
    language: str = "ru"
    spacy_model: str = DEFAULT_SPACY_MODEL
    ner_model: str = DEFAULT_NER_MODEL
    ner_revision: str | None = DEFAULT_NER_REVISION
    ner_disabled: bool = False
    device: str | None = None  # None -> resolve_device(): cuda if present else cpu

    # ── NER inference ─────────────────────────────────────────────
    ner_max_length: int = 512
    ner_stride: int = 128
    # Fixed score reported for every NER span. Not a threshold: the branch takes
    # an argmax over the label logits and emits whatever wins, so there is nothing
    # to compare a cutoff against. The value matters because score breaks ties in
    # `_resolve_by_score` when two overlapping spans have no other ordering --
    # 0.70 keeps NER below the checksum-backed rules, which score 0.75 and up.
    ner_score: float = 0.70

    # ── Masking (mode="mask") ─────────────────────────────────────
    mask_char: str = "*"
    # Presidio masks exactly this many characters of a span and returns the rest
    # verbatim, so any finite default silently leaks the tail of a longer span.
    # 99 covered every document number and looked safe, but a base64 blob or a
    # merged address span runs past it: a 248-character blob holding an ИНН came
    # back with 149 characters of encoded PII in the clear. A number no span can
    # reach means "mask all of it"; lower it deliberately if partial masking (last
    # four digits of a card, say) is what you actually want.
    mask_chars_to_mask: int = 1_000_000_000
    mask_from_end: bool = False

    # ── Bypass preprocessors ──────────────────────────────────────
    enable_translit: bool = True
    enable_en_numbers: bool = True
    enable_base64: bool = True
    translit_lang_threshold: float = 0.90

    # ── Policy ────────────────────────────────────────────────────
    entities: tuple[str, ...] = DEFAULT_ENTITIES

    # ── Limits (enforced by the server layer, not the library) ────
    max_text_chars: int = 100_000
    max_texts: int = 64


    @classmethod
    def load(cls, **overrides: Any) -> Config:
        """Build a config from explicit overrides, environment and YAML."""
        yaml_data = _load_yaml()
        kwargs: dict[str, Any] = {}

        for f in cls.__dataclass_fields__.values():
            if f.name.startswith("_"):
                continue
            if f.name in overrides:
                kwargs[f.name] = overrides[f.name]
                continue

            raw = os.getenv(f"{_ENV_PREFIX}{f.name.upper()}")
            if raw is None:
                raw = yaml_data.get(f.name)
            if raw is None:
                continue
            kwargs[f.name] = _coerce(f.name, raw)

        unknown = set(overrides) - set(cls.__dataclass_fields__)
        if unknown:
            raise TypeError(f"unknown config field(s): {', '.join(sorted(unknown))}")

        return cls(**kwargs)

    @property
    def ner_enabled(self) -> bool:
        """False when the NER branch is switched off or has no model to load."""
        return not self.ner_disabled and bool(self.ner_model)


_BOOL_FIELDS = {
    "ner_disabled", "mask_from_end",
    "enable_translit", "enable_en_numbers", "enable_base64",
}
_INT_FIELDS = {
    "ner_max_length", "ner_stride", "mask_chars_to_mask",
    "max_text_chars", "max_texts",
}
_FLOAT_FIELDS = {"ner_score", "translit_lang_threshold"}


def _coerce(name: str, raw: Any) -> Any:
    """Cast a value coming from the environment (always a string) or YAML."""
    if name in _BOOL_FIELDS:
        return _as_bool(raw)
    if name in _INT_FIELDS:
        return int(raw)
    if name in _FLOAT_FIELDS:
        return float(raw)
    if name == "entities":
        items = raw.split(",") if isinstance(raw, str) else list(raw)
        parsed = tuple(dict.fromkeys(str(i).strip().upper() for i in items if str(i).strip()))
        if not parsed:
            # Empty means "not configured", never "detect nothing" -- the
            # `VAR=${VAR}` compose form yields an empty string for an unset shell
            # variable, and an empty allowlist returns every text untouched, a
            # fail-open that reads as "no PII here". Callers wanting no detection
            # pass `entities=[]` per call, where it is explicit.
            logger.warning(
                "event=entities_empty action=using_defaults var=%sENTITIES", _ENV_PREFIX
            )
            return DEFAULT_ENTITIES
        return parsed
    if name in {"ner_revision", "device"}:
        text = str(raw).strip()
        return text or None
    return str(raw)


__all__ = ["Config", "DEFAULT_ENTITIES", "NER_ONLY_ENTITIES"]
