"""Output modes — how a detected span is rendered in the returned text.

All three take the *normalised* text plus the entity list the engine produced, so
the spans line up. See the note on span coordinates in
:class:`pii_guard.anonymizer.AnonymizeResult`.

============  ==============================================  ==========
mode          example                                         reversible
============  ==============================================  ==========
``mask``      ``Иван Петров`` -> ``***********``               no
``tag``       ``Иван Петров`` -> ``[PERSON]``                   no
``pseudonymize``  -> ``<PII type="PERSON" gender="male" id="1" />``  yes
============  ==============================================  ==========

``mask`` is produced by the Presidio ``AnonymizerEngine`` inside
:meth:`pii_guard.engine.Engine.anonymize`; the other two are built here.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

Mode = Literal["mask", "tag", "pseudonymize"]

MODES: tuple[Mode, ...] = ("mask", "tag", "pseudonymize")


def build_tagged_text(text: str, entities: Iterable[dict]) -> str:
    """Replace every entity span with ``[ENTITY_TYPE]``.

    Spans are substituted right-to-left so that earlier offsets stay valid.
    """
    tagged = text
    for entity in sorted(entities, key=lambda e: e.get("start", 0), reverse=True):
        start = entity.get("start", 0)
        end = entity.get("end", 0)
        tag = entity.get("entity_type") or entity.get("type") or "ENTITY"
        tagged = tagged[:start] + f"[{tag}]" + tagged[end:]
    return tagged


def validate_mode(mode: str) -> Mode:
    """Return *mode* unchanged, or raise with the list of valid values."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {', '.join(MODES)}")
    return mode  # type: ignore[return-value]


__all__ = ["MODES", "Mode", "build_tagged_text", "validate_mode"]
