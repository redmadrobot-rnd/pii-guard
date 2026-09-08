"""In-process stand-in for :class:`GuardApiClient`.

The quality gate was written against the HTTP endpoint of the service this code
was extracted from, where the model lived behind a queue and could only be reached
over the network. Here the engine is a library, so the gate can call it directly:
no container to start, no port to guess, and one less moving part between the
dataset and the numbers.

Duck-typed against ``GuardApiClient``: the gate only ever calls ``.anonymize(text)``
and expects the ``/anonymize`` response shape, so both transports are
interchangeable. Set ``PII_GATE_TRANSPORT=http`` to go back over the network --
useful for testing a built image rather than the working tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover -- import-time typing only
    from pii_guard.engine import Engine

    from .config import GateConfig


@dataclass
class LocalEngineClient:
    """Calls :class:`pii_guard.engine.Engine` in-process.

    Returns exactly the keys the HTTP endpoint returned, so
    ``utils.evaluate`` needs no changes.
    """

    engine: Engine
    entities: tuple[str, ...] | None = None
    _allowed: set[str] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._allowed = (
            {e.strip().upper() for e in self.entities if str(e).strip()}
            if self.entities is not None
            else None
        )

    def __repr__(self) -> str:
        return (
            "LocalEngineClient("
            f"ner_enabled={self.engine.ner_enabled}, "
            f"ner_model={self.engine.config.ner_model!r}, "
            f"spacy_model={self.engine.config.spacy_model!r})"
        )

    def anonymize(self, text: str) -> dict[str, Any]:
        from pii_guard.operators import build_tagged_text

        anonymized_text, entities, normalized_text = self.engine.anonymize(
            text, allowed_entities=self._allowed
        )
        return {
            "status": True,
            "anonymized_text": anonymized_text,
            "anonymized_text_tags": build_tagged_text(normalized_text, entities),
            "normalized_text": normalized_text,
            "entities": entities,
        }


def create_local_client(config: GateConfig) -> LocalEngineClient:
    """Build an in-process client, loading the engine from the ambient config.

    Entity allowlist is left unset on purpose: the gate scores everything the
    detector finds against the gold labels, and narrowing it here would silently
    depress recall for the excluded types.
    """
    from pii_guard.config import Config
    from pii_guard.engine import Engine

    return LocalEngineClient(engine=Engine(Config.load()))


__all__ = ["LocalEngineClient", "create_local_client"]
