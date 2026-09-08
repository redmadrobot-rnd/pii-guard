"""Request and response models for the HTTP layer."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from pii_guard.operators import MODES, Mode

# Hard ceilings on the pseudonym mapping, which had no bound at all while the
# text side has `max_texts` / `max_text_chars`. Ceilings, not operating limits:
# one entry per distinct PII value in a full batch is far below them.
MAX_MAPPING_ENTRIES = 4096
MAX_MAPPING_STRING_CHARS = 4096

_MappingString = Annotated[str, StringConstraints(max_length=MAX_MAPPING_STRING_CHARS)]


class Entity(BaseModel):
    start: int = Field(..., description="Entity start offset in normalized_text", examples=[11])
    end: int = Field(..., description="Entity end offset in normalized_text", examples=[22])
    entity_type: str = Field(..., description="Entity type", examples=["PERSON"])
    score: float | None = Field(None, description="Detector confidence", examples=[0.98])


class AnonymizeRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        description="Texts to process.",
        examples=[["Меня зовут Иван Петров, ИНН 7707083893"]],
    )
    mode: Mode = Field(
        "mask",
        description=(
            f"Output mode, one of {', '.join(MODES)}. "
            "`mask` replaces spans with the mask character, `tag` with "
            "`[ENTITY_TYPE]`, `pseudonymize` with reversible `<PII ... />` tags."
        ),
    )
    entities: list[str] | None = Field(
        None,
        description=(
            "Allowlist of entity types. Omit for the server default. The list "
            "gates the reported entities as well as the masking, so an empty "
            "list returns the text unchanged and no entities."
        ),
        examples=[["PERSON", "INN"]],
    )


class AnonymizeItem(BaseModel):
    text: str = Field(..., description="Text rendered according to `mode`")
    entities: list[Entity] = Field(default_factory=list, description="Detected entities")
    normalized_text: str = Field(
        ...,
        description=(
            "Text the entity offsets refer to. Normalisation and the "
            "English-numeral preprocessor can change length, so offsets index "
            "this string rather than the submitted one."
        ),
    )


class AnonymizeResponse(BaseModel):
    mode: Mode = Field(..., description="Mode that was applied")
    items: list[AnonymizeItem] = Field(default_factory=list, description="Parallel to request texts")
    mapping: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "`{tag: original}` — populated only for mode=pseudonymize. Keep it "
            "and send it back to /deanonymize; the service stores nothing."
        ),
    )


class DeanonymizeRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        description="Texts containing `<PII ... />` tags",
        examples=[['Позвоните <PII type="PERSON" gender="male" id="1" />']],
    )
    mapping: dict[_MappingString, _MappingString] = Field(
        ...,
        description="The mapping returned by /anonymize with mode=pseudonymize",
        max_length=MAX_MAPPING_ENTRIES,
    )


class DeanonymizeResponse(BaseModel):
    texts: list[str] = Field(default_factory=list, description="Texts with original values restored")


class HealthResponse(BaseModel):
    status: str = Field(..., description="ok when the engine is loaded", examples=["ok"])
    version: str = Field(..., description="pii-guard version")
    ner_enabled: bool = Field(
        ...,
        description=(
            "False when running rules-only. See `unavailable_entities` for the "
            "types that yields nothing in this deployment."
        ),
    )
    device: str | None = Field(None, description="Torch device used for NER inference")
    entities: list[str] = Field(default_factory=list, description="Default entity allowlist")
    unavailable_entities: list[str] = Field(
        default_factory=list,
        description=(
            "Configured entity types that cannot be detected in this deployment: "
            "the four NER-only types when the NER branch is off, plus any "
            "configured name no classifier answers to (a typo, typically). "
            "Without this they would be silently never returned."
        ),
        examples=[["LOCATION", "PERSON", "PHONE_NUMBER", "URL"]],
    )


__all__ = [
    "AnonymizeItem",
    "AnonymizeRequest",
    "AnonymizeResponse",
    "DeanonymizeRequest",
    "DeanonymizeResponse",
    "Entity",
    "HealthResponse",
]
