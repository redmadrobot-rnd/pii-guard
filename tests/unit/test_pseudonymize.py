"""Unit tests for reversible pseudonymisation.

Deliberately avoids PERSON/LOCATION *de*-anonymisation: putting a name back in
the right grammatical case needs a spaCy model, which CI does not download. The
pseudonymisation direction needs only pymorphy3, which is a base dependency, so
tag generation and id stability are covered for every type.
"""

from __future__ import annotations

from pii_guard.pseudonymize import deanonymize, deanonymize_batch, pseudonymize_batch


def _tag(entity_type: str, ident: int) -> str:
    return f'<PII type="{entity_type}" id="{ident}" />'


def test_round_trip_non_person_types() -> None:
    text = "ИНН 7707083893, телефон +7 916 123-45-67"
    entities = [[
        {"start": 4, "end": 14, "entity_type": "INN", "score": 0.9},
        {"start": 24, "end": 40, "entity_type": "PHONE_NUMBER", "score": 0.7},
    ]]

    pseudo, mapping, _state = pseudonymize_batch([text], entities)

    assert pseudo[0] == f"ИНН {_tag('INN', 1)}, телефон {_tag('PHONE_NUMBER', 1)}"
    assert mapping == {_tag("INN", 1): "7707083893", _tag("PHONE_NUMBER", 1): "+7 916 123-45-67"}
    assert deanonymize_batch(pseudo, mapping) == [text]


def test_repeated_value_keeps_one_id() -> None:
    text = "ИНН 7707083893 и ещё раз 7707083893"
    entities = [[
        {"start": 4, "end": 14, "entity_type": "INN", "score": 0.9},
        {"start": 25, "end": 35, "entity_type": "INN", "score": 0.9},
    ]]

    pseudo, mapping, _state = pseudonymize_batch([text], entities)

    assert len(mapping) == 1
    assert pseudo[0].count(_tag("INN", 1)) == 2
    assert deanonymize_batch(pseudo, mapping) == [text]


def test_state_keeps_ids_stable_across_calls() -> None:
    entities = [[{"start": 4, "end": 14, "entity_type": "INN", "score": 0.9}]]

    first, mapping, state = pseudonymize_batch(["ИНН 7707083893"], entities)
    second, mapping_2, _ = pseudonymize_batch(["ИНН 7707083893"], entities, state=state)

    assert first[0] == second[0]
    assert mapping == mapping_2


def test_unknown_tag_is_scrubbed_not_left_in_place() -> None:
    # A model that invents a tag must not have it echoed back to the caller: the
    # tag is not PII, but leaving it there hands the caller something that looks
    # like a restorable reference and is not.
    hallucinated = 'Звоните <PII type="PHONE_NUMBER" id="42" />'

    assert deanonymize(hallucinated, {}) == "Звоните ****"
    assert deanonymize(hallucinated, {_tag("INN", 1): "7707083893"}) == "Звоните ****"


def test_all_occurrences_of_one_tag_are_restored() -> None:
    tag = _tag("INN", 1)
    text = f"{tag} и {tag} и снова {tag}"

    assert deanonymize(text, {tag: "7707083893"}) == "7707083893 и 7707083893 и снова 7707083893"


def test_self_referential_mapping_terminates() -> None:
    # `{tag: tag}` reinserts the tag on every substitution. Without a bound the
    # loop never ends, and `mapping` is unauthenticated caller input.
    tag = _tag("INN", 1)

    assert deanonymize(tag, {tag: tag}) == "****"


def test_chained_mapping_cannot_amplify() -> None:
    # Each value re-emits two copies of the *next* key's tag. A per-key bound let
    # the tag count double per key -- exponential work from a ~1 KB body. The
    # budget is global, so the whole call makes at most one substitution per tag
    # present in the input and the leftovers are scrubbed.
    mapping = {_tag("INN", i): f"{_tag('INN', i + 1)} {_tag('INN', i + 1)}" for i in range(1, 33)}

    result = deanonymize(_tag("INN", 1), mapping)

    assert "<PII" not in result
    assert len(result) < 100
