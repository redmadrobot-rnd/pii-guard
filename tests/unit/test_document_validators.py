"""Unit tests for checksum validators and strict document patterns."""

from __future__ import annotations

# Plain imports. These files used to be re-executed under synthetic module names
# via importlib, which fired every `@register_entity` a second time and left
# duplicates in the process-global `_REGISTERED_CLASSES` for the rest of the
# session. The registry deduplicates by entity type, so nothing broke -- but
# there is no reason to load the same code twice, and the symbols the tests need
# are reachable normally.
from pii_guard.entities import birth_certificate as BIRTH_CERT
from pii_guard.entities import inn as INN
from pii_guard.entities import military_id as MILITARY_ID
from pii_guard.entities import snils as SNILS
from pii_guard.framework.utils import luhn_valid


def test_snils_checksum_validation() -> None:
    assert SNILS._valid_snils("11223344595")
    assert SNILS._valid_snils("00002999900")
    assert SNILS._valid_snils("00003899900")
    assert not SNILS._valid_snils("11223344566")
    assert not SNILS._valid_snils("00002999901")
    assert not SNILS._valid_snils("1122334459")
    assert not SNILS._valid_snils("1122334459A")


def test_inn_checksum_validation() -> None:
    assert INN._valid_inn10("7715964180")
    assert not INN._valid_inn10("7715964181")
    assert not INN._valid_inn10("771596418")
    assert not INN._valid_inn10("771596418A")

    assert INN._valid_inn12("500100732259")
    assert not INN._valid_inn12("500100732258")
    assert not INN._valid_inn12("50010073225")
    assert not INN._valid_inn12("50010073225A")


def test_luhn_checksum_validation_for_card_and_oms() -> None:
    valid_16 = "4111111111111111"
    valid_15 = "378282246310005"
    invalid = "4111111111111112"
    short = "123456789"
    non_digit = "411111111111111A"

    assert luhn_valid(valid_16)
    assert luhn_valid(valid_15)
    assert not luhn_valid(invalid)
    assert not luhn_valid(short)
    assert not luhn_valid(non_digit)


def test_military_id_strict_pattern() -> None:
    assert MILITARY_ID.MILITARY_STRICT_RE.fullmatch("АБ 1234567")
    assert MILITARY_ID.MILITARY_STRICT_RE.fullmatch("PT-7654321")
    assert not MILITARY_ID.MILITARY_STRICT_RE.fullmatch("PT7654321")
    assert not MILITARY_ID.MILITARY_STRICT_RE.fullmatch("РФ 123456")


def test_birth_certificate_strict_pattern() -> None:
    assert BIRTH_CERT.BIRTH_CERT_STRICT_RE.fullmatch("I-МЮ 654321")
    assert BIRTH_CERT.BIRTH_CERT_STRICT_RE.fullmatch("VII-SHSH 333238")
    assert not BIRTH_CERT.BIRTH_CERT_STRICT_RE.fullmatch("VII-SHSH 33323")
    assert not BIRTH_CERT.BIRTH_CERT_STRICT_RE.fullmatch("серия I-МЮ 654321")
