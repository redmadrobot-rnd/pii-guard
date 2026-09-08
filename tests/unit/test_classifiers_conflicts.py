"""Unit tests for deterministic classifier and conflict-resolution branches."""

from __future__ import annotations

from pii_guard.entities.birth_certificate import BirthCertificateClassifier
from pii_guard.entities.credit_card import CreditCardClassifier
from pii_guard.entities.driver_license import DriverLicenseClassifier
from pii_guard.entities.military_id import MilitaryIdClassifier
from pii_guard.entities.oms import OmsClassifier
from pii_guard.entities.passport import PassportClassifier
from pii_guard.framework.utils import digits_only


def _classify_in_context(classifier, raw: str, ctx: str, wide_ctx: str | None = None):
    ctx_pos = ctx.index(raw)
    wide_ctx_pos = wide_ctx.index(raw) if wide_ctx is not None else 0
    return classifier.classify(
        raw,
        digits_only(raw),
        ctx,
        wide_ctx,
        ctx_pos,
        wide_ctx_pos,
    )


def test_passport_classifier_matches_passport_context() -> None:
    passport = PassportClassifier()
    raw = "4510 678234"
    ctx = "Паспорт гражданина РФ: 4510 678234"

    assert _classify_in_context(passport, raw, ctx) == ("PASSPORT", 0.80)


def test_driver_license_classifier_matches_driver_license_context() -> None:
    driver_license = DriverLicenseClassifier()
    raw = "4510 678234"
    ctx = "Водительские права: 4510 678234"

    assert _classify_in_context(driver_license, raw, ctx) == (
        "DRIVER_LICENSE",
        0.80,
    )


def test_passport_vs_driver_license_conflict_prefers_nearest_keyword() -> None:
    passport = PassportClassifier()
    driver_license = DriverLicenseClassifier()
    raw = "4510 678234"

    passport_ctx = "Водительские права, паспорт гражданина 4510 678234"
    assert _classify_in_context(passport, raw, passport_ctx) == ("PASSPORT", 0.75)
    assert _classify_in_context(driver_license, raw, passport_ctx) is None

    driver_license_ctx = "Паспорт гражданина, водительские права ГИБДД 4510 678234"
    assert _classify_in_context(passport, raw, driver_license_ctx) is None
    assert _classify_in_context(driver_license, raw, driver_license_ctx) == (
        "DRIVER_LICENSE",
        0.75,
    )


def test_credit_card_and_oms_classifiers_follow_context() -> None:
    credit_card = CreditCardClassifier()
    oms = OmsClassifier()
    raw = "4111111111111111"

    card_ctx = "Номер банковской карты 4111111111111111"
    assert _classify_in_context(credit_card, raw, card_ctx) == ("CREDIT_CARD", 0.90)
    assert _classify_in_context(oms, raw, card_ctx) is None

    oms_ctx = "Номер полиса ОМС 4111111111111111"
    assert _classify_in_context(credit_card, raw, oms_ctx) is None
    assert _classify_in_context(oms, raw, oms_ctx) == ("OMS", 0.90)


def test_credit_card_vs_oms_conflict_prefers_nearest_keyword() -> None:
    credit_card = CreditCardClassifier()
    oms = OmsClassifier()
    raw = "4111111111111111"
    ctx = "Банковская карта, номер полиса ОМС 4111111111111111"

    assert _classify_in_context(credit_card, raw, ctx) is None
    assert _classify_in_context(oms, raw, ctx) == ("OMS", 0.85)


def test_military_id_classifier_requires_keyword_for_ambiguous_series() -> None:
    classifier = MilitaryIdClassifier()

    assert _classify_in_context(classifier, "АБ 1234567", "АБ 1234567") == (
        "MILITARY_ID",
        0.95,
    )
    assert _classify_in_context(classifier, "РФ 1234567", "РФ 1234567") is None
    assert _classify_in_context(
        classifier,
        "РФ 1234567",
        "Военный билет РФ 1234567",
    ) == ("MILITARY_ID", 0.90)


def test_birth_certificate_classifier_distinguishes_strict_and_contextual_forms() -> None:
    classifier = BirthCertificateClassifier()

    assert _classify_in_context(classifier, "I-МЮ 654321", "I-МЮ 654321") == (
        "BIRTH_CERTIFICATE",
        0.85,
    )
    assert _classify_in_context(
        classifier,
        "I-МЮ 654321",
        "Свидетельство о рождении I-МЮ 654321",
    ) == ("BIRTH_CERTIFICATE", 0.95)
    assert _classify_in_context(
        classifier,
        "серия II-МЮ №950132",
        "серия II-МЮ №950132",
    ) == ("BIRTH_CERTIFICATE", 0.80)
    assert _classify_in_context(
        classifier,
        "серия II-МЮ №950132",
        "Свидетельство о рождении серия II-МЮ №950132",
    ) == ("BIRTH_CERTIFICATE", 0.90)
