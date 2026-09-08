"""Unit tests for the domain entity classifiers.

BANK_ACCOUNT, POSTAL_CODE, TELEGRAM, BIK — deterministic checksum / context
branches mirrored from ``new_tags.ipynb`` and ``new_tags_validation.md``.
"""

from __future__ import annotations

from pii_guard.entities.bank_account import BankAccountClassifier, valid_account
from pii_guard.entities.bik import BikClassifier, valid_bik
from pii_guard.entities.postal_code import PostalCodeClassifier
from pii_guard.entities.telegram import TelegramClassifier
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


# ── BANK_ACCOUNT ────────────────────────────────────────────────


def test_bank_account_checksum_validator() -> None:
    # Корреспондентский счёт + БИК с валидной контрольной суммой.
    assert valid_account("30101810800000000451", "044525451") is True
    assert valid_account("30101810400000000225", "044525225") is False
    assert valid_account("1234", "044525451") is False  # не 20 цифр


def test_bank_account_valid_checksum_with_bik() -> None:
    bank = BankAccountClassifier()
    raw = "30101810800000000451"
    ctx = f"БИК 044525451 . Корреспондентский счёт : {raw} ."
    assert _classify_in_context(bank, raw, ctx) == ("BANK_ACCOUNT", 0.97)


def test_bank_account_invalid_checksum_but_context() -> None:
    bank = BankAccountClassifier()
    raw = "30101810400000000225"  # реальный корсчёт, синтетическая сумма не сходится
    ctx = f"БИК 044525225 . Корреспондентский счёт : {raw} ."
    assert _classify_in_context(bank, raw, ctx) == ("BANK_ACCOUNT", 0.60)


def test_bank_account_no_bik_with_context() -> None:
    bank = BankAccountClassifier()
    raw = "40817810500000067890"
    ctx = f"Расчётный счёт получателя : {raw}"
    assert _classify_in_context(bank, raw, ctx) == ("BANK_ACCOUNT", 0.90)


def test_bank_account_no_context_returns_none() -> None:
    bank = BankAccountClassifier()
    raw = "40817810500000067890"
    ctx = f"Случайное число {raw} в тексте"
    assert _classify_in_context(bank, raw, ctx) is None


def test_bank_account_wrong_length_returns_none() -> None:
    bank = BankAccountClassifier()
    raw = "12345"
    ctx = f"Расчётный счёт {raw}"
    assert _classify_in_context(bank, raw, ctx) is None


# ── POSTAL_CODE ─────────────────────────────────────────────────


def test_postal_code_plausible_prefix() -> None:
    postal = PostalCodeClassifier()
    raw = "191023"
    ctx = f"г . Санкт-Петербург , индекс {raw}"
    assert _classify_in_context(postal, raw, ctx) == ("POSTAL_CODE", 0.92)


def test_postal_code_prefix_out_of_range_returns_none() -> None:
    postal = PostalCodeClassifier()
    raw = "923456"  # префикс 923 > 699 — не существует
    ctx = f"почтовый адрес : {raw}"
    assert _classify_in_context(postal, raw, ctx) is None


def test_postal_code_prefix_500_599_returns_none() -> None:
    postal = PostalCodeClassifier()
    raw = "523456"  # префикс 523 — диапазон 500–599 отсутствует в системе
    ctx = f"почтовый адрес : {raw}"
    assert _classify_in_context(postal, raw, ctx) is None


def test_postal_code_prefix_600_699_valid() -> None:
    postal = PostalCodeClassifier()
    raw = "655017"  # префикс 655 — валидный (Хакасия)
    ctx = f"почтовый индекс {raw}"
    assert _classify_in_context(postal, raw, ctx) == ("POSTAL_CODE", 0.92)


def test_postal_code_negative_context_returns_none() -> None:
    postal = PostalCodeClassifier()
    raw = "123456"
    ctx = f"номер заказа ORD {raw}"  # «заказ» — негативный контекст
    assert _classify_in_context(postal, raw, ctx) is None


def test_postal_code_no_context_returns_none() -> None:
    postal = PostalCodeClassifier()
    raw = "123456"
    ctx = f"контрольное число {raw} в логах"
    assert _classify_in_context(postal, raw, ctx) is None


# ── POSTAL_CODE: близость ключевого слова ───────────────────────
# Регрессия по отчёту клиента от 05.08.2026: ключевые слова искались по всему
# окну, поэтому заголовок секции за 190 символов подавлял настоящий индекс, а
# слово «адресу» из следующего предложения превращало номер водительского
# удостоверения в индекс. Решает сравнение расстояний: побеждает ближайшее.


def test_postal_code_keyword_after_digits_is_accepted() -> None:
    # Самая частая форма русского адреса — индекс первым, ключ только справа.
    # Пробелы здесь как в живом тексте, а не как в остальных кейсах файла:
    # KW_POSTAL ищет «г.» слитно, и на «г .» это правило молчало и до правки.
    postal = PostalCodeClassifier()
    raw = "630009"
    ctx = f"{raw}, г. Новосибирск, ул. Ленина"
    assert _classify_in_context(postal, raw, ctx) == ("POSTAL_CODE", 0.92)


def test_postal_code_distant_keyword_after_digits_returns_none() -> None:
    # Тот же ключ, но за границей хвостового окна: следующее предложение
    # больше не голосует за шестизначное число из предыдущего.
    postal = PostalCodeClassifier()
    raw = "630009"
    ctx = (
        f"контрольное число {raw} в логах сборки , выгрузка прошла без замечаний , "
        "и только потом отдельной строкой шёл город Новосибирск"
    )
    assert _classify_in_context(postal, raw, ctx) is None


def test_postal_code_distant_negative_keyword_does_not_suppress() -> None:
    # «обращения» в заголовке секции — 190 символов до числа, вне окна.
    postal = PostalCodeClassifier()
    raw = "630009"
    ctx = (
        "Суть обращения . В службу поддержки обратилась клиентка , проживающая "
        f"по адресу : г . Новосибирск , ул . Ленина , д . 45 , кв . 112 ( индекс {raw} )"
    )
    assert _classify_in_context(postal, raw, ctx) == ("POSTAL_CODE", 0.92)


def test_postal_code_yields_driver_license_number() -> None:
    # Номер прав — тоже шесть цифр. Адресного ключа перед ним нет, название
    # документа есть, поэтому индекс отказывается от числа.
    postal = PostalCodeClassifier()
    raw = "123456"
    ctx = f"водительское удостоверение 77 АВ {raw} , доставка по адресу"
    assert _classify_in_context(postal, raw, ctx) is None


def test_postal_code_exchange_index_is_vetoed_not_outranked() -> None:
    # «биржевой» стоит перед «индекс», то есть всегда дальше от цифр, чем слово,
    # которое он дисквалифицирует. Сравнение расстояний тут не работает — нужно
    # безусловное вето, иначе биржевые сводки дают ложные индексы.
    postal = PostalCodeClassifier()
    raw = "275761"
    ctx = f"Биржевой индекс сегодня вырос на {raw} пунктов, рынок реагирует."
    assert _classify_in_context(postal, raw, ctx) is None


def test_postal_code_confirmed_by_street_type_two_components_later() -> None:
    # Индекс первым, подтверждающий ключ — тип улицы через город: «шоссе» стоит
    # в 35 символах, и хвостовое окно должно вмещать слово целиком.
    postal = PostalCodeClassifier()
    raw = "115487"
    ctx = f'ООО "Рога и копыта" {raw}, Регион 77, Чапаевск, Дмитровское шоссе, дом 10'
    assert _classify_in_context(postal, raw, ctx) == ("POSTAL_CODE", 0.92)


def test_postal_code_nearest_keyword_wins_over_document_word() -> None:
    # Оба слова в окне: «индекс» ближе, чем «удостоверение» — индекс побеждает.
    postal = PostalCodeClassifier()
    raw = "630009"
    ctx = f"водительское удостоверение 77 АВ 123456 , индекс {raw}"
    assert _classify_in_context(postal, raw, ctx) == ("POSTAL_CODE", 0.92)


# ── TELEGRAM (поддерживается только @username) ──────────────────


def test_telegram_username() -> None:
    tg = TelegramClassifier()
    raw = "@durov_official"
    ctx = f"пишите в телеграм {raw}"
    assert _classify_in_context(tg, raw, ctx) == ("TELEGRAM", 0.90)


def test_telegram_username_instagram_returns_none() -> None:
    tg = TelegramClassifier()
    raw = "@durov_official"
    ctx = f"мой инстаграм {raw}"
    assert _classify_in_context(tg, raw, ctx) is None


def test_telegram_tme_link_no_longer_detected() -> None:
    # Форма t.me/… больше не поддерживается — только @username.
    tg = TelegramClassifier()
    raw = "t.me/durov_official"
    ctx = f"мой канал {raw}"
    assert _classify_in_context(tg, raw, ctx) is None


def test_telegram_numeric_id_no_longer_detected() -> None:
    # Числовой user_id больше не распознаётся (давал FP на телефонах/кодах).
    tg = TelegramClassifier()
    raw = "123456789"
    ctx = f"telegram id пользователя {raw}"
    assert _classify_in_context(tg, raw, ctx) is None


def test_telegram_bare_nick_no_longer_detected() -> None:
    # Голый ник без @ больше не распознаётся (хватал локалпарт e-mail).
    tg = TelegramClassifier()
    raw = "ivan_petrov"
    ctx = f"мой ник в телеграме {raw}"
    assert _classify_in_context(tg, raw, ctx) is None


# ── BIK ─────────────────────────────────────────────────────────


def test_bik_structural_validator() -> None:
    assert valid_bik("044525225") is True  # 04… + номер банка 225
    assert valid_bik("123456789") is False  # код страны не 04
    assert valid_bik("044525030") is False  # номер банка 030 < 050
    assert valid_bik("04452522") is False  # не 9 цифр


def test_bik_with_context_valid_structure() -> None:
    bik = BikClassifier()
    raw = "044525225"  # 04… + номер банка 225 → валидный
    ctx = f"БИК {raw}"
    assert _classify_in_context(bik, raw, ctx) == ("BIK", 0.95)


def test_bik_with_context_invalid_country_code() -> None:
    bik = BikClassifier()
    raw = "123456789"  # 9 цифр, но код страны не 04
    ctx = f"БИК банка получателя {raw}"
    assert _classify_in_context(bik, raw, ctx) == ("BIK", 0.85)


def test_bik_with_context_invalid_bank_number() -> None:
    bik = BikClassifier()
    raw = "044525030"  # номер банка 030 < 050 → структура не сходится
    ctx = f"БИК {raw}"
    assert _classify_in_context(bik, raw, ctx) == ("BIK", 0.85)


def test_bik_without_context_returns_none() -> None:
    bik = BikClassifier()
    raw = "044525225"
    ctx = f"номер обращения {raw} в системе"
    assert _classify_in_context(bik, raw, ctx) is None


def test_bik_wrong_length_returns_none() -> None:
    bik = BikClassifier()
    raw = "04452522"  # 8 цифр
    ctx = f"БИК {raw}"
    assert _classify_in_context(bik, raw, ctx) is None
