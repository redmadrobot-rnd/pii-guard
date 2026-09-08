"""Unit tests for the HTTP layer: authentication, request limits, readiness.

The engine is never loaded here. `TestClient` is used without its context
manager on purpose, so `lifespan` does not run: that is exactly the state in
which the server must answer 503 rather than 500, and it keeps the suite free of
spaCy models and model weights.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="server extra not installed")

from fastapi import HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from pii_guard_server import auth  # noqa: E402

# ── Authentication ──────────────────────────────────────────────

def test_auth_off_accepts_anything(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth, "API_KEYS", frozenset())

    assert auth.require_api_key(None) is None
    assert auth.require_api_key("Bearer whatever") is None


@pytest.mark.parametrize(
    "header",
    [
        None,
        "",
        "secret",  # no scheme
        "Basic secret",  # wrong scheme
        "Bearer",  # scheme only
        "Bearer wrong",
        "Bearer secre",  # prefix of a valid key
        "Bearer secretx",
    ],
)
def test_auth_rejects(header: str | None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth, "API_KEYS", frozenset({"secret", "other"}))

    with pytest.raises(HTTPException) as excinfo:
        auth.require_api_key(header)

    assert excinfo.value.status_code == 401
    assert excinfo.value.headers == {"WWW-Authenticate": "Bearer"}


@pytest.mark.parametrize("key", ["secret", "other"])
def test_auth_accepts_every_configured_key(key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth, "API_KEYS", frozenset({"secret", "other"}))

    assert auth.require_api_key(f"Bearer {key}") is None
    # Surrounding whitespace is the shape a shell `export` leaves behind.
    assert auth.require_api_key(f"Bearer  {key} ") is None


def test_auth_env_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PII_GUARD_API_KEYS", " a , b ,, c ")
    assert auth._load_keys() == frozenset({"a", "b", "c"})

    monkeypatch.setenv("PII_GUARD_API_KEYS", "")
    assert auth._load_keys() == frozenset()

    monkeypatch.delenv("PII_GUARD_API_KEYS")
    assert auth._load_keys() == frozenset()


# ── Request limits and readiness ─────────────────────────────────

pytest.importorskip("presidio_anonymizer", reason="core dependencies not installed")

from pii_guard.config import Config  # noqa: E402
from pii_guard_server import main  # noqa: E402


@pytest.fixture
def configured(monkeypatch: pytest.MonkeyPatch):
    """`_state` as lifespan leaves it, minus the engine itself."""
    config = Config.load(ner_disabled=True, max_texts=2, max_text_chars=10)
    monkeypatch.setitem(main._state, "config", config)
    return config


def test_limits_reject_empty_list(configured: Config) -> None:
    with pytest.raises(HTTPException) as excinfo:
        main._check_limits([])
    assert excinfo.value.status_code == 422


def test_limits_reject_too_many_texts(configured: Config) -> None:
    with pytest.raises(HTTPException) as excinfo:
        main._check_limits(["a", "b", "c"])
    assert excinfo.value.status_code == 413


def test_limits_reject_oversized_text(configured: Config) -> None:
    with pytest.raises(HTTPException) as excinfo:
        main._check_limits(["a" * 11])
    assert excinfo.value.status_code == 413
    assert "texts[0]" in excinfo.value.detail


def test_limits_accept_the_boundary(configured: Config) -> None:
    assert main._check_limits(["a" * 10, "b"]) is None


def test_limits_before_startup_are_503(monkeypatch: pytest.MonkeyPatch) -> None:
    # A request that arrives before lifespan finished, or after shutdown cleared
    # the state. Indexing `_state` here used to make this a 500.
    monkeypatch.setattr(main, "_state", {})

    with pytest.raises(HTTPException) as excinfo:
        main._check_limits(["короткий текст"])
    assert excinfo.value.status_code == 503


def test_endpoints_before_startup_are_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "_state", {})
    monkeypatch.setattr(auth, "API_KEYS", frozenset())
    client = TestClient(main.app)  # no `with`: lifespan never runs

    assert client.post("/anonymize", json={"texts": ["привет"]}).status_code == 503
    assert client.post(
        "/deanonymize", json={"texts": ["привет"], "mapping": {}}
    ).status_code == 503
    assert client.get("/health").status_code == 503


def test_auth_is_checked_before_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    # Order matters: an unauthenticated caller must not be able to tell a loading
    # engine from a loaded one.
    monkeypatch.setattr(main, "_state", {})
    monkeypatch.setattr(auth, "API_KEYS", frozenset({"secret"}))
    client = TestClient(main.app)

    assert client.post("/anonymize", json={"texts": ["привет"]}).status_code == 401
    assert client.post(
        "/anonymize", json={"texts": ["привет"]}, headers={"Authorization": "Bearer secret"}
    ).status_code == 503


def test_mapping_size_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    from pii_guard_server.schemas import MAX_MAPPING_ENTRIES, MAX_MAPPING_STRING_CHARS

    monkeypatch.setattr(main, "_state", {})
    monkeypatch.setattr(auth, "API_KEYS", frozenset())
    client = TestClient(main.app)

    too_many = {f"<PII type=\"INN\" id=\"{i}\" />": "x" for i in range(MAX_MAPPING_ENTRIES + 1)}
    assert client.post(
        "/deanonymize", json={"texts": ["привет"], "mapping": too_many}
    ).status_code == 422

    too_long = {"<PII type=\"INN\" id=\"1\" />": "x" * (MAX_MAPPING_STRING_CHARS + 1)}
    assert client.post(
        "/deanonymize", json={"texts": ["привет"], "mapping": too_long}
    ).status_code == 422
