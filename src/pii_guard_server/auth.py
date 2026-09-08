"""Optional bearer-token check.

Off unless ``PII_GUARD_API_KEYS`` is set, and there is no default key on purpose:
a shipped placeholder that nobody rotates is worse than no authentication at all,
because it reads as protection while granting none.

This service holds no data at rest, mutates nothing and bills nobody -- the only
thing worth protecting is CPU time. For an internal component that is a network
boundary concern, so publish the port on loopback and put a proxy in front if it
must cross zones. The knob exists so nobody has to fork the project to add one.
"""

from __future__ import annotations

import logging
import os
import secrets

from fastapi import Header, HTTPException, status

logger = logging.getLogger("pii_guard.server.auth")

_ENV_VAR = "PII_GUARD_API_KEYS"


def _load_keys() -> frozenset[str]:
    raw = os.getenv(_ENV_VAR, "")
    return frozenset(k.strip() for k in raw.split(",") if k.strip())


API_KEYS: frozenset[str] = _load_keys()


def auth_enabled() -> bool:
    return bool(API_KEYS)


def require_api_key(authorization: str | None = Header(default=None)) -> None:
    """FastAPI dependency: enforce ``Authorization: Bearer <key>`` when configured."""
    if not API_KEYS:
        return

    token = ""
    if authorization:
        scheme, _, value = authorization.partition(" ")
        token = value.strip() if scheme.lower() == "bearer" else ""

    # Every comparison is `compare_digest`, so none leaks how long a prefix
    # matched. `any` stops at the first hit, so total time depends on the key's
    # position in the set -- that reveals how many were checked, not any key's
    # value. Byte-level leakage is the property that matters.
    if not any(secrets.compare_digest(token, key) for key in API_KEYS):
        # Never log the presented token.
        logger.warning("event=auth_rejected reason=%s", "missing" if not token else "mismatch")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


__all__ = ["API_KEYS", "auth_enabled", "require_api_key"]
