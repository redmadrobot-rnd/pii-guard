"""Chooses how the gate reaches the detector.

``local`` (default) calls the library in-process; ``http`` goes through the
service. Same numbers either way -- the HTTP layer adds no detection logic -- so
the default is the one that needs no container running.

Use ``http`` when the thing under test is a built image or a deployed instance
rather than the working tree.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover -- import-time typing only
    from .config import GateConfig

_ENV_VAR = "PII_GATE_TRANSPORT"
_DEFAULT = "local"


def transport_name() -> str:
    return (os.getenv(_ENV_VAR) or _DEFAULT).strip().lower()


def create_client(config: GateConfig) -> Any:
    """Return a client exposing ``.anonymize(text) -> dict``."""
    name = transport_name()
    if name == "local":
        from .local_client import create_local_client

        return create_local_client(config)
    if name == "http":
        from .api_client import create_guard_api_client

        return create_guard_api_client(config)
    raise ValueError(f"unknown {_ENV_VAR}={name!r}; expected 'local' or 'http'")


__all__ = ["create_client", "transport_name"]
