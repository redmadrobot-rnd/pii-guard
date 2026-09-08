"""HTTP wrapper around :class:`pii_guard.Anonymizer`.

    POST /anonymize     texts -> masked / tagged / pseudonymised + mapping
    POST /deanonymize   texts + mapping -> originals
    GET  /health

Stateless: the pseudonym mapping is returned to the caller and never stored, so
there is no mapping store, no TTL and no correlation id.

Two things the queue-based service got for free and this one has to do itself:

* **Back-pressure.** Inference is serialised (spaCy and torch are not safe to
  drive concurrently), so a burst would otherwise pile up threads waiting on a
  lock. A bounded semaphore with a timeout turns overload into an explicit 503.
* **Fail at start-up.** The model is loaded during lifespan, so a missing model
  or a bad device kills the process at boot instead of failing the first request.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import anyio
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from pii_guard import __version__
from pii_guard.anonymizer import Anonymizer
from pii_guard.config import Config
from pii_guard.engine import Base64BudgetExceeded
from pii_guard_server.auth import auth_enabled, require_api_key
from pii_guard_server.schemas import (
    AnonymizeItem,
    AnonymizeRequest,
    AnonymizeResponse,
    DeanonymizeRequest,
    DeanonymizeResponse,
    HealthResponse,
)

# Starlette renamed these two (`HTTP_422_UNPROCESSABLE_ENTITY` ->
# `..._UNPROCESSABLE_CONTENT`, `HTTP_413_REQUEST_ENTITY_TOO_LARGE` ->
# `HTTP_413_CONTENT_TOO_LARGE`) and the old names now warn on every access, while
# the new ones are absent from the older end of the supported `fastapi>=0.115`
# range. The numbers are fixed by RFC 9110, so use them directly.
HTTP_413_TOO_LARGE = 413
HTTP_422_INVALID = 422

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    force=True,
)
logger = logging.getLogger("pii_guard.server")

# How long a request waits for its turn on the engine before giving up. Short on
# purpose: a caller blocked for a minute has already timed out upstream.
QUEUE_TIMEOUT_SECONDS = 30.0

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = Config.load()
    logger.info(
        "event=startup ner_model=%s revision=%s ner_enabled=%s auth=%s",
        config.ner_model, config.ner_revision or "unpinned",
        config.ner_enabled, "on" if auth_enabled() else "off",
    )
    # anyio, not threading: the wait happens in the event loop. A threading lock
    # taken inside the worker thread would first need one of anyio's 40 threads,
    # and that wait has no timeout -- past 40 concurrent requests callers queued
    # forever instead of getting the 503. Set before the slow model load so
    # `_run_exclusive` never sees a half-built state.
    _state["slot"] = anyio.Semaphore(1)
    _state["config"] = config
    # Load eagerly: a missing model must break the boot, not the first request.
    anonymizer = await anyio.to_thread.run_sync(lambda: Anonymizer(config))
    _state["anonymizer"] = anonymizer
    logger.info("event=startup_complete ner_enabled=%s", anonymizer.ner_enabled)
    try:
        yield
    finally:
        logger.info("event=shutdown")
        _state.clear()


app = FastAPI(
    title="pii-guard",
    version=__version__,
    description=(
        "PII detection, anonymisation and reversible pseudonymisation for Russian "
        "text. Rules branch (checksum-validated documents, regex entities, bypass "
        "preprocessors) plus an optional transformer NER branch for names and "
        "addresses."
    ),
    lifespan=lifespan,
    openapi_tags=[
        {"name": "PII", "description": "Anonymisation and de-anonymisation"},
        {"name": "Health", "description": "Liveness and readiness"},
    ],
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """422 without echoing the rejected value.

    FastAPI's default handler puts the offending input in the response body. On
    this service that input is PII: a mapping entry over the length cap came back
    with the name and the tax id in the clear, to a caller who may not be the one
    who sent it (proxy logs, error trackers, browser history). `_check_limits`
    already reports sizes only; this keeps the pydantic path to the same rule.
    """
    return JSONResponse(
        status_code=HTTP_422_INVALID,
        content={
            "detail": [
                {
                    "type": err.get("type"),
                    # `loc` names the field. For a dict field it includes the key,
                    # which here is a `<PII .../>` tag -- not a secret, but it is
                    # caller-supplied, so cap it rather than trust its length.
                    "loc": [str(part)[:64] for part in err.get("loc", ())],
                    "msg": err.get("msg"),
                }
                for err in exc.errors()
            ]
        },
    )


def _anonymizer() -> Anonymizer:
    anonymizer = _state.get("anonymizer")
    if anonymizer is None:  # pragma: no cover -- only before lifespan completes
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "engine not ready")
    return anonymizer


def _check_limits(texts: list[str]) -> None:
    # Both endpoints validate before touching the engine, so this runs first and
    # must survive a request that arrives before lifespan finished. Indexing
    # `_state` directly turned that window into a 500 and made the deliberate 503
    # in `_anonymizer()` unreachable.
    config: Config | None = _state.get("config")
    if config is None:  # pragma: no cover -- only before lifespan completes
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "engine not ready")
    if not texts:
        raise HTTPException(HTTP_422_INVALID, "texts must not be empty")
    if len(texts) > config.max_texts:
        raise HTTPException(
            HTTP_413_TOO_LARGE,
            f"too many texts: {len(texts)} > {config.max_texts}",
        )
    for index, text in enumerate(texts):
        if len(text) > config.max_text_chars:
            raise HTTPException(
                HTTP_413_TOO_LARGE,
                f"texts[{index}] is {len(text)} chars, limit is {config.max_text_chars}",
            )


async def _run_exclusive(fn, *args, **kwargs):
    """Run *fn* in a worker thread, one request at a time, or fail with 503."""
    slot: anyio.Semaphore | None = _state.get("slot")
    if slot is None:
        # Same window as `_check_limits`: `lifespan` populates `_state` before
        # yielding and clears it on shutdown, so a request in flight across
        # either edge finds it missing. Indexing turned that into a 500.
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "engine not ready")

    try:
        with anyio.fail_after(QUEUE_TIMEOUT_SECONDS):
            await slot.acquire()
    except TimeoutError:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "engine busy, retry later",
        ) from None

    try:
        return await anyio.to_thread.run_sync(lambda: fn(*args, **kwargs))
    finally:
        slot.release()


@app.post(
    "/anonymize",
    response_model=AnonymizeResponse,
    tags=["PII"],
    summary="Detect PII and render it per the chosen mode",
    dependencies=[Depends(require_api_key)],
)
async def anonymize(request: AnonymizeRequest) -> AnonymizeResponse:
    _check_limits(request.texts)
    anonymizer = _anonymizer()

    try:
        result = await _run_exclusive(
            anonymizer.anonymize,
            request.texts,
            mode=request.mode,
            entities=request.entities,
        )
    except Base64BudgetExceeded as exc:
        # The engine refuses rather than analysing part of the base64 and letting
        # the rest through unmasked. The message carries counters only.
        raise HTTPException(HTTP_413_TOO_LARGE, str(exc)) from None

    items = [
        AnonymizeItem(text=text, entities=entities, normalized_text=normalized)
        for text, entities, normalized in zip(
            result.texts, result.entities, result.normalized_texts, strict=True
        )
    ]
    return AnonymizeResponse(mode=result.mode, items=items, mapping=result.mapping)


@app.post(
    "/deanonymize",
    response_model=DeanonymizeResponse,
    tags=["PII"],
    summary="Restore original values from a pseudonymisation mapping",
    dependencies=[Depends(require_api_key)],
)
async def deanonymize(request: DeanonymizeRequest) -> DeanonymizeResponse:
    _check_limits(request.texts)
    anonymizer = _anonymizer()
    texts = await _run_exclusive(anonymizer.deanonymize, request.texts, request.mapping)
    return DeanonymizeResponse(texts=texts)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Liveness and engine status",
)
async def health() -> HealthResponse:
    anonymizer = _state.get("anonymizer")
    config: Config | None = _state.get("config")
    if anonymizer is None or config is None:
        return JSONResponse(  # type: ignore[return-value]
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "loading", "version": __version__, "ner_enabled": False},
        )
    device = None
    # Say out loud which configured types this deployment cannot find, rather
    # than letting a caller assume an empty result means "no PII here".
    from pii_guard.config import DEFAULT_ENTITIES, NER_ONLY_ENTITIES
    from pii_guard.detect import default_registry

    configured = set(config.entities)
    # A misspelled type is accepted verbatim by the config layer -- it is a plain
    # string list, deliberately not coupled to the registry -- so `PERSSON` would
    # otherwise be reported as configured and simply never match anything.
    known = set(DEFAULT_ENTITIES) | set(default_registry().entity_types) | NER_ONLY_ENTITIES
    unavailable = sorted(configured - known)
    if anonymizer.ner_enabled:
        from pii_guard.ner.recognizer import resolve_device

        device = resolve_device(config.device)
    else:
        unavailable = sorted(set(unavailable) | (NER_ONLY_ENTITIES & configured))
    return HealthResponse(
        status="ok",
        version=__version__,
        ner_enabled=anonymizer.ner_enabled,
        device=device,
        entities=list(config.entities),
        unavailable_entities=unavailable,
    )


def run() -> None:  # pragma: no cover -- console entry point
    """Serve on loopback by default.

    Authentication is off unless ``PII_GUARD_API_KEYS`` is set, so binding every
    interface here would put an unauthenticated PII endpoint on the network for
    anyone who followed the README's one-liner. Exposing it is an explicit
    decision, made by setting ``PII_GUARD_HOST`` -- not a default. The container
    image binds 0.0.0.0 because there the network boundary is the published port.
    """
    import uvicorn

    host = os.getenv("PII_GUARD_HOST", "127.0.0.1")
    port = int(os.getenv("PII_GUARD_PORT", "8080"))
    uvicorn.run("pii_guard_server.main:app", host=host, port=port, access_log=False)


if __name__ == "__main__":  # pragma: no cover
    run()
