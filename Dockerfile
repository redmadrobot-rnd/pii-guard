# syntax=docker/dockerfile:1
# =============================================================================
# pii-guard — single container, single process, no queue and no state store.
#
#   CPU (default):  docker build -t pii-guard:cpu .
#   CUDA:           docker build -t pii-guard:cuda --build-arg TORCH_INDEX= .
#   Rules only:     docker build -t pii-guard:rules-only \
#                     --build-arg EXTRAS=server --build-arg NER_DISABLED=1 .
#
# Private model repo (needs a Hub token):
#   HF_TOKEN=hf_... docker build --secret id=hf_token,env=HF_TOKEN -t pii-guard:cpu .
#
# The token goes in as a BuildKit secret, never as a build arg: build args are
# recorded in the image and visible in `docker history`, secrets are not. Nothing
# is needed for a public model.
#
# No nvidia base image is needed for the CUDA variant: the torch wheels bundle
# the CUDA runtime, so the host only has to provide the driver and
# nvidia-container-toolkit. Run it with `--gpus all`.
# =============================================================================
FROM python:3.11-slim AS base

# CPU wheels are ~200 MB; the default PyPI wheels carry CUDA and are ~2.5 GB.
# Pass an empty TORCH_INDEX to get the CUDA build.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG EXTRAS=ner,server
ARG NER_MODEL=redmadrobot-rnd/rubert-base-pii-ner
ARG NER_REVISION=
ARG SPACY_MODEL=ru_core_news_sm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /app

# Create the runtime user before anything large lands on disk. Doing it after the
# model download and fixing ownership with `chown -R` would rewrite every cached
# file into a fresh layer -- a second, redundant copy of the ~700 MB of weights.
RUN useradd --create-home --uid 10001 piiguard \
    && mkdir -p "$HF_HOME" \
    && chown piiguard:piiguard "$HF_HOME" /app

# torch first and on its own layer: it is by far the largest dependency and the
# slowest to fetch, so it should not be invalidated by application changes.
RUN if echo "$EXTRAS" | grep -q ner; then \
      if [ -n "$TORCH_INDEX" ]; then \
        pip install torch --index-url "$TORCH_INDEX"; \
      else \
        pip install torch; \
      fi; \
    fi

COPY --chown=piiguard:piiguard pyproject.toml README.md LICENSE NOTICE ./
COPY --chown=piiguard:piiguard src ./src
RUN pip install ".[${EXTRAS}]" && python -m spacy download "$SPACY_MODEL"

# Drop privileges before the download so the cache is written owned by the
# runtime user -- no ownership fix-up, no duplicated layer.
USER piiguard

# Bake the weights into the image so the runtime never reaches for the network:
# the primary deployment target is an air-gapped environment, and N replicas
# should not each download the same 700 MB.
# uid= on the secret mount is required now that this RUN is not root.
RUN --mount=type=secret,id=hf_token,required=false,uid=10001 \
    if echo "$EXTRAS" | grep -q ner; then \
      HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)" \
      python -c "\
from transformers import AutoTokenizer, AutoModelForTokenClassification as M; \
rev = '${NER_REVISION}' or None; \
AutoTokenizer.from_pretrained('${NER_MODEL}', revision=rev); \
M.from_pretrained('${NER_MODEL}', revision=rev)"; \
    fi

# Rules-only builds have no weights, so the branch must be switched off
# explicitly -- otherwise start-up fails on a missing model. Pass
# `--build-arg EXTRAS=server --build-arg NER_DISABLED=1` for that variant.
ARG NER_DISABLED=0

ENV PII_GUARD_NER_MODEL=${NER_MODEL} \
    PII_GUARD_NER_REVISION=${NER_REVISION} \
    PII_GUARD_SPACY_MODEL=${SPACY_MODEL} \
    PII_GUARD_NER_DISABLED=${NER_DISABLED} \
    HF_HUB_OFFLINE=1

EXPOSE 8080

# The probe pays for a cold interpreter plus `import urllib` every time, which
# measured 2.2-5.1 s on a loaded machine -- a 5 s timeout marked a container
# serving 200s as unhealthy after nine checks in a row.
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health').status==200 else 1)"

# `python -m uvicorn`, not the `uvicorn` console script. That script is generated
# at install time and hard-codes `from uvicorn.main import main`, so it breaks
# whenever the script and the installed module get out of step -- a partial or
# cache-mixed install fails at start-up with `cannot import name 'main'`, and a
# base image whose interpreter no longer matches the script's shebang fails the
# same way. `-m` goes through the package's own `__main__.py`, where the two
# cannot disagree. Exec form either way: PID 1 stays python and signals reach
# uvicorn directly.
CMD ["python", "-m", "uvicorn", "pii_guard_server.main:app", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
