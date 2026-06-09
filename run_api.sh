#!/bin/bash
set -euo pipefail

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

DD_SERVICE="${DD_SERVICE:-music-recommender}"

DD_LLMOBS_ENABLED=1 \
DD_LLMOBS_AGENTLESS_ENABLED=1 \
DD_LLMOBS_ML_APP="${DD_SERVICE}" \
ddtrace-run uvicorn api:app --reload --port 8000
