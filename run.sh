#!/bin/bash
set -euo pipefail

# Load environment variables from .env if present
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

DD_LLMOBS_ENABLED=1 \
DD_LLMOBS_AGENTLESS_ENABLED=1 \
DD_LLMOBS_ML_APP="${DD_SERVICE}" \
ddtrace-run streamlit run app.py
