#!/bin/bash
set -euo pipefail

# Load environment variables
set -a
source .env
set +a

DD_LLMOBS_ENABLED=1 \
DD_LLMOBS_AGENTLESS_ENABLED=1 \
DD_LLMOBS_ML_APP="${DD_SERVICE}" \
ddtrace-run streamlit run app.py
