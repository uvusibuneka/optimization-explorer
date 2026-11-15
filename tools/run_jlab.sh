#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if (( $# < 1 )); then
    echo "Usage: bash tools/run_jlab.sh DATABASE [run options]" >&2
    exit 2
fi
db=$1
shift

export OPENKILO_MODEL="${OPENKILO_MODEL:-openai/worker}"
export OPENKILO_BASE_URL="${OPENKILO_BASE_URL:-http://127.0.0.1:18765/v1}"
export GROQ_MODEL="${GROQ_MODEL:-openai/validator}"
export GROQ_BASE_URL="${GROQ_BASE_URL:-http://127.0.0.1:18766/v1}"
if [[ ${GROQ_BASE_URL} != "http://127.0.0.1:18766/v1" || ${GROQ_MODEL} != openai/* ]]; then
    if [[ -z ${GROQ_API_KEY:-} && -t 0 ]]; then
        read -rsp "Groq API key: " GROQ_API_KEY
        printf '\n' >&2
    fi
    export GROQ_API_KEY="${GROQ_API_KEY:?Set GROQ_API_KEY for a non-local Groq endpoint}"
fi
export ORCHESTRATOR_BASE_URL="${ORCHESTRATOR_BASE_URL:-http://127.0.0.1:18327/v1}"
export ORCHESTRATOR_MODEL="${ORCHESTRATOR_MODEL:-openai/qwen3-32b}"

exec .venv/bin/python -m optimization_explorer --db "$db" run --mode conductor --log-model-metrics "$@"
