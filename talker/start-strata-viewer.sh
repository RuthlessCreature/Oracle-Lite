#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export ORACLE_VIEWER_MODEL_BASE_URL="${ORACLE_VIEWER_MODEL_BASE_URL:-http://127.0.0.1:8081}"
export ORACLE_VIEWER_MODEL="${ORACLE_VIEWER_MODEL:-qwen3.8-flash-next-iq2_xs}"
export ORACLE_VIEWER_MODEL_BACKEND="${ORACLE_VIEWER_MODEL_BACKEND:-strata-native}"
export ORACLE_TALKER_OPENAI_BASE_URL="${ORACLE_TALKER_OPENAI_BASE_URL:-$ORACLE_VIEWER_MODEL_BASE_URL}"
export ORACLE_TALKER_OPENAI_MODEL="${ORACLE_TALKER_OPENAI_MODEL:-$ORACLE_VIEWER_MODEL}"
export ORACLE_TALKER_INFERENCE_BACKEND="${ORACLE_TALKER_INFERENCE_BACKEND:-$ORACLE_VIEWER_MODEL_BACKEND}"
export ORACLE_TALKER_OPENAI_API_KEY="${ORACLE_TALKER_OPENAI_API_KEY:-${ORACLE_VIEWER_MODEL_API_KEY:-}}"
export ORACLE_LOCAL_VISION_BASE_URL="${ORACLE_LOCAL_VISION_BASE_URL:-http://127.0.0.1:7871}"
export ORACLE_LOCAL_VISION_MODEL="${ORACLE_LOCAL_VISION_MODEL:-oracle-qwen35-adapted}"

if ! curl --noproxy '*' -fsS --max-time 5 "${ORACLE_VIEWER_MODEL_BASE_URL%/}/health" >/dev/null; then
  echo "Configured local model service is unavailable at ${ORACLE_VIEWER_MODEL_BASE_URL}." >&2
  exit 2
fi

exec .venv/bin/uvicorn talker.app:create_app --factory --host "${ORACLE_TALKER_HOST:-0.0.0.0}" \
  --port "${ORACLE_TALKER_PORT:-7870}" --log-level info
