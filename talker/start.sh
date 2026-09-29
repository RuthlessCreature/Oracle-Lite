#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
fi
python -m talker
