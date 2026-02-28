#!/usr/bin/env bash
set -euo pipefail

# Resolve to the directory where this script lives so relative paths work.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate uv-managed virtual environment if present.
UV_VENV="/workspace/my_deltamodel/.venv"
if [ -f "$UV_VENV/bin/activate" ]; then
	# shellcheck source=/dev/null
	source "$UV_VENV/bin/activate"
fi

# Allow overriding the Python interpreter if needed (e.g., PYTHON=python).
: "${PYTHON:=python3}"

echo "[1/4] Running calibrate_models.py..."
$PYTHON "$SCRIPT_DIR/calibrate_models.py"

echo "[2/4] Running calibrate_models_invkge.py..."
$PYTHON "$SCRIPT_DIR/calibrate_models_invkge.py"

echo "[3/4] Running calibrate_special_models.py..."
$PYTHON "$SCRIPT_DIR/calibrate_special_models.py"

echo "[4/4] Running calibrate_special_models_invkge.py..."
$PYTHON "$SCRIPT_DIR/calibrate_special_models_invkge.py"

echo "All calibration scripts finished."
