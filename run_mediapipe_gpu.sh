#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${SCRIPT_DIR}/venv/bin/activate" ]]; then
  # Reuse the project virtualenv when it exists.
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/venv/bin/activate"
fi

export __NV_PRIME_RENDER_OFFLOAD="${__NV_PRIME_RENDER_OFFLOAD:-1}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export OPENCV_LOG_LEVEL="${OPENCV_LOG_LEVEL:-ERROR}"

if [[ -d /usr/share/fonts/truetype/dejavu ]]; then
  export QT_QPA_FONTDIR="${QT_QPA_FONTDIR:-/usr/share/fonts/truetype/dejavu}"
fi

exec python3 "${SCRIPT_DIR}/app.py" \
  --backend mediapipe \
  --mediapipe_delegate gpu \
  "$@"

# ./run_mediapipe_gpu.sh --device 1 --rosbridge_enable --filter ema
