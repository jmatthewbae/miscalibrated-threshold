#!/usr/bin/env bash
# One-time setup (Linux, macOS).
set -euo pipefail
cd "$(dirname "$0")"

echo "============================================================"
echo " Miscalibrated Threshold experiment - setup"
echo "============================================================"

command -v python3 >/dev/null || { echo "ERROR: python3 not found."; exit 1; }
python3 -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3,10) else 1)' \
    || { echo "ERROR: Python 3.10 or newer required."; exit 1; }

[ -d .venv ] || python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip --quiet

if [[ "$(uname -s)" == "Darwin" ]]; then
    echo "macOS detected. Installing PyTorch with Apple MPS support."
    pip install torch torchvision
else
    echo "Installing PyTorch with CUDA 12.8 support (Blackwell / RTX 50-series)."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
fi

pip install -r requirements.txt

echo
echo "============================================================"
echo " Verifying GPU and data"
echo "============================================================"
python check_setup.py

echo
echo "Setup complete. Now run:  ./run.sh"
