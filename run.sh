#!/usr/bin/env bash
# Run the full experiment. Run setup.sh first.
set -euo pipefail
cd "$(dirname "$0")"

[ -x .venv/bin/python ] || { echo "No virtual environment. Run ./setup.sh first."; exit 1; }
# shellcheck disable=SC1091
source .venv/bin/activate

echo "============================================================"
echo " Running the full experiment"
echo " A transcript is written to outputs/run_log.txt"
echo "============================================================"

python -u run_all.py "$@"

echo
echo "Finished."
echo "  Table   outputs/results/ddi_results.csv"
echo "  Gaps    outputs/results/gap_summary.json"
echo "  Figures outputs/figures/"
