#!/usr/bin/env bash
# Run MTGC on CIFAR-10 for 5 seeds. Each seed writes a self-contained folder
# under ../gc_results/mtgc_seed{seed}/. After all seeds finish, zip the
# gc_results directory to download.
#
# Usage:
#   bash run_5_seeds.sh                       # all defaults
#   bash run_5_seeds.sh --global-rounds 50    # forward args to train_mtgc.py
#
# Env overrides:
#   PYTHON=python3.11 SEEDS="42 43 44" bash run_5_seeds.sh
set -euo pipefail

cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3}
SEEDS=${SEEDS:-"42 43 44 45 46"}

EXPERIMENT_TAG=${EXPERIMENT_TAG:-""}    # e.g. "alpha01" or "alpha05"; empty = legacy flat layout
if [ -n "$EXPERIMENT_TAG" ]; then
    GC_RESULTS_DIR="$(realpath ../gc_results)/${EXPERIMENT_TAG}"
else
    GC_RESULTS_DIR="$(realpath ../gc_results)"
fi
mkdir -p "$GC_RESULTS_DIR"

echo "[run_5_seeds] python=$PYTHON   seeds=$SEEDS"
echo "[run_5_seeds] outputs -> $GC_RESULTS_DIR"
echo "[run_5_seeds] extra args = $*"
echo

START_ALL=$(date +%s)
for SEED in $SEEDS; do
    OUT="$GC_RESULTS_DIR/mtgc_seed${SEED}"
    LOG="$GC_RESULTS_DIR/mtgc_seed${SEED}.log"
    echo "============================================================"
    echo "[run_5_seeds] seed=$SEED  out=$OUT"
    echo "============================================================"
    START=$(date +%s)
    "$PYTHON" train_mtgc.py --seed "$SEED" --out-dir "$OUT" "$@" 2>&1 | tee "$LOG"
    END=$(date +%s)
    echo "[run_5_seeds] seed=$SEED done in $((END-START))s"
    echo
done
END_ALL=$(date +%s)
echo "[run_5_seeds] all 5 seeds done in $((END_ALL-START_ALL))s"
echo
echo "[run_5_seeds] gc_results contents:"
ls -la "$GC_RESULTS_DIR"
echo
echo "[run_5_seeds] to zip for download:"
echo "  (cd $(dirname "$GC_RESULTS_DIR") && zip -r mtgc_cifar10_5seeds.zip gc_results)"
