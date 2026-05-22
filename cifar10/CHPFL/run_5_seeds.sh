#!/usr/bin/env bash
# Run CHPFL on CIFAR-10 for 5 seeds. Each seed writes a self-contained
# folder under ../gc_results/chpfl_seed{seed}/.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3}
SEEDS=${SEEDS:-"42 43 44 45 46"}

GC_RESULTS_DIR="$(realpath ../gc_results)"
mkdir -p "$GC_RESULTS_DIR"

echo "[run_5_seeds] python=$PYTHON   seeds=$SEEDS"
echo "[run_5_seeds] outputs -> $GC_RESULTS_DIR"
echo "[run_5_seeds] extra args = $*"
echo

START_ALL=$(date +%s)
for SEED in $SEEDS; do
    OUT="$GC_RESULTS_DIR/chpfl_seed${SEED}"
    LOG="$GC_RESULTS_DIR/chpfl_seed${SEED}.log"
    echo "============================================================"
    echo "[run_5_seeds] seed=$SEED  out=$OUT"
    echo "============================================================"
    START=$(date +%s)
    "$PYTHON" train_chpfl.py --seed "$SEED" --out-dir "$OUT" "$@" 2>&1 | tee "$LOG"
    END=$(date +%s)
    echo "[run_5_seeds] seed=$SEED done in $((END-START))s"
    echo
done
END_ALL=$(date +%s)
echo "[run_5_seeds] all 5 seeds done in $((END_ALL-START_ALL))s"
echo
echo "[run_5_seeds] to zip for download:"
echo "  (cd $(dirname "$GC_RESULTS_DIR") && zip -r chpfl_cifar10_5seeds.zip gc_results)"
