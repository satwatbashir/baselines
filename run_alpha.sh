#!/usr/bin/env bash
# Run all 5 CIFAR-10 baselines sequentially at a given Dirichlet alpha.
# Designed to be invoked twice (once per alpha) in two parallel tmux
# sessions so the VM runs two baselines at once (one per session).
#
# Usage:
#   bash run_alpha.sh 0.1     # tags outputs as ../gc_results/alpha01/
#   bash run_alpha.sh 0.5     # tags outputs as ../gc_results/alpha05/
#
# Both --alpha-server and --alpha-client are set to the same value.
# Extra args after the alpha are forwarded to each train_*.py:
#   bash run_alpha.sh 0.1 --num-workers 6 --global-rounds 100
#
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: bash run_alpha.sh <alpha> [extra args forwarded to train_*.py]"
    echo "example: bash run_alpha.sh 0.1"
    exit 2
fi

ALPHA="$1"
shift
# alpha=0.1 -> tag alpha01; alpha=0.5 -> alpha05; alpha=0.05 -> alpha005
TAG="alpha$(echo "$ALPHA" | tr -d '.')"

ORDER=${ORDER:-"HierFAVG MTGC CHPFL PHE-FL ESPerHFL"}

echo "==================================================================="
echo "[run_alpha] alpha=$ALPHA  tag=$TAG  order=$ORDER"
echo "[run_alpha] forwarded args = $*"
echo "==================================================================="

cd "$(dirname "$0")"
START_ALL=$(date +%s)
for METHOD in $ORDER; do
    DIR="cifar10/$METHOD"
    if [ ! -d "$DIR" ]; then
        echo "[run_alpha] WARN: $DIR not found, skipping"
        continue
    fi
    echo
    echo ">>> $METHOD @ alpha=$ALPHA"
    (
        cd "$DIR"
        EXPERIMENT_TAG="$TAG" bash run_5_seeds.sh \
            --alpha-server "$ALPHA" --alpha-client "$ALPHA" "$@"
    )
done
END_ALL=$(date +%s)
echo
echo "==================================================================="
echo "[run_alpha] alpha=$ALPHA done in $((END_ALL-START_ALL))s"
echo "[run_alpha] outputs under: cifar10/gc_results/$TAG/"
echo "==================================================================="
