#!/usr/bin/env bash
# Run all 5 baselines sequentially at a given Dirichlet alpha, for a given
# dataset (default cifar10). Designed to be invoked twice (once per alpha)
# in two parallel tmux sessions so the VM runs two baselines at once.
#
# Usage:
#   bash run_alpha.sh 0.1                     # cifar10 (default), tag alpha01
#   bash run_alpha.sh 0.5                     # cifar10 (default), tag alpha05
#   DATASET=fmnist bash run_alpha.sh 0.1      # fmnist, tag alpha01
#   DATASET=fmnist bash run_alpha.sh 0.5      # fmnist, tag alpha05
#
# Both --alpha-server and --alpha-client are set to the same value.
# Extra args after the alpha are forwarded to each train_*.py:
#   bash run_alpha.sh 0.1 --num-workers 6
#
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: bash run_alpha.sh <alpha> [extra args forwarded to train_*.py]"
    echo "       DATASET=fmnist bash run_alpha.sh <alpha> [...]"
    echo "example: bash run_alpha.sh 0.1"
    echo "example: DATASET=fmnist bash run_alpha.sh 0.5"
    exit 2
fi

ALPHA="$1"
shift
TAG="alpha$(echo "$ALPHA" | tr -d '.')"

DATASET=${DATASET:-cifar10}
ORDER=${ORDER:-"HierFAVG MTGC CHPFL PHE-FL ESPerHFL"}

echo "==================================================================="
echo "[run_alpha] dataset=$DATASET  alpha=$ALPHA  tag=$TAG  order=$ORDER"
echo "[run_alpha] forwarded args = $*"
echo "==================================================================="

cd "$(dirname "$0")"
if [ ! -d "$DATASET" ]; then
    echo "[run_alpha] ERROR: dataset folder '$DATASET' not found under $(pwd)"
    exit 3
fi

START_ALL=$(date +%s)
for METHOD in $ORDER; do
    DIR="$DATASET/$METHOD"
    if [ ! -d "$DIR" ]; then
        echo "[run_alpha] WARN: $DIR not found, skipping"
        continue
    fi
    echo
    echo ">>> $METHOD @ $DATASET / alpha=$ALPHA"
    (
        cd "$DIR"
        EXPERIMENT_TAG="$TAG" bash run_5_seeds.sh \
            --alpha-server "$ALPHA" --alpha-client "$ALPHA" "$@"
    )
done
END_ALL=$(date +%s)
echo
echo "==================================================================="
echo "[run_alpha] dataset=$DATASET  alpha=$ALPHA  done in $((END_ALL-START_ALL))s"
echo "[run_alpha] outputs under: $DATASET/gc_results/$TAG/"
echo "==================================================================="
