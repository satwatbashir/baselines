#!/usr/bin/env bash
# One-shot setup for a fresh Ubuntu 22.04 GCP VM (g2-standard-8 + 1x L4).
#
# Run as the default user (NOT root). Reboots once after the driver install.
# After reboot, re-run this script to complete the Python setup.
#
#   bash setup_vm.sh                # full setup
#   bash setup_vm.sh --skip-driver  # if driver already installed
#
set -euo pipefail

SKIP_DRIVER=0
for arg in "$@"; do
    case "$arg" in
        --skip-driver) SKIP_DRIVER=1 ;;
        *) echo "unknown arg: $arg"; exit 2 ;;
    esac
done

REPO_URL="https://github.com/satwatbashir/baselines.git"
REPO_DIR="$HOME/baselines"

is_installed() { command -v "$1" >/dev/null 2>&1; }
has_gpu() { is_installed nvidia-smi && nvidia-smi >/dev/null 2>&1; }

echo "[setup_vm] step 1/4: apt packages"
sudo apt-get update -y
sudo apt-get install -y \
    git tmux htop unzip wget curl \
    python3 python3-pip python3-venv \
    build-essential

if [ "$SKIP_DRIVER" -eq 0 ] && ! has_gpu; then
    echo "[setup_vm] step 2/4: installing NVIDIA driver (will reboot)"
    sudo apt-get install -y "linux-headers-$(uname -r)" dkms gcc make
    sudo apt-get install -y nvidia-driver-535
    echo
    echo "[setup_vm] driver installed. Rebooting in 5s..."
    echo "[setup_vm] After reboot, re-run:  bash setup_vm.sh --skip-driver"
    sleep 5
    sudo reboot
    exit 0
else
    echo "[setup_vm] step 2/4: GPU already visible to driver"
    nvidia-smi | head -15 || true
fi

echo "[setup_vm] step 3/4: cloning / updating repo"
if [ -d "$REPO_DIR/.git" ]; then
    git -C "$REPO_DIR" pull --ff-only
else
    git clone "$REPO_URL" "$REPO_DIR"
fi

echo "[setup_vm] step 4/4: Python deps"
cd "$REPO_DIR"
# Use a venv to avoid clashing with system Python
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
# CUDA wheel for torch (12.1 toolkit ships with nvidia-driver-535)
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
pip install -r requirements.txt

echo
echo "[setup_vm] DONE. To run the MTGC 5-seed sweep:"
echo
echo "  source ~/baselines/.venv/bin/activate"
echo "  cd ~/baselines/cifar10/MTGC"
echo "  tmux new -s mtgc"
echo "  bash run_5_seeds.sh"
echo
echo "  # Then in another shell, monitor with:"
echo "  tail -f ~/baselines/cifar10/gc_results/mtgc_seed42.log"
echo
echo "[setup_vm] After sweep, package + download:"
echo "  cd ~/baselines/cifar10 && zip -r mtgc_cifar10_5seeds.zip gc_results"
