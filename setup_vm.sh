#!/usr/bin/env bash
# One-shot setup for a fresh Ubuntu 22.04 GCP VM.
# Auto-detects whether NVIDIA GPU hardware is actually present:
#   * GPU present, driver missing -> installs nvidia-driver-535 + reboots
#   * GPU present, driver loaded  -> skips driver step
#   * No GPU hardware             -> skips driver step entirely (installs
#                                    the CPU torch wheel instead)
# Pass --skip-driver to force the post-reboot path on GPU machines.
#
#   bash setup_vm.sh                # full setup
#   bash setup_vm.sh --skip-driver  # GPU VM after reboot
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

is_installed()         { command -v "$1" >/dev/null 2>&1; }
has_gpu_hardware()     { lspci 2>/dev/null | grep -qi 'nvidia'; }
has_gpu_driver_loaded(){ is_installed nvidia-smi && nvidia-smi >/dev/null 2>&1; }

echo "[setup_vm] step 1/4: apt packages"
sudo apt-get update -y
sudo apt-get install -y \
    git tmux htop unzip wget curl pciutils \
    python3 python3-pip python3-venv \
    build-essential

# ---- step 2: NVIDIA driver (only if NVIDIA HW present) ---------------------
if has_gpu_hardware && [ "$SKIP_DRIVER" -eq 0 ] && ! has_gpu_driver_loaded; then
    echo "[setup_vm] step 2/4: NVIDIA GPU detected and no driver loaded — installing"
    sudo apt-get install -y "linux-headers-$(uname -r)" dkms gcc make
    sudo apt-get install -y nvidia-driver-535
    echo
    echo "[setup_vm] driver installed. Rebooting in 5s..."
    echo "[setup_vm] After reboot, re-run:  bash setup_vm.sh --skip-driver"
    sleep 5
    sudo reboot
    exit 0
elif has_gpu_driver_loaded; then
    echo "[setup_vm] step 2/4: GPU driver already loaded"
    nvidia-smi | head -15
elif has_gpu_hardware; then
    echo "[setup_vm] step 2/4: NVIDIA HW present but --skip-driver passed; assuming driver setup is handled"
else
    echo "[setup_vm] step 2/4: no NVIDIA GPU detected -> CPU-only setup"
fi

# ---- step 3: clone repo (idempotent) ---------------------------------------
echo "[setup_vm] step 3/4: cloning / updating repo"
if [ -d "$REPO_DIR/.git" ]; then
    git -C "$REPO_DIR" pull --ff-only
else
    git clone "$REPO_URL" "$REPO_DIR"
fi

# ---- step 4: Python venv + torch (CPU or CUDA wheel per hardware) ----------
echo "[setup_vm] step 4/4: Python deps"
cd "$REPO_DIR"
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
if has_gpu_driver_loaded; then
    echo "[setup_vm]   installing CUDA 12.1 torch wheel"
    pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
else
    echo "[setup_vm]   installing CPU torch wheel"
    pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
fi
pip install -r requirements.txt

echo
echo "[setup_vm] DONE."
if has_gpu_driver_loaded; then
    echo "[setup_vm] GPU run example:"
else
    echo "[setup_vm] CPU run example (dual-alpha sweep in two tmux sessions):"
fi
echo
echo "  source ~/baselines/.venv/bin/activate"
echo "  # session 1:"
echo "  tmux new -s a01"
echo "  cd ~/baselines && bash run_alpha.sh 0.1"
echo "  # session 2 (in another SSH window):"
echo "  tmux new -s a05"
echo "  cd ~/baselines && bash run_alpha.sh 0.5"
echo
echo "[setup_vm] After sweep, package + download:"
echo "  cd ~/baselines/cifar10 && zip -r baselines_cifar10_dual_alpha.zip gc_results"
