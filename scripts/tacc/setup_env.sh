#!/bin/bash
# Build the PSPE environment on a TACC login node (login nodes have internet;
# compute nodes may not, so everything must be installed before sbatch).
#
#   bash scripts/tacc/setup_env.sh
#
# Handles both systems the repo has been run on. They differ in the one place
# that matters — the torch wheel:
#   vista  GH200, aarch64 + CUDA 12.x  -> aarch64 wheels from the cu126 index
#   ls6    x86_64 + A100               -> the usual cu124 wheels
#
# Installs into $WORK, not $HOME: $HOME is small-quota on TACC and a torch+CUDA
# tree will fill it.
set -euo pipefail

SYSTEM="${TACC_SYSTEM:-unknown}"
VENV="${VENV:-$WORK/pspe-venv}"
REPO="${REPO:-$WORK/pspe}"

case "$SYSTEM" in
    vista) TORCH_INDEX="https://download.pytorch.org/whl/cu126" ;;
    ls6)   TORCH_INDEX="https://download.pytorch.org/whl/cu124" ;;
    *)     echo "unknown TACC_SYSTEM='$SYSTEM'; defaulting to the PyPI wheel" >&2
           TORCH_INDEX="" ;;
esac

module purge
case "$SYSTEM" in
    # Lmod on Vista is hierarchical: cuda and python3 only become visible
    # once a compiler is loaded, so gcc comes first or both fail to resolve.
    vista) module load gcc/13.2.0 cuda/12.6 python3/3.11.8 ;;
    *)     module load python3 2>/dev/null || true; module load cuda 2>/dev/null || true ;;
esac
# The system python3 is 3.9; the package needs >=3.10, so the module is not optional.
python3 -c 'import sys; assert sys.version_info >= (3, 10), sys.version'

if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

python -m pip install --upgrade pip
if [ -n "$TORCH_INDEX" ]; then
    python -m pip install --index-url "$TORCH_INDEX" torch
else
    python -m pip install torch
fi
python -m pip install -r "$REPO/requirements.txt"
python -m pip install -e "$REPO"

# A CPU-only wheel installs and imports perfectly happily; only this check
# distinguishes it from a working GPU build, and get_device() would silently
# fall back to CPU rather than raise.
# Login nodes have no GPU, so `available False` here proves nothing either way —
# only the CUDA build string does. The compute-node assert lives in the job
# script, where a CPU-only wheel must fail loudly rather than quietly produce
# CPU timings under a GPU label.
python - <<'PY'
import platform, torch
print("machine", platform.machine())
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
assert torch.version.cuda, "CPU-only wheel: no CUDA in this build. Reinstall from the cu126 index."
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
else:
    print("no GPU visible here (expected on a login node); the job script asserts on the compute node")
PY
echo "env ready: $VENV"
