#!/bin/bash
# One-shot environment setup: local machine or the ULHPC clusters (Aion/Iris).
#
# Aion and Iris share the same $HOME, so one run there serves both clusters.
#
#   bash setup_env.sh            # NumPy + Matplotlib (requirements.txt)
#   bash setup_env.sh --gpu      # also requirements-gpu.txt (Iris or local NVIDIA GPU)
#   bash setup_env.sh --recreate # delete and rebuild the env
#
# Uses an existing conda (~/miniforge3, ~/miniconda3, ~/anaconda3) if found,
# otherwise installs Miniforge into ~/miniforge3. Re-running is safe.
# Overrides: LJ_ENV (env name, default lj-melt), LJ_PYTHON (default 3.13),
# CONDA_ROOT (conda install to use or create).

set -euo pipefail

ENV_NAME=${LJ_ENV:-lj-melt}
PYTHON_VERSION=${LJ_PYTHON:-3.13}
CUDA_MODULE=${CUDA_MODULE:-system/CUDA/12.6.0}
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

WITH_GPU=0
RECREATE=0
for arg in "$@"; do
    case $arg in
        --gpu) WITH_GPU=1 ;;
        --recreate) RECREATE=1 ;;
        -h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done

log() { printf '\n==> %s\n' "$*"; }

case $(hostname) in
    *aion*) MACHINE=aion ;;
    *iris*) MACHINE=iris ;;
    *) MACHINE=local ;;
esac
log "Machine: $MACHINE ($(hostname)), repo: $REPO_DIR"

# 1) Conda: reuse an existing install, else install Miniforge
if [[ -z ${CONDA_ROOT:-} ]]; then
    for dir in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3"; do
        if [[ -x $dir/bin/conda ]]; then CONDA_ROOT=$dir; break; fi
    done
fi
CONDA_ROOT=${CONDA_ROOT:-$HOME/miniforge3}

if [[ ! -x "$CONDA_ROOT/bin/conda" ]]; then
    log "Installing Miniforge into $CONDA_ROOT"
    installer=$(mktemp "${TMPDIR:-/tmp}/miniforge.XXXXXX")
    url=https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh
    if command -v wget >/dev/null; then wget -qO "$installer" "$url"; else curl -fsSL -o "$installer" "$url"; fi
    bash "$installer" -b -p "$CONDA_ROOT"
    rm -f "$installer"
    # Make `conda activate` work in future interactive shells, without
    # activating base on every login.
    "$CONDA_ROOT/bin/conda" init bash
    "$CONDA_ROOT/bin/conda" config --set auto_activate false
    echo "Miniforge installed; conda is now initialised in ~/.bashrc."
else
    log "Reusing conda at $CONDA_ROOT"
fi

# conda's shell functions reference unset variables, so relax `set -u` here.
set +u
source "$CONDA_ROOT/etc/profile.d/conda.sh"

# 2) Conda env (conda-forge only, so Anaconda's `defaults` channel and its
# terms of service never come into play, even with Miniconda/Anaconda)
if [[ $RECREATE == 1 ]] && conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    log "Removing existing env $ENV_NAME"
    conda env remove -y -n "$ENV_NAME"
fi
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    log "Creating env $ENV_NAME (python=$PYTHON_VERSION)"
    conda create -y -n "$ENV_NAME" -c conda-forge --override-channels "python=$PYTHON_VERSION" pip
else
    log "Reusing env $ENV_NAME"
fi
conda activate "$ENV_NAME"
set -u

# 3) Python packages
log "Installing requirements.txt"
python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade -r "$REPO_DIR/requirements.txt"

if [[ $WITH_GPU == 1 ]]; then
    # Packages that build against CUDA need the toolkit: on the clusters it
    # only exists on Iris GPU nodes (as a module); locally it is whatever CUDA
    # you have installed. Prebuilt wheels install fine anywhere.
    if command -v module >/dev/null 2>&1 && module is-avail "$CUDA_MODULE" 2>/dev/null; then
        module load "$CUDA_MODULE"
        log "Loaded $CUDA_MODULE"
    fi
    if grep -qv '^\s*\(#\|$\)' "$REPO_DIR/requirements-gpu.txt"; then
        log "Installing requirements-gpu.txt"
        python -m pip install --upgrade -r "$REPO_DIR/requirements-gpu.txt"
    else
        log "requirements-gpu.txt lists no packages yet; nothing extra to install"
    fi
fi

# 4) Smoke test
log "Checking imports"
python - <<'EOF'
import sys
import matplotlib
import numpy
print(f"python {sys.version.split()[0]}, numpy {numpy.__version__}, matplotlib {matplotlib.__version__}")
EOF

log "Done. Env '$ENV_NAME' is ready."
echo "Activate it with: source $CONDA_ROOT/etc/profile.d/conda.sh && conda activate $ENV_NAME"
if [[ $MACHINE != local ]]; then
    echo "Submit main.py:   hpc/submit.sh  (CPU)   |   hpc/submit.sh gpu  (Iris only)"
fi
