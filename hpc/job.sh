#!/bin/bash -l
# Slurm batch script that runs main.py in the project env.
#
# Submit it through hpc/submit.sh, which picks resources for the current cluster.
# A direct `sbatch hpc/job.sh [main.py args]` from the repo root uses the
# defaults below: 1 core on the batch partition for 2 hours.
#
#SBATCH --job-name=lj-melt
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

ENV_NAME=${LJ_ENV:-lj-melt}
if [[ -z ${CONDA_ROOT:-} ]]; then
    for dir in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3"; do
        if [[ -x $dir/bin/conda ]]; then CONDA_ROOT=$dir; break; fi
    done
fi
: "${CONDA_ROOT:?no conda install found; run setup_env.sh first}"
CUDA_MODULE=${CUDA_MODULE:-system/CUDA/12.6.0}

cd "${SLURM_SUBMIT_DIR:-.}"

set +u
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
set -u

# GPU jobs (Iris only) get the CUDA toolkit to match the GPU wheels.
GPUS=${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-}}
if [[ -n $GPUS ]]; then
    module load "$CUDA_MODULE"
fi

# One BLAS/OpenMP thread per process by default: LJ clusters are tiny, so
# cores are better spent on parallel runs (e.g. one process per energy).
# main.py can size its worker pool from NCPUS.
export NCPUS=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export MPLBACKEND=Agg

echo "job      : ${SLURM_JOB_NAME:-local} ${SLURM_JOB_ID:-}"
echo "node     : $(hostname) (partition ${SLURM_JOB_PARTITION:-n/a})"
echo "cpus     : $NCPUS  (OMP_NUM_THREADS=$OMP_NUM_THREADS)"
echo "gpus     : ${GPUS:-none}"
echo "python   : $(command -v python)"
echo "args     : $*"
echo "started  : $(date -Is)"
if [[ -n $GPUS ]] && command -v nvidia-smi >/dev/null; then nvidia-smi -L; fi
echo

status=0
python main.py "$@" || status=$?

echo
echo "finished : $(date -Is) (exit $status)"
exit $status
