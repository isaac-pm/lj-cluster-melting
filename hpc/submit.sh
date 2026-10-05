#!/bin/bash
# Submit main.py as a Slurm job with resources chosen for the current cluster.
#
#   hpc/submit.sh [cpu|gpu] [sbatch options...] [-- main.py args...]
#
# Profiles:
#   aion cpu   full node: 128 cores, ~224 GB, exclusive    (partition batch)
#   iris cpu   7 cores, 28 GB, shared node                 (partition batch)
#   iris gpu   1 V100 GPU + 7 cores                        (partition gpu)
#   aion gpu   not available (Aion has no GPUs)
#
# Any sbatch options given here override the profile, for example:
#   hpc/submit.sh cpu -t 08:00:00 -- --n-atoms 13
#   hpc/submit.sh gpu -t 00:30:00 -J quick-test
#   hpc/submit.sh --dry-run            # print the sbatch command only

set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

PROFILE=cpu
DRY_RUN=0
SBATCH_ARGS=()
MAIN_ARGS=()

if [[ ${1:-} == cpu || ${1:-} == gpu ]]; then
    PROFILE=$1
    shift
fi
while (($#)); do
    case $1 in
        --) shift; MAIN_ARGS=("$@"); break ;;
        --dry-run) DRY_RUN=1 ;;
        -h|--help) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) SBATCH_ARGS+=("$1") ;;
    esac
    shift
done

case $(hostname) in
    *aion*) CLUSTER=aion ;;
    *iris*) CLUSTER=iris ;;
    *) echo "Run this from an Aion or Iris login node (hostname: $(hostname))." >&2; exit 1 ;;
esac

case $CLUSTER-$PROFILE in
    aion-cpu)
        RESOURCES=(--partition=batch --qos=normal --nodes=1 --ntasks=1
                   --cpus-per-task=128 --exclusive --time=02:00:00) ;;
    iris-cpu)
        RESOURCES=(--partition=batch --qos=normal --nodes=1 --ntasks=1
                   --cpus-per-task=7 --time=02:00:00) ;;
    iris-gpu)
        RESOURCES=(--partition=gpu --qos=normal --nodes=1 --ntasks=1
                   --cpus-per-task=7 --gpus=1 --time=02:00:00) ;;
    aion-gpu)
        echo "Aion has no GPUs; submit GPU jobs from Iris (or use 'cpu' here)." >&2
        exit 1 ;;
esac

mkdir -p "$REPO_DIR/logs"
CMD=(sbatch --job-name="lj-melt-$CLUSTER-$PROFILE" --chdir="$REPO_DIR"
     --output="$REPO_DIR/logs/%x-%j.out"
     "${RESOURCES[@]}" "${SBATCH_ARGS[@]}"
     "$REPO_DIR/hpc/job.sh" "${MAIN_ARGS[@]}")

if [[ $DRY_RUN == 1 ]]; then
    printf '%q ' "${CMD[@]}"
    echo
    exit 0
fi

"${CMD[@]}"
echo "Logs: $REPO_DIR/logs/   (watch with: squeue --me)"
