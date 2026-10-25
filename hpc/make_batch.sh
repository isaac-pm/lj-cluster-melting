#!/bin/bash
# Self-assembling Slurm batch script generator: you give it the *physics*
# parameters, it translates them into an appropriately sized sbatch script
# and submits it (or just writes it: --write-only / --dry-run).
#
# Sizing rules:
#   work items  = (energy-grid size) x replicas        [scan; 1 for single]
#   ranks       = 1 master + 1 worker per run, rounded up to whole nodes
#                 (aion: 128 cores/node, iris: 7)
#   walltime    = steps*runs / (rate x workers) + ~20% slack, >= 30 min
#                 (rate measured at 45000 steps/s/rank for N=13; scales
#                 with the O(N^2) force loop when --num-particles differs)
#
# Usage:
#   hpc/make_batch.sh [sizing options] [main.py options] [-- raw main.py args]
#
# Sizing / resource options (consumed here, not forwarded):
#   --mode scan|single     (default: scan)
#   --ranks N              override the computed rank count
#   --nodes N              override the computed node count
#   --partition P          (default: batch)
#   --qos Q                (default: normal)
#   --time HH:MM:SS        override the estimated walltime
#   --rate S               measured steps/s per rank (default: 45000)
#   --env NAME             conda env in the script (default: lj-melt)
#   --job-name NAME        (default: lj-melt-<mode>)
#   --write-only           write the script, don't submit
#   --dry-run              print the sbatch command instead of submitting
#
# Physics options (forwarded to main.py and used for sizing):
#   --start-energy E0 --end-energy E1 --de dE   (energy grid, E1 included)
#   --energies E1 E2 ...                        (explicit list)
#   --replicas R --steps S --dt DT --num-particles N
#   --sample-every M --cutoff C --seed S --outdir D
#   --thermostat berendsen --target-temperature T --tau-t TT
#
# Examples:
#   hpc/make_batch.sh --start-energy -43 --end-energy -24 --de 0.5 --replicas 3
#   hpc/make_batch.sh --start-energy -43 --end-energy -30 --de 0.25 \
#                     --replicas 2 --steps 200000
#   hpc/make_batch.sh --mode single --start-energy -36 --steps 50000 --write-only
#   hpc/make_batch.sh --start-energy -43 --end-energy -24 --de 1 -- --dt 0.004
#   hpc/make_batch.sh -h            # this help
set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# ------------------------------- defaults -------------------------------
MODE=scan RANKS= NODES= PARTITION=batch QOS=normal TIME= RATE=45000
ENV_NAME=${LJ_ENV:-lj-melt} JOB_NAME= WRITE_ONLY=0 DRY_RUN=0
START= END= DE= ELIST=0 REPS=3 STEPS=100000 NPART=13
MAIN=()                       # forwarded main.py args, token by token

die()  { echo "make_batch.sh: $*" >&2; exit 1; }
help() { sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }
need() { [[ $# -ge 2 ]] || die "$1 requires a value"; }

# ----------------------------------------------------------------- parse
while (($#)); do
    case $1 in
        # sizing / resource options
        --mode)        need "$@"; MODE=$2; shift 2 ;;
        --ranks)       need "$@"; RANKS=$2; shift 2 ;;
        --nodes)       need "$@"; NODES=$2; shift 2 ;;
        --partition)   need "$@"; PARTITION=$2; shift 2 ;;
        --qos)         need "$@"; QOS=$2; shift 2 ;;
        --time)        need "$@"; TIME=$2; shift 2 ;;
        --rate)        need "$@"; RATE=$2; shift 2 ;;
        --env)         need "$@"; ENV_NAME=$2; shift 2 ;;
        --job-name)    need "$@"; JOB_NAME=$2; shift 2 ;;
        --write-only)  WRITE_ONLY=1; shift ;;
        --dry-run)     DRY_RUN=1; shift ;;
        -h|--help)     help ;;
        --) shift; while (($#)); do MAIN+=("$1"); shift; done; break ;;
        # physics flags with a single value: forward (also seed sizing)
        --start-energy|--end-energy|--de|--replicas|--steps|--dt|\
        --num-particles|--sample-every|--cutoff|--seed|--outdir|\
        --sigma|--epsilon|--thermostat|--target-temperature|--tau-t)
            need "$@"
            case $1 in
                --start-energy)   START=$2 ;;
                --end-energy)     END=$2 ;;
                --de)             DE=$2 ;;
                --replicas)       REPS=$2 ;;
                --steps)          STEPS=$2 ;;
                --num-particles)  NPART=$2 ;;
                --mode)           MODE=$2 ;;
            esac
            MAIN+=("$1" "$2"); shift 2 ;;
        # physics flag with a value list: forward verbatim. The numeric
        # consumption must happen HERE (outer scope) — a function's shift
        # does not propagate to the caller in bash.
        --energies)
            ELIST=1; need "$@"
            MAIN+=("$1"); shift
            while (($#)) && [[ $1 =~ ^-?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]]; do
                MAIN+=("$1"); shift
            done ;;
        *) die "unknown option: $1 (use -- to pass raw args)" ;;
    esac
done

# ------------------------------------------------- size the work grid
ITEMS=1901    # main.py's default preset sweep (0.01 resolution)
if [[ $MODE == single ]]; then
    ITEMS=1
elif [[ $ELIST == 1 ]]; then
    # count the numbers following --energies
    ITEMS=0; seen=0
    for tok in "${MAIN[@]}"; do
        if [[ $tok == --energies ]]; then seen=1; continue; fi
        if [[ $seen == 1 ]]; then
            if [[ $tok =~ ^-?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then
                ITEMS=$((ITEMS + 1))
            else
                break
            fi
        fi
    done
    (( ITEMS == 0 )) && ITEMS=1
elif [[ -n $START && -n $END ]]; then
    DE=${DE:-0.01}
    ITEMS=$(python3 - "$START" "$END" "$DE" <<'PY'
import sys
e0, e1, de = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
if de <= 0: sys.exit("--de must be positive")
print(max(1, int(round(abs(e1 - e0) / de)) + 1))
PY
    )
elif [[ -n $START || -n $END ]]; then
    die "--start-energy and --end-energy must be given together"
fi
RUNS=$((ITEMS * REPS))
echo "work grid: $ITEMS energy window(s) x $REPS replica(s) = $RUNS run(s)"

# ---------------------------------------------------------- size ranks
CORES_PER_NODE=128
case $(hostname) in
    *iris*) CORES_PER_NODE=7 ;;
    *aion*) CORES_PER_NODE=128 ;;
esac
if [[ -n $RANKS ]]; then
    NTASKS=$RANKS
elif [[ -n $NODES ]]; then
    NTASKS=$((NODES * CORES_PER_NODE))
else
    # single mode is ONE trajectory: extra ranks would be idle (workers exit
    # immediately), so book the master core only unless ranks were forced
    if [[ $MODE == single ]]; then
        NTASKS=1
    else
        NODES=$(( (RUNS + 1 + CORES_PER_NODE - 1) / CORES_PER_NODE ))   # +1 master
        NTASKS=$((NODES * CORES_PER_NODE))
    fi
fi
(( NTASKS >= 1 )) || die "need at least one rank"

# ------------------------------------------------------- size walltime
clock() { printf '%02d:%02d:%02d' $((($1 / 3600) % 100)) $((($1 % 3600) / 60)) $(($1 % 60)); }
if [[ -z $TIME ]]; then
    WORKERS=$(( NTASKS > 1 ? NTASKS - 1 : 1 ))
    RATE_N=$(( RATE * 169 / (NPART * NPART) ))    # rate ~ 1/N^2 in the pair loop
    SECS=$(( (STEPS * RUNS * 130) / (RATE_N * WORKERS * 100) + 600 ))
    (( SECS < 1800 ))   && SECS=1800     # >= 30 min
    (( SECS > 172800 )) && SECS=172800   # <= 48 h
    TIME=$(clock "$SECS")
    echo "estimated walltime: $TIME  (~$RATE_N steps/s/rank x $WORKERS workers)"
fi
MPI_LAUNCH=(python main.py)
if (( NTASKS > 1 )); then MPI_LAUNCH=(mpiexec -n "$NTASKS" python main.py); fi

# ----------------------------------------------------- assemble script
mkdir -p "$REPO_DIR/hpc/generated" "$REPO_DIR/logs"
NAME=${JOB_NAME:-lj-melt-$MODE}
OUT="$REPO_DIR/hpc/generated/$NAME-$(date +%H%M%S).sbatch"

{
    echo '#!/bin/bash -l'
    echo "# Auto-generated by hpc/make_batch.sh at $(date -Is)"
    echo "# physics: start=${START:-default} end=${END:-default} de=${DE:-default} replicas=$REPS steps=$STEPS N=$NPART"
    echo "# work:    $ITEMS energy window(s) x $REPS = $RUNS run(s) on $NTASKS rank(s)"
    echo "#SBATCH --job-name=$NAME"
    echo "#SBATCH --partition=$PARTITION"
    echo "#SBATCH --qos=$QOS"
    echo "#SBATCH --nodes=${NODES:-1}"
    echo "#SBATCH --ntasks=$NTASKS"
    echo '#SBATCH --cpus-per-task=1'
    echo "#SBATCH --time=$TIME"
    echo "#SBATCH --output=$REPO_DIR/logs/%x-%j.out"
    cat <<BODY

set -euo pipefail
cd "\$SLURM_SUBMIT_DIR"

ENV_NAME=$ENV_NAME
if [[ -z \${CONDA_ROOT:-} ]]; then
    for dir in "\$HOME/miniforge3" "\$HOME/miniconda3" "\$HOME/anaconda3"; do
        [[ -x \$dir/bin/conda ]] && CONDA_ROOT=\$dir && break
    done
fi
: "\${CONDA_ROOT:?no conda found, run setup_env.sh first}"
set +u; source "\$CONDA_ROOT/etc/profile.d/conda.sh"; conda activate "\$ENV_NAME"; set -u

export OMP_NUM_THREADS=\${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=\$OMP_NUM_THREADS
export MKL_NUM_THREADS=\$OMP_NUM_THREADS
export MPLBACKEND=Agg
mkdir -p "\$SLURM_SUBMIT_DIR/logs"

echo "job \${SLURM_JOB_ID:-local}: \$(hostname), \${SLURM_NTASKS:-1} ranks, started \$(date -Is)"
echo
BODY
    printf '%q ' "${MPI_LAUNCH[@]}" "${MAIN[@]}"
    echo
    echo 'echo; echo "finished $(date -Is)"'
} > "$OUT"
chmod +x "$OUT"

echo "wrote $OUT"
if [[ $DRY_RUN == 1 ]]; then
    echo "dry-run: sbatch $OUT"
    exit 0
fi
if [[ $WRITE_ONLY == 1 ]]; then exit 0; fi
sbatch "$OUT"
