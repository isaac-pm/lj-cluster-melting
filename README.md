# lj-cluster-melting

Project 6 (Computational Methods, Université du Luxembourg): a molecular model of melting. We simulate a small Lennard-Jones cluster (N = 13, Mackay icosahedron) with velocity Verlet and follow the crossover from solid-like to liquid-like behaviour as the energy increases.

## Documents

- [Design notes](docs/design.md): architecture, two-level MPI strategy, units, integrator, diagnostics.
- [Literature review](docs/literature-review.md): Stage 2 survey covering per-paper analysis, cross-paper synthesis, a research-gap map and proposed research directions.
- [Bibliography (BibTeX)](docs/references.bib): generated from Crossref metadata.

## Environment setup

One script, [`setup_env.sh`](setup_env.sh), builds the conda env `lj-melt` (Python 3.13, NumPy, Matplotlib) both on your own machine and on the clusters:

1. it reuses an existing conda install (`~/miniforge3`, `~/miniconda3` or `~/anaconda3`), or installs Miniforge into `~/miniforge3` if there is none. On a fresh install it runs `conda init bash` and turns off auto-activation of `base`;
2. it creates the env from conda-forge only, so Anaconda's `defaults` channel (and its terms of service) is never used;
3. it runs `pip install -r requirements.txt` and checks the imports.

You can re-run the script whenever you like, for example after editing `requirements.txt`: it reuses what already exists and updates the packages. Options:

| Option / variable | Effect                                                                                                                         |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `--gpu`           | also install [`requirements-gpu.txt`](requirements-gpu.txt) (GPU runs on Iris or a local NVIDIA GPU; it lists no packages yet) |
| `--mpi`           | also install OpenMPI + mpi4py from conda-forge (mpi4py has no PyPI wheel, so this must go through conda)                     |
| `--recreate`      | delete the env and build it again                                                                                              |
| `LJ_ENV=name`     | use another env name (pass the same variable when you submit)                                                                  |
| `LJ_PYTHON=3.12`  | choose the Python version                                                                                                      |
| `CONDA_ROOT=path` | use (or install) conda at this path instead of the auto-detected one                                                           |

### Local machine

```bash
git clone git@github.com:isaac-pm/lj-cluster-melting.git
cd lj-cluster-melting
bash setup_env.sh --mpi            # drop --mpi for a serial-only setup
conda activate lj-melt             # or: source ~/miniconda3/etc/profile.d/conda.sh && conda activate lj-melt
python main.py --mode single       # one trajectory + HTML viewer
```

## Usage

```bash
# Single run: trajectory.json + outputs/cluster.html (drag to rotate, space to play)
python main.py --mode single --energies -36 --steps 50000

# Melting scan: full 0.01-resolved sweep, merged caloric curve
python main.py --mode scan --replicas 3

# Custom energies / cluster size / timestep
python main.py --mode scan --energies -43 -40 -36 --num-particles 13 --dt 0.005

# Or: start/end energy grid with a dE step size (endpoint included)
python main.py --mode scan --start-energy -43.0 --end-energy -30.0 --de 0.5 --replicas 3

# Physical LJ parameters (e.g. argon-like): lengths in sigma, energies in epsilon
python main.py --mode single --start-energy -36 --steps 50000 --sigma 2.0 --epsilon 3.0

# Same scan on 8 cores (identical code path, master/worker split)
mpiexec -n 8 python main.py --mode scan --energies -43 -40 -36 --replicas 3
```

Outputs land in `outputs/`:

| mode    | files                                                                                          |
| ------- | ---------------------------------------------------------------------------------------------- |
| single  | `trajectory.json`, `cluster.html` (self-contained CSS 3D viewer with a live in-browser velocity-Verlet MD panel that anneals the cluster from `E_0` to `E_1`, works from `file://`) |
| scan    | `merged.json`, `caloric.csv`, `caloric.png`, `summary.json`, `per_run/e*_r*.csv`               |

`summary.json` reports the transition markers: the heat-capacity peak C = dE/dT and the Lindemann 0.1 crossing (finite-resolution estimates — see design notes).

MPI is optional: without mpi4py installed, everything runs serially. With it, the same script parallelises under `mpiexec` — one process per (energy, replica), rank 0 merges. For large clusters, `lj.physics.AtomDecomposedForces` additionally splits the O(N²) force loop across ranks inside each worker.

## Running on the ULHPC clusters (Aion and Iris)

Aion and Iris share the same `$HOME` (GPFS), so you install the environment **once** and both clusters use it. The two clusters have different hardware:

|                                  | Aion                         | Iris                                                                |
| -------------------------------- | ---------------------------- | ------------------------------------------------------------------- |
| CPU nodes                        | 128 cores (AMD EPYC), 224 GB | 28 cores (Intel), 112 GB                                            |
| GPUs                             | none                         | `gpu` partition: 4× V100 per node, CUDA module `system/CUDA/12.6.0` |
| Default job from `hpc/submit.sh` | full node, exclusive         | 7 cores (CPU) or 1 GPU + 7 cores (GPU)                              |

Iris jobs request less because its GPU queue is usually busy and a small request starts sooner.

### Connecting

```bash
ssh -p 8022 <user>@access-aion.uni.lu
ssh -p 8022 <user>@access-iris.uni.lu
```

### First-time setup (one command, from either login node)

```bash
cd ~
git clone https://github.com/isaac-pm/lj-cluster-melting.git
cd lj-cluster-melting
bash setup_env.sh --mpi            # mpi4py must come from conda-forge
```

Prebuilt wheels (such as `cupy-cuda12x`) install fine from a login node. A package that has to **compile** against CUDA needs the toolkit, which exists only on Iris GPU nodes, so in that case run the setup inside a GPU allocation:

```bash
salloc -p gpu -q normal -N 1 -n 1 -c 7 -G 1 -t 01:00:00
bash setup_env.sh --gpu            # loads system/CUDA/12.6.0 automatically
```

### Submitting `main.py`

Run these from the repo on a login node. [`hpc/submit.sh`](hpc/submit.sh) detects which cluster it's on and picks the resources:

```bash
hpc/submit.sh                               # CPU: full Aion node / 7 cores on Iris, 2 h
hpc/submit.sh gpu                           # Iris only: 1 GPU + 7 cores, 2 h
hpc/submit.sh mpi                           # MPI scan: whole node as ranks (see below)
hpc/submit.sh cpu -t 08:00:00               # extra sbatch options override the profile
hpc/submit.sh -- --mode single --energies -36   # everything after -- goes to main.py
hpc/submit.sh mpi --dry-run                 # print the sbatch command without submitting
```

| Profile    | Partition | Resources                        |
| ---------- | --------- | -------------------------------- |
| Aion `cpu` | `batch`   | 1 node, 128 cores, `--exclusive` |
| Iris `cpu` | `batch`   | 1 task, 7 cores (4 GB/core)      |
| Iris `gpu` | `gpu`     | 1 task, 7 cores, 1 GPU           |
| Aion `mpi`| `batch`   | 1 node, 128 tasks, 1 core each   |
| Iris `mpi` | `batch`   | 1 node, 7 tasks, 1 core each     |

The `mpi` profile runs the scan with `mpiexec -n <tasks> python main.py` (wired through `LJ_MPI=1` in [`hpc/job.sh`](hpc/job.sh)): rank 0 hands out `(energy, seed)` items and merges, workers run independent trajectories. Both profiles run on a single node, so the conda OpenMPI's local launcher is used (no PMIx/`srun` complications).

Logs go to `logs/<job-name>-<jobid>.out`. Check progress with `squeue --me`, and cancel a job with `scancel <jobid>`.

The batch script [`hpc/job.sh`](hpc/job.sh) activates the env and loads CUDA when the job has a GPU. It then prints the node, CPU, GPU and Python details and runs `main.py`. It sets `OMP_NUM_THREADS=1` (LJ clusters are too small to benefit from threaded BLAS) and exports `NCPUS` (the cores allocated to the job).

### Interactive sessions

```bash
# CPU (both clusters; the interactive partition allows up to 2 h)
salloc -p interactive -q normal -N 1 -n 1 -c 4 -t 01:00:00

# GPU (Iris)
salloc -p gpu -q normal -N 1 -n 1 -c 7 -G 1 -t 01:00:00
module load system/CUDA/12.6.0

source ~/miniforge3/etc/profile.d/conda.sh && conda activate lj-melt
mpiexec -n 4 python main.py --mode scan --energies -40 -36 -32 --replicas 2
```
