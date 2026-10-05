# lj-cluster-melting

Project 6 (Computational Methods, Université du Luxembourg): a molecular model of melting. We simulate a small Lennard-Jones cluster (N = 13) with velocity Verlet and follow the crossover from solid-like to liquid-like behaviour as the energy increases.

## Documents

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
| `--recreate`      | delete the env and build it again                                                                                              |
| `LJ_ENV=name`     | use another env name (pass the same variable when you submit)                                                                  |
| `LJ_PYTHON=3.12`  | choose the Python version                                                                                                      |
| `CONDA_ROOT=path` | use (or install) conda at this path instead of the auto-detected one                                                           |

### Local machine

```bash
git clone git@github.com:isaac-pm/lj-cluster-melting.git
cd lj-cluster-melting
bash setup_env.sh
conda activate lj-melt      # or: source ~/miniconda3/etc/profile.d/conda.sh && conda activate lj-melt
python main.py
```

If your machine has an NVIDIA GPU, use `bash setup_env.sh --gpu` once GPU packages are listed in `requirements-gpu.txt`. Those packages use CUDA 12 wheels: Iris provides CUDA 12.6, and newer drivers (`nvidia-smi` reports the highest CUDA version they support) also run CUDA 12 code, so one requirements file works in both places.

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
bash setup_env.sh
```

Prebuilt wheels (such as `cupy-cuda12x`) install fine from a login node. A package that has to **compile** against CUDA needs the toolkit, which exists only on Iris GPU nodes, so in that case run the setup inside a GPU allocation:

```bash
salloc -p gpu -q normal -N 1 -n 1 -c 7 -G 1 -t 01:00:00
bash setup_env.sh --gpu     # loads system/CUDA/12.6.0 automatically
```

### Submitting `main.py`

Run these from the repo on a login node. [`hpc/submit.sh`](hpc/submit.sh) detects which cluster it's on and picks the resources:

```bash
hpc/submit.sh                               # CPU: full Aion node / 7 cores on Iris, 2 h
hpc/submit.sh gpu                           # Iris only: 1 GPU + 7 cores, 2 h
hpc/submit.sh cpu -t 08:00:00               # extra sbatch options override the profile
hpc/submit.sh -- --some-flag value          # everything after -- goes to main.py
hpc/submit.sh gpu --dry-run                 # print the sbatch command without submitting
```

| Profile    | Partition | Resources                        |
| ---------- | --------- | -------------------------------- |
| Aion `cpu` | `batch`   | 1 node, 128 cores, `--exclusive` |
| Iris `cpu` | `batch`   | 1 task, 7 cores (4 GB/core)      |
| Iris `gpu` | `gpu`     | 1 task, 7 cores, 1 GPU           |

Logs go to `logs/<job-name>-<jobid>.out`. Check progress with `squeue --me`, and cancel a job with `scancel <jobid>`.

The batch script [`hpc/job.sh`](hpc/job.sh) activates the env and loads CUDA when the job has a GPU. It then prints the node, CPU, GPU and Python details and runs `python main.py`. It sets `OMP_NUM_THREADS=1` (LJ clusters are too small to benefit from threaded BLAS) and exports `NCPUS` (the cores allocated to the job), which `main.py` can use to size a process pool. For example, use one process per energy. To use threaded BLAS instead, set `OMP_NUM_THREADS` in your shell before you submit.

### Interactive sessions

```bash
# CPU (both clusters; the interactive partition allows up to 2 h)
salloc -p interactive -q normal -N 1 -n 1 -c 4 -t 01:00:00

# GPU (Iris)
salloc -p gpu -q normal -N 1 -n 1 -c 7 -G 1 -t 01:00:00
module load system/CUDA/12.6.0

source ~/miniforge3/etc/profile.d/conda.sh && conda activate lj-melt
python main.py
```
