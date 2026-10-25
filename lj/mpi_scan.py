"""MPI replica driver: an embarrassingly-parallel scan over (energy, seed).

Rank 0 is the master: it hands work items out dynamically (first-come,
first-served, so slow energies do not strand workers) and merges results.
Workers run independent trajectories and return downsampled summaries.

The same entry point runs serially when mpi4py is absent or size == 1,
so a laptop and an HPC cluster share one code path. For large clusters,
:class:`lj.physics.AtomDecomposedForces` adds data-level parallelism
inside each worker.
"""

from __future__ import annotations

import csv
import json
import os
from collections import deque
from dataclasses import asdict, dataclass, replace

from .analysis import (
    caloric_table,
    melting_summary,
    write_caloric_plot,
    write_lj_potential_plot,
)
from .config import SimulationConfig
from .physics import SerialForces
from .simulation import run_trajectory

TAG_WORK = 11
TAG_RESULT = 12
TAG_STOP = 13


@dataclass(frozen=True)
class WorkItem:
    """One independent trajectory: an energy window and a replica index."""

    total_energy: float
    seed: int
    replica: int


def build_work_items(
    energies: list[float] | tuple[float, ...],
    replicas: int,
    base_seed: int,
) -> list[WorkItem]:
    """Cross product of energy windows and replica indices."""
    items = []
    for i, energy in enumerate(energies):
        for r in range(replicas):
            items.append(
                WorkItem(
                    total_energy=float(energy),
                    seed=base_seed + 1000 * i + r,
                    replica=r,
                )
            )
    return items


def _mpi_launched() -> bool:
    """Running under mpiexec/Slurm even when mpi4py failed to import."""
    return any(
        key in os.environ
        for key in (
            "PMI_RANK", "PMI_SIZE", "PMIX_RANK", "PMIX_NAMESPACE",
            "OMPI_COMM_WORLD_RANK", "MV2_RANK", "ALPS_APP_PES",
        )
    )


def _get_comm():
    """Return MPI.COMM_WORLD, or None when mpi4py is unavailable or size 1.

    Refuses the dangerous silent case: launched by mpiexec/Slurm *without*
    mpi4py, every rank would then fall back to running the full scan
    itself — all ranks doing identical work, N x slower, racing each
    other's output files.
    """
    try:
        from mpi4py import MPI
    except ImportError:
        if _mpi_launched():
            raise RuntimeError(
                "launched under mpiexec/Slurm but mpi4py is not importable; "
                "every rank would repeat the whole scan (no speed-up). "
                "Install mpi4py in the job environment "
                "(conda install -c conda-forge mpi4py, see setup_env.sh) "
                "or run serially without mpiexec."
            )
        return None
    comm = MPI.COMM_WORLD
    return comm if comm.size > 1 else None


def run_scan(
    config: SimulationConfig,
    energies: list[float] | tuple[float, ...],
    replicas: int,
    outdir: str,
    comm=None,
) -> list[dict] | None:
    """Run the full (energy x replica) scan.

    Returns the merged results on the master (rank 0), None on workers.
    Master writes per-run CSVs, merged.json, caloric.csv and caloric.png
    into ``outdir``.
    """
    if comm is None:
        comm = _get_comm()
    rank = comm.rank if comm is not None else 0
    size = comm.size if comm is not None else 1

    items = build_work_items(energies, replicas, config.seed or 0)

    if rank != 0:
        _worker(items, config, comm)
        return None

    if comm is None:
        return _run_serial(items, config, outdir)
    return _master(items, config, outdir, comm, size)


def _run_serial(items: list[WorkItem], config: SimulationConfig, outdir: str) -> list[dict]:
    """Serial fallback: rank 0 runs every item itself, no MPI required."""
    os.makedirs(outdir, exist_ok=True)
    results: list[dict] = []
    for item_index, item in enumerate(items):
        run_config = replace(config, total_energy=item.total_energy, seed=item.seed)
        try:
            result = run_trajectory(run_config, force_eval=SerialForces(run_config.cutoff, run_config.sigma, run_config.epsilon))
            payload = _result_payload(result, item, item_index)
            results.append(payload)
            _write_per_run_csv(os.path.join(outdir, "per_run"), item, {"result": payload})
        except Exception as exc:
            print(f"  item E={item.total_energy:.4f} r{item.replica} failed: {exc}")
            results.append(
                {
                    "item_index": item_index,
                    "total_energy": item.total_energy,
                    "replica": item.replica,
                    "seed": item.seed,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    _write_merged(outdir, config, results)
    return results


def _master(
    items: list[WorkItem],
    config: SimulationConfig,
    outdir: str,
    comm,
    size: int,
) -> list[dict]:
    os.makedirs(outdir, exist_ok=True)
    results: list[dict] = []
    pending = deque(range(len(items)))
    active: dict[int, int] = {}  # worker rank -> item index

    # Seed the workers: one item each, keep the rest in the queue.
    for worker in range(1, size):
        if pending:
            idx = pending.popleft()
            comm.send(("run", idx, items[idx]), dest=worker, tag=TAG_WORK)
            active[worker] = idx

    while pending or active:
        from mpi4py import MPI

        msg = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_RESULT)
        worker = msg["worker"]
        idx = msg["item_index"]
        item = items[idx]
        if msg.get("error"):
            print(
                f"  [rank 0] item {idx} (E={item.total_energy:.4f}, "
                f"replica {item.replica}) failed on rank {worker}: {msg['error']}"
            )
            results.append(
                {
                    "item_index": idx,
                    "total_energy": item.total_energy,
                    "replica": item.replica,
                    "seed": item.seed,
                    "ok": False,
                    "error": msg["error"],
                }
            )
        else:
            results.append(msg["result"])
            _write_per_run_csv(os.path.join(outdir, "per_run"), item, msg)
        if pending:
            next_idx = pending.popleft()
            comm.send(("run", next_idx, items[next_idx]), dest=worker, tag=TAG_WORK)
            active[worker] = next_idx
        else:
            comm.send(("stop", None), dest=worker, tag=TAG_STOP)
            del active[worker]

    results.sort(key=lambda r: r["item_index"])
    _write_merged(outdir, config, results)
    return results


def _worker(items: list[WorkItem], config: SimulationConfig, comm) -> None:
    from mpi4py import MPI

    while True:
        received = comm.recv(source=0, tag=MPI.ANY_TAG)
        if received[0] == "stop":
            return
        _, idx, work = received
        # visible proof of specialisation in the job log: each rank runs a
        # *different* (energy, replica) item, handed out by rank 0
        print(
            f"  [rank {comm.rank}] E={work.total_energy:.3f} "
            f"replica {work.replica}",
            flush=True,
        )
        run_config = replace(config, total_energy=work.total_energy, seed=work.seed)
        try:
            result = run_trajectory(run_config, force_eval=SerialForces(run_config.cutoff, run_config.sigma, run_config.epsilon))
            comm.send(
                {
                    "worker": comm.rank,
                    "item_index": idx,
                    "result": _result_payload(result, work, idx),
                },
                dest=0,
                tag=TAG_RESULT,
            )
        except Exception as exc:  # keep the pool alive on a single failure
            comm.send(
                {
                    "worker": comm.rank,
                    "item_index": idx,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                dest=0,
                tag=TAG_RESULT,
            )


def _result_payload(result, work: WorkItem, item_index: int) -> dict:
    return {
        "item_index": item_index,
        "total_energy": work.total_energy,
        "replica": work.replica,
        "seed": work.seed,
        "ok": True,
        "averages": result.averages,
        "samples": {k: v.tolist() for k, v in result.downsampled_samples().items()},
    }


def _write_per_run_csv(outdir: str, item: WorkItem, msg: dict) -> None:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"e{item.total_energy:.4f}_r{item.replica}.csv")
    samples = msg["result"]["samples"]
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(samples.keys())
        writer.writerows(zip(*samples.values()))


def _write_merged(outdir: str, config: SimulationConfig, results: list[dict]) -> None:
    ok = [r for r in results if r.get("ok")]
    with open(os.path.join(outdir, "merged.json"), "w") as fh:
        json.dump(
            {
                "config": asdict(config),
                "n_items": len(results),
                "n_ok": len(ok),
                "results": ok,
            },
            fh,
            indent=2,
            default=float,
        )

    if ok:
        table = caloric_table(ok)
        _write_caloric_csv(outdir, table)
        summary = melting_summary(table)
        with open(os.path.join(outdir, "summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2, default=float)
        try:
            write_caloric_plot(table, os.path.join(outdir, "caloric.png"))
        except Exception as exc:  # plotting is optional on headless nodes
            print(f"  [rank 0] caloric.png skipped: {exc}")


def _write_caloric_csv(outdir: str, table) -> None:
    with open(os.path.join(outdir, "caloric.csv"), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(table.keys())
        writer.writerows(zip(*table.values()))


def run_single(
    config: SimulationConfig,
    outdir: str,
    dump_trajectory: bool = True,
) -> RunResult | None:
    """Run one trajectory (rank 0 only in MPI mode) and optionally dump it.

    Returns the RunResult on rank 0, None on workers. Workers simply exit:
    replica parallelism belongs to :func:`run_scan`.
    """
    comm = _get_comm()
    if comm is not None and comm.rank != 0:
        return None
    result = run_trajectory(config, force_eval=SerialForces(config.cutoff, config.sigma, config.epsilon))
    if dump_trajectory:
        from .visualize import write_trajectory_outputs

        write_trajectory_outputs(result, outdir)
        write_lj_potential_plot(
            os.path.join(outdir, "potential.png"),
            cutoff=config.cutoff,
            sigma=config.sigma,
            epsilon=config.epsilon,
        )
    return result
