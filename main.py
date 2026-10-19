"""CLI entry point: LJ cluster melting, serial on a laptop or MPI on HPC.

Examples
--------
Single run with trajectory + viewer:

    python main.py --mode single --energies -36 --steps 50000

Energy scan (one process per (energy, replica); scales with mpiexec):

    python main.py --mode scan --energies -44.3 -40 -36 -32 -28 -24 \
        --replicas 3 --steps 100000

    mpiexec -n 8 python main.py --mode scan --energies ... --replicas 3
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time

import numpy as np

from lj.config import SimulationConfig
from lj.mpi_scan import run_scan, run_single

# Scan window for the melting study, in the potential-shifted energy frame
# (the Mackay minimum is E* ~ -44.3268 unshifted; the 2.5-sigma potential
# shift moves it to ~ -43.05, so -43.0 is the coldest usable start).
# Default resolution is 0.01 sigma_E across the full solid->liquid->vapour range.
E_MIN, E_MAX, E_STEP = -43.0, -24.0, 0.01
DEFAULT_ENERGIES = np.round(np.arange(E_MIN, E_MAX + E_STEP / 2, E_STEP), 2).tolist()


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=["single", "scan"],
        default="scan",
        help="single: one trajectory + HTML viewer; scan: (energy x replica) table",
    )
    parser.add_argument(
        "--energies",
        type=float,
        nargs="+",
        default=DEFAULT_ENERGIES,
        help="total-energy windows for the scan (default: a preset melting sweep)",
    )
    parser.add_argument("--replicas", type=int, default=3, help="runs per energy")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument(
        "--num-particles",
        type=int,
        default=13,
        help="cluster size N (13 -> Mackay icosahedron; other N use a BCC cluster)",
    )
    parser.add_argument("--sample-every", type=int, default=200)
    parser.add_argument("--cutoff", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument(
        "--thermostat", choices=["none", "berendsen"], default="none",
        help="none = microcanonical NVE (default); berendsen = NVT",
    )
    parser.add_argument("--target-temperature", type=float, default=None)
    parser.add_argument("--tau-t", type=float, default=0.1, help="Berendsen coupling time")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> SimulationConfig:
    return SimulationConfig(
        num_particles=args.num_particles,
        total_energy=args.energies[0],
        dt=args.dt,
        num_steps=args.steps,
        sample_every=args.sample_every,
        cutoff=args.cutoff,
        seed=args.seed,
        thermostat=None if args.thermostat == "none" else args.thermostat,
        target_temperature=args.target_temperature,
        tau_t=args.tau_t,
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    config = build_config(args)

    started = time.perf_counter()
    if args.mode == "single":
        # Rank 0 runs one trajectory and dumps trajectory.json + cluster.html.
        energies = args.energies[:1]
        if len(energies) != 1:
            print("single mode uses only the first energy", file=sys.stderr)
        config = dataclasses.replace(
            config,
            total_energy=energies[0],
            trajectory_every=max(1, args.steps // 2000),
        )
        result = run_single(config, args.outdir)
        if result is None:
            return 0  # MPI worker: nothing to do
        print(f"trajectory done in {result.averages['wall_time']:.1f} s")
        print(f"  mean T*   = {result.averages['temperature_mean']:.4f}")
        print(f"  Lindemann = {result.averages['lindemann_mean']:.4f}")
        print(f"  |dE/E|    = {result.averages['drift_rel']:.2e}")
        print("wrote outputs/trajectory.json and outputs/cluster.html")
        return 0

    # scan mode: master distributes (energy, replica) items over MPI ranks
    kind = "icosahedral" if args.num_particles == 13 else "bcc"
    print(
        f"scan: {len(args.energies)} energies x {args.replicas} replicas "
        f"= {len(args.energies) * args.replicas} runs, "
        f"{args.steps} steps each (N={args.num_particles}, {kind} cluster)"
    )
    results = run_scan(config, args.energies, args.replicas, args.outdir)
    if results is None:
        return 0  # MPI worker
    ok = [r for r in results if r.get("ok")]
    failed = len(results) - len(ok)
    print(
        f"\nscan finished in {time.perf_counter() - started:.1f} s "
        f"({len(ok)} ok, {failed} failed)"
    )
    print("wrote outputs/{merged.json, caloric.csv, caloric.png, summary.json, per_run/}")
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
