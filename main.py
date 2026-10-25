"""CLI entry point: LJ cluster melting, serial on a laptop or MPI on HPC.

Examples
--------
Single run with trajectory + viewer:

    python main.py --mode single --energies -36 --steps 50000

Energy scan (one process per (energy, replica); scales with mpiexec):

    python main.py --mode scan --start-energy -44.0 --end-energy -24.0 --de 0.5 \
        --replicas 3 --steps 100000

    # equivalent: explicit list
    python main.py --mode scan --energies -44.3 -40 -36 -32 -28 -24 \
        --replicas 3 --steps 100000

    mpiexec -n 8 python main.py --mode scan --energies ... --replicas 3

Predefined cluster configurations (sizes 2-7, 13, 38, 55, 147, 309, 561):

    python main.py --list-clusters                  # table of presets
    python main.py --mode scan --num-particles 55   # geometry auto-selected
    python main.py --mode single --num-particles 7 --kind pentagonal_bipyramid
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time

import numpy as np

from lj import cluster
from lj.config import SimulationConfig
from lj.mpi_scan import run_scan, run_single

# Scan window for the melting study, in the potential-shifted energy frame
# (the Mackay minimum is E* ~ -44.3268 unshifted; the 2.5-sigma potential
# shift moves it to ~ -43.05, so -43.0 is the coldest usable start).
# Default resolution is 0.01 sigma_E across the full solid->liquid->vapour range.
E_MIN, E_MAX, E_STEP = -43.0, -24.0, 0.01
DEFAULT_ENERGIES = np.round(np.arange(E_MIN, E_MAX + E_STEP / 2, E_STEP), 2).tolist()
# coarser default grid for other cluster sizes (0.01 over ~1.5N span is overkill)
OTHER_N_STEP = 0.5


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=["single", "scan"],
        default="single",
        help="single: one trajectory + HTML viewer; scan: (energy x replica) table",
    )
    parser.add_argument(
        "--energies",
        type=float,
        nargs="+",
        default=None,
        help="explicit total-energy windows for the scan (overrides start/end/de)",
    )
    parser.add_argument(
        "--start-energy", type=float, default=None,
        help="scan start energy E0 (e.g. -43.0, the Mackay minimum)",
    )
    parser.add_argument(
        "--end-energy", type=float, default=None,
        help="scan end energy E1 (inclusive, e.g. -24.0)",
    )
    parser.add_argument(
        "--de", "--dE", dest="de", type=float,
        help="energy step size dE for the start->end grid",
    )
    parser.add_argument("--replicas", type=int, default=3, help="runs per energy")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument(
        "--num-particles",
        type=int,
        default=13,
        help="cluster size N; a preset geometry is chosen automatically for "
        "N = 2..7, 13, 38, 55, 147, 309, 561 (see --list-clusters), other N "
        "use a relaxed BCC cluster",
    )
    parser.add_argument(
        "--kind", default="auto",
        help="geometry family: auto (default), icosahedral, "
        "truncated_octahedral, bcc, or a preset name from --list-clusters",
    )
    parser.add_argument(
        "--relax", action=argparse.BooleanOptionalAction, default=True,
        help="FIRE-relax the starting geometry to its nearest local "
        "LJ minimum (default: --relax); --no-relax keeps the idealised "
        "preset coordinates",
    )
    parser.add_argument(
        "--list-clusters", action="store_true",
        help="print the predefined cluster configurations and exit",
    )
    parser.add_argument("--sample-every", type=int, default=200)
    parser.add_argument("--cutoff", type=float, default=2.5)
    parser.add_argument(
        "--sigma", type=float, default=1.0,
        help="LJ length parameter (1 = reduced units; e.g. argon 3.405 A)",
    )
    parser.add_argument(
        "--epsilon", type=float, default=1.0,
        help="LJ well depth (1 = reduced units)",
    )
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
        sigma=args.sigma,
        epsilon=args.epsilon,
        seed=args.seed,
        thermostat=None if args.thermostat == "none" else args.thermostat,
        target_temperature=args.target_temperature,
        tau_t=args.tau_t,
        cluster_kind=args.kind,
        relax_geometry=args.relax,
    )


def expand_energies(args: argparse.Namespace) -> list[float]:
    """Resolve the CLI energy spec into the grid passed to run_scan.

    Priority: explicit --energies list > --start/--end/--de grid >
    preset window (N=13 keeps the flagship full-resolution sweep;
    other sizes get a suggested window at 0.5-resolution).
    """
    if args.energies:
        return args.energies
    if args.start_energy is not None and args.end_energy is not None:
        de = args.de if args.de is not None else E_STEP
        if de <= 0:
            raise SystemExit("--de must be positive")
        up = args.end_energy >= args.start_energy
        # include the endpoint; float-safe rounding (powers of 0.01 etc.)
        eps = 1e-9 * de
        span = np.arange(
            args.start_energy,
            args.end_energy + (de / 2 if up else -de / 2) + (eps if up else -eps),
            de,
        )
        return np.round(span, 8).tolist()
    if args.de is not None:
        raise SystemExit("--de needs --start-energy and --end-energy")
    if args.num_particles == 13 and args.kind in ("auto", "icosahedral"):
        return DEFAULT_ENERGIES
    # suggested window for a preset/other-size cluster: just above the
    # ground state, up to the vicinity of where melting/evaporation starts
    e_low, e_high = cluster.suggested_energy_window(
        args.num_particles, cutoff=args.cutoff, sigma=args.sigma, epsilon=args.epsilon
    )
    return np.round(np.arange(e_low, e_high + OTHER_N_STEP / 2, OTHER_N_STEP), 8).tolist()


def print_clusters() -> None:
    presets = cluster.list_clusters()
    print(f"{len(presets)} predefined cluster configurations "
          f"(all relaxed to their nearest LJ local minimum by default):\n")
    print(f"  {'key':24s} {'N':>4s}  description")
    for key, preset in presets:
        print(f"  {key:24s} {preset.num_particles:4d}  {preset.description}")
    print(
        "\nUse with --num-particles (geometry chosen automatically) or\n"
        "--kind <key>. Reference energies are for the full unshifted LJ\n"
        "potential in reduced units; runs use the potential-shifted\n"
        "cutoff (--cutoff, default 2.5 sigma), so reported E* is slightly\n"
        "higher than the reference."
    )


def _mpi_rank() -> int:
    """Rank of this process under MPI (0 when serial or mpi4py missing)."""
    try:
        from mpi4py import MPI
    except ImportError:
        return 0
    return MPI.COMM_WORLD.rank


def n_ranks() -> int:
    try:
        from mpi4py import MPI
    except ImportError:
        return 1
    return MPI.COMM_WORLD.size


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list_clusters:
        print_clusters()
        return 0
    args.energies = expand_energies(args)
    config = build_config(args)
    rank = _mpi_rank()

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
        print("wrote outputs/trajectory.json, outputs/cluster.html, and outputs/potential.png")
        return 0

    # scan mode: master distributes (energy, replica) items over MPI ranks

    if rank == 0:              # banner only once: identical per-rank banners
        print(                 # make the scan log look like duplicate work
            f"scan [rank 0 of {n_ranks()}]: {len(args.energies)} energies "
            f"x {args.replicas} replicas = {len(args.energies) * args.replicas} "
            f"runs, {args.steps} steps each ({cluster.describe_size(args.num_particles, args.kind)})"
        )
    results = run_scan(config, args.energies, args.replicas, args.outdir)
    if results is None:
        return 0  # MPI worker
    if rank != 0:
        return 0
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
