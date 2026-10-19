"""Trajectory driver: initial conditions, sampling, equilibration averages."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import cluster, physics
from .config import SimulationConfig
from .integrator import berendsen_factor, verlet_step
from .physics import ForceEvaluator, SerialForces

OBSERVABLES = ("time", "ke", "pe", "e_total", "temperature", "rg")


class LindemannAccumulator:
    """Running Lindemann index over the bonds of the initial configuration.

    A "bond" is any pair closer than ``bond_cutoff`` in the initial frame
    (1.5 sigma for the Mackay icosahedron captures the 30 edges plus the
    12 centre bonds). The index is the bond-averaged relative root-mean-
    square fluctuation of those distances:

        delta_L = mean_bonds [ sqrt(<d^2> - <d>^2) / <d> ]

    Values above ~0.1 signal liquid-like behaviour; divergence indicates
    bond breaking / evaporation.
    """

    def __init__(self, positions: np.ndarray, bond_cutoff: float = 1.5):
        distances, i, j = physics.pair_distances(positions)
        keep = distances < bond_cutoff
        self.i = i[keep]
        self.j = j[keep]
        self.count = 0
        self.sum = np.zeros(int(keep.sum()))
        self.sumsq = np.zeros(int(keep.sum()))

    def add(self, positions: np.ndarray) -> None:
        d = np.linalg.norm(positions[self.i] - positions[self.j], axis=1)
        self.sum += d
        self.sumsq += d * d
        self.count += 1

    def index(self) -> float:
        if self.count == 0:
            return float("nan")
        mean = self.sum / self.count
        var = np.maximum(self.sumsq / self.count - mean**2, 0.0)
        return float(np.mean(np.sqrt(var) / mean))


@dataclass
class RunResult:
    """Output of one trajectory.

    samples: dict of equal-length NumPy arrays, one row per sample.
    averages: tail-half means (after equilibration) plus energy drift.
    frames: stored positions snapshots (empty unless trajectory_every > 0).
    """

    config: SimulationConfig
    samples: dict[str, np.ndarray]
    averages: dict[str, float]
    frames: list[np.ndarray] = field(default_factory=list)
    frame_times: list[float] = field(default_factory=list)
    frame_temperatures: list[float] = field(default_factory=list)
    frame_energies: list[float] = field(default_factory=list)
    wall_time: float = 0.0

    def downsampled_samples(self, max_points: int = 1000) -> dict[str, np.ndarray]:
        """Thin the time series for cheap transport over MPI."""
        n = len(self.samples["time"])
        stride = max(1, n // max_points)
        return {k: v[::stride] for k, v in self.samples.items()}


def run_trajectory(
    config: SimulationConfig,
    force_eval: ForceEvaluator | None = None,
    progress: "callable[[int, int], None] | None" = None,
) -> RunResult:
    """Run one (NVE or NVT) trajectory from the Mackay minimum.

    Initial velocities are Maxwell-Boltzmann at a temperature estimated
    from the target energy, then rescaled so E_total = E* exactly and the
    centre-of-mass velocity is removed. Total energy should be below the
    potential energy of the initial configuration.
    """
    if force_eval is None:
        force_eval = SerialForces(config.cutoff)

    rng = np.random.default_rng(config.seed)
    kind = "icosahedral" if config.num_particles == 13 else "bcc"
    positions = cluster.build_cluster(config.num_particles, kind=kind)
    _, pe0 = force_eval.compute(positions)

    ndof = 3 * config.num_particles - 3
    t_guess = 2.0 * max(config.total_energy - pe0, 0.05) / ndof
    velocities = cluster.maxwell_boltzmann_velocities(config.num_particles, t_guess, rng)
    # Project out the COM velocity *before* rescaling, otherwise COM removal
    # would drain kinetic energy and E_total would miss the target.
    cluster.remove_com_velocity(velocities)
    velocities = cluster.rescale_to_total_energy(velocities, pe0, config.total_energy)

    forces, pe = force_eval.compute(positions)
    lindemann = LindemannAccumulator(positions)

    samples: dict[str, list[float]] = {k: [] for k in OBSERVABLES}
    frames: list[np.ndarray] = []
    frame_times: list[float] = []
    frame_temperatures: list[float] = []
    frame_energies: list[float] = []
    last_temperature = 0.0

    equil_start = int(config.num_steps * config.equil_fraction)
    equil_samples: dict[str, list[float]] = {k: [] for k in OBSERVABLES}
    start_time = time.perf_counter()

    for step in range(1, config.num_steps + 1):
        positions, velocities, forces, pe = verlet_step(
            positions, velocities, forces, force_eval, config.dt
        )

        if config.com_remove_every and step % config.com_remove_every == 0:
            cluster.remove_com_velocity(velocities)

        if config.thermostat == "berendsen":
            ke_now = physics.kinetic_energy(velocities)
            t_now = physics.temperature(ke_now, config.num_particles)
            velocities *= berendsen_factor(
                t_now, config.target_temperature, config.dt, config.tau_t
            )

        if step % config.sample_every == 0:
            ke = physics.kinetic_energy(velocities)
            last_temperature = physics.temperature(ke, config.num_particles)
            row = {
                "time": step * config.dt,
                "ke": ke,
                "pe": pe,
                "e_total": ke + pe,
                "temperature": last_temperature,
                "rg": physics.gyration_radius(positions),
            }
            for k, v in row.items():
                samples[k].append(v)
                if step > equil_start:
                    equil_samples[k].append(v)

        if step > equil_start:
            lindemann.add(positions)

        if config.trajectory_every and step % config.trajectory_every == 0:
            frames.append(positions.copy())
            frame_times.append(step * config.dt)
            frame_temperatures.append(
                physics.temperature(physics.kinetic_energy(velocities), config.num_particles)
            )
            frame_energies.append(
                physics.kinetic_energy(velocities) + pe
            )

        if progress and step % max(config.num_steps // 20, 1) == 0:
            progress(step, config.num_steps)

    wall_time = time.perf_counter() - start_time

    sample_arrays = {k: np.asarray(v, dtype=float) for k, v in samples.items()}

    averages = {
        f"{k}_mean": float(np.mean(v)) for k, v in equil_samples.items()
    }
    averages["lindemann_mean"] = lindemann.index()
    e_total = sample_arrays["e_total"]
    averages["drift_rel"] = float(
        np.max(np.abs(e_total - e_total[0])) / max(abs(e_total[0]), 1e-12)
    )
    averages["wall_time"] = wall_time

    return RunResult(
        config=config,
        samples=sample_arrays,
        averages=averages,
        frames=frames,
        frame_times=frame_times,
        frame_temperatures=frame_temperatures,
        frame_energies=frame_energies,
        wall_time=wall_time,
    )
