"""Simulation configuration.

All quantities are in Lennard-Jones reduced units:
sigma = epsilon = kB = m = 1, so energies are in epsilon, lengths in sigma,
times in tau = sigma * sqrt(m / epsilon).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters of a single molecular-dynamics run.

    Attributes:
        num_particles: cluster size (13 -> Mackay icosahedron).
        total_energy: target total energy E* used to rescale the initial
            velocities (microcanonical NVE runs).
        dt: Verlet timestep (in tau). 0.005 is a good default for LJ.
        num_steps: number of integration steps.
        sample_every: record observables every this many steps.
        trajectory_every: store full positions every this many steps.
            Set to 0 to skip trajectory output entirely (scan mode).
        cutoff: LJ cutoff in sigma, potential-shifted.
        seed: RNG seed; None picks one from OS entropy.
        thermostat: None for NVE, or "berendsen" for NVT.
        target_temperature: NVT target T* (required for berendsen).
        tau_t: Berendsen coupling time (in tau).
        com_remove_every: remove centre-of-mass velocity drift every this
            many steps. 0 disables (e.g. already removed by construction).
        equil_fraction: fraction of the trajectory discarded as
            equilibration when computing averages.
    """

    num_particles: int = 13
    total_energy: float = -38.0
    dt: float = 0.005
    num_steps: int = 100_000
    sample_every: int = 200
    trajectory_every: int = 0
    cutoff: float = 2.5
    seed: int | None = None
    thermostat: str | None = None
    target_temperature: float | None = None
    tau_t: float = 0.1
    com_remove_every: int = 50
    equil_fraction: float = 0.5

    def __post_init__(self) -> None:
        if self.num_particles < 2:
            raise ValueError("num_particles must be >= 2")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if self.cutoff <= 1.0:
            raise ValueError("cutoff must be larger than 1 sigma")
        if self.sample_every < 1 or self.trajectory_every < 0:
            raise ValueError("sample_every must be >= 1 and trajectory_every >= 0")
        if self.thermostat not in (None, "berendsen"):
            raise ValueError(f"unknown thermostat {self.thermostat!r}")
        if self.thermostat == "berendsen" and not self.target_temperature:
            raise ValueError("berendsen thermostat requires target_temperature")
        if not 0.0 < self.equil_fraction < 1.0:
            raise ValueError("equil_fraction must be in (0, 1)")
