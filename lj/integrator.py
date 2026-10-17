"""Velocity Verlet integrator (kick-drift-kick) and small thermostats."""

from __future__ import annotations

import numpy as np

from .physics import ForceEvaluator


def verlet_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    forces: np.ndarray,
    force_eval: ForceEvaluator,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Advance one kick-drift-kick Velocity Verlet step. Mass = 1.

    Returns (positions, velocities, forces, potential_energy) at t + dt.
    The returned velocities are full-step velocities, so kinetic
    temperature and energy are directly available after each step.
    """
    velocities = velocities + 0.5 * dt * forces  # half-kick
    positions = positions + dt * velocities  # drift
    forces, pe = force_eval.compute(positions)  # new forces
    velocities = velocities + 0.5 * dt * forces  # half-kick
    return positions, velocities, forces, pe


def berendsen_factor(
    measured_temperature: float, target_temperature: float, dt: float, tau_t: float
) -> float:
    """Velocity scaling factor of the Berendsen thermostat.

    Gentle exponential relaxation of T towards the target; not a rigorous
    canonical sampler, but adequate for producing NVT trajectories and for
    controlled heating/cooling ramps.
    """
    if measured_temperature <= 0.0:
        return 1.0
    return float(np.sqrt(1.0 + (dt / tau_t) * (target_temperature / measured_temperature - 1.0)))
