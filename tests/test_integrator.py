"""Integrator and simulation-driver tests."""

import numpy as np

from lj.cluster import mackay_icosahedron
from lj.config import SimulationConfig
from lj.integrator import berendsen_factor, verlet_step
from lj.physics import SerialForces, kinetic_energy, temperature
from lj.simulation import LindemannAccumulator, run_trajectory


def test_verlet_conserves_energy_nve():
    """NVE energy drift over 20k steps should be tiny at dt=0.005."""
    config = SimulationConfig(
        total_energy=-40.0,
        dt=0.005,
        num_steps=20_000,
        sample_every=100,
        seed=123,
        com_remove_every=50,
    )
    result = run_trajectory(config)
    drift = result.averages["drift_rel"]
    assert drift < 1e-4, f"relative energy drift {drift:.2e} too large"
    # COM stays put (removed every 50 steps): kinetic T stays physical
    assert 0.0 < result.averages["temperature_mean"] < 2.0


def test_initial_total_energy_is_exact():
    config = SimulationConfig(
        total_energy=-36.0, num_steps=10, sample_every=5, seed=5
    )
    result = run_trajectory(config)
    e0 = result.samples["e_total"][0]
    assert abs(e0 - (-36.0)) < 0.5  # first sample already after a few steps


def test_lindemann_index_rises_when_cluster_breaks_apart():
    """Vaporised snapshot -> large delta (cold solid is ~0.01-0.05)."""
    rng = np.random.default_rng(3)
    hot = LindemannAccumulator(mackay_icosahedron())
    for f in rng.normal(scale=6.0, size=(50, 13, 3)):
        hot.add(f)
    assert hot.index() > 0.3

    cold = LindemannAccumulator(mackay_icosahedron())
    for _ in range(50):
        cold.add(mackay_icosahedron())
    assert cold.index() < 0.05


def test_berendsen_factor_bounds():
    # measured below target -> factor > 1 (heating)
    assert berendsen_factor(0.5, 1.0, 0.005, 0.1) > 1.0
    # measured at target -> factor 1
    assert np.isclose(berendsen_factor(1.0, 1.0, 0.005, 0.1), 1.0)


def test_verlet_step_advances_state():
    config = SimulationConfig(dt=0.005)
    force_eval = SerialForces(config.cutoff)
    positions = mackay_icosahedron()
    rng = np.random.default_rng(11)
    velocities = rng.normal(0.0, 1.0, size=(13, 3))  # non-zero start
    forces, pe = force_eval.compute(positions)
    p2, v2, f2, pe2 = verlet_step(positions, velocities, forces, force_eval, 0.005)
    assert p2.shape == positions.shape
    assert not np.allclose(p2, positions)  # moved
    assert kinetic_energy(v2) >= 0.0
    assert temperature(kinetic_energy(v2), 13) >= 0.0
    assert np.isfinite(pe2)
