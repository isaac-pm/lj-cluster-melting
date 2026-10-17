"""Force/potential unit tests (serial; no MPI needed)."""

import numpy as np

from lj.cluster import LJ_MIN_DISTANCE, mackay_icosahedron
from lj.physics import (
    AtomDecomposedForces,
    SerialForces,
    lj_energy_and_forces,
    per_atom_energy,
    temperature,
)


def test_force_vanishes_at_pair_minimum():
    """Two atoms at r = 2**(1/6) feel no force."""
    positions = np.array([[0.0, 0.0, 0.0], [LJ_MIN_DISTANCE, 0.0, 0.0]])
    forces, pe = lj_energy_and_forces(positions, cutoff=3.0)
    assert np.allclose(forces, 0.0, atol=1e-12)
    # shifted pair minimum: U(r_min) - U(rc)
    expected = -1.0 - 4.0 * ((1.0 / 3.0) ** 12 - (1.0 / 3.0) ** 6)
    assert np.isclose(pe, expected)


def test_force_repulsive_inside_minimum():
    positions = np.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]])
    forces, _ = lj_energy_and_forces(positions, cutoff=3.0)
    # atom 0 pushed in -x away from atom 1
    assert forces[0, 0] < 0.0
    assert forces[1, 0] > 0.0
    assert np.allclose(forces[0], -forces[1])  # Newton's third law


def test_mackay_icosahedron_energy():
    """The scaled icosahedron is the LJ-13 global minimum, E* ~ -44.3268."""
    positions = mackay_icosahedron()
    _, pe = lj_energy_and_forces(positions, cutoff=10.0)
    assert pe == np.float64(np.float64(pe))  # sanity, not nan
    assert abs(pe - (-44.3268)) < 0.01


def test_cutoff_excludes_distant_pairs():
    positions = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    forces, pe = lj_energy_and_forces(positions, cutoff=2.5)
    assert pe == 0.0
    assert np.allclose(forces, 0.0)


def test_per_atom_energy_sums_to_total():
    rng = np.random.default_rng(42)
    positions = rng.normal(size=(8, 3))
    _, pe_total = lj_energy_and_forces(positions)
    pe_atoms = per_atom_energy(positions)
    assert np.isclose(pe_atoms.sum(), pe_total)


def test_atom_decomposed_matches_serial():
    """Atom decomposition without a communicator reduces to the serial path."""
    rng = np.random.default_rng(7)
    positions = rng.normal(size=(11, 3)) * 0.8
    f_serial, pe_serial = SerialForces(2.5).compute(positions)
    f_decomp, pe_decomp = AtomDecomposedForces(2.5, comm=None).compute(positions)
    assert np.allclose(f_decomp, f_serial)
    assert np.isclose(pe_decomp, pe_serial)


def test_temperature_equipartition_factor():
    # 3N-3 dof: T = 2*KE/(3N-3) = 2*54/36 = 3
    assert np.isclose(temperature(ke=54.0, num_particles=13), 3.0)
