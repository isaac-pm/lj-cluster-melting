"""Settable Lennard-Jones sigma/epsilon parameters.

The reduced units sigma = epsilon = 1 must remain the untouched default;
setting sigma != 1, epsilon != 1 rescales lengths and energies exactly:
U(x) = eps * U_red(x/sigma),  F(x) = (eps/sigma) * F_red(x/sigma),
with the cutoff interpreted in units of sigma.
"""

import numpy as np
import pytest

from lj.cluster import build_cluster
from lj.physics import (
    AtomDecomposedForces,
    SerialForces,
    lj_energy_and_forces,
    pair_distances,
)


def test_energy_and_force_scaling():
    rng = np.random.default_rng(5)
    pos = rng.normal(scale=1.2, size=(13, 3))
    f_red, pe_red = lj_energy_and_forces(pos, cutoff=2.5)
    f, pe = lj_energy_and_forces(
        2 * pos, cutoff=2.5, sigma=2.0, epsilon=3.0
    )
    assert np.isclose(pe / pe_red, 3.0)
    # F_vec scales as eps / sigma = 3 / 2
    assert np.allclose(f, 1.5 * f_red, rtol=1e-10)


def test_forces_object_matches_function():
    rng = np.random.default_rng(1)
    pos = rng.normal(scale=1.0, size=(13, 3))
    f_ref, pe_ref = lj_energy_and_forces(pos, 3.4, sigma=1.36, epsilon=1.0)
    f_s, pe_s = SerialForces(3.4, sigma=1.36, epsilon=1.0).compute(pos)
    f_d, pe_d = AtomDecomposedForces(3.4, sigma=1.36, epsilon=1.0).compute(pos)
    for f, pe in ((f_s, pe_s), (f_d, pe_d)):
        assert np.allclose(f, f_ref, atol=1e-12)
        assert np.isclose(pe, pe_ref, atol=1e-12)


def test_cluster_geometry_scales_with_sigma():
    pa = build_cluster(13, kind="icosahedral")
    pb = build_cluster(13, kind="icosahedral", sigma=2.0)
    da, _, _ = pair_distances(pa)
    db, _, _ = pair_distances(pb)
    assert np.allclose(np.sort(db), 2.0 * np.sort(da))

    _, e_a = lj_energy_and_forces(pa, cutoff=10.0)
    _, e_b = lj_energy_and_forces(pb, cutoff=10.0, sigma=2.0, epsilon=3.0)
    assert np.isclose(e_b / e_a, 3.0, rtol=1e-9)


def test_nve_run_equivalence():
    from lj.config import SimulationConfig
    from lj.simulation import run_trajectory

    cfg_a = SimulationConfig(total_energy=-43.0, num_steps=40_000,
                             sample_every=50, seed=7)
    cfg_b = SimulationConfig(total_energy=-129.0, num_steps=40_000,
                             sample_every=50, seed=7, sigma=2.0, epsilon=3.0)
    ra, rb = run_trajectory(cfg_a), run_trajectory(cfg_b)
    assert np.isclose(
        rb.averages["temperature_mean"] / ra.averages["temperature_mean"], 3.0,
        rtol=0.02,
    )
    assert np.isclose(rb.averages["lindemann_mean"], ra.averages["lindemann_mean"],
                      rtol=0.05)
    assert np.isclose(rb.averages["rg_mean"], 2.0 * ra.averages["rg_mean"], rtol=0.05)
    assert rb.averages["drift_rel"] < 1e-3


def test_config_validation():
    from lj.config import SimulationConfig

    with pytest.raises(ValueError):
        SimulationConfig(sigma=0.0)
    with pytest.raises(ValueError):
        SimulationConfig(epsilon=-1.0)
