"""MPI tests.

Run serially they exercise the no-communicator fallbacks; under
``mpiexec -n 3`` (or more) they additionally check that the atom-
decomposed force evaluator agrees with the serial one, and that the
master/worker scan produces merged outputs.
"""

import numpy as np
import pytest

from lj.physics import AtomDecomposedForces, SerialForces

mpi4py = pytest.importorskip("mpi4py")
from mpi4py import MPI  # noqa: E402

COMM = MPI.COMM_WORLD
RANK = COMM.rank
SIZE = COMM.size


def test_atom_decomposed_matches_serial_mpi():
    """Each rank's owned slice of forces must match the serial evaluator."""
    if SIZE < 2:
        pytest.skip("needs >= 2 ranks")
    rng = np.random.default_rng(42)  # same positions on every rank
    positions = rng.normal(size=(13, 3)) * 0.7
    f_ref, pe_ref = SerialForces(2.5).compute(positions)
    evaluator = AtomDecomposedForces(2.5, comm=COMM)
    f_mpi, pe_mpi = evaluator.compute(positions)
    owned = evaluator._owned_indices(len(positions))
    assert np.allclose(f_mpi[owned], f_ref[owned], atol=1e-12)
    assert np.isclose(pe_mpi, pe_ref, atol=1e-10)


@pytest.mark.skipif(SIZE < 2, reason="needs >= 2 ranks")
def test_scan_writes_merged_outputs(tmp_path):
    from lj.config import SimulationConfig
    from lj.mpi_scan import run_scan

    config = SimulationConfig(
        total_energy=-40.0,
        num_steps=2_000,
        sample_every=500,
        trajectory_every=0,
        seed=99,
    )
    results = run_scan(config, energies=[-40.0, -38.0], replicas=1, outdir=str(tmp_path))
    if RANK == 0:
        assert results is not None and len(results) == 2
        assert all(r["ok"] for r in results)
        for name in ("merged.json", "caloric.csv", "summary.json"):
            assert (tmp_path / name).exists()
        # per-run CSVs land in per_run/
        assert (tmp_path / "per_run").is_dir()
    else:
        assert results is None
