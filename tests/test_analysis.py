"""Analysis and visualization tests (no MPI needed)."""

import numpy as np
import pytest

from lj.analysis import caloric_table, melting_summary

mpi4py = pytest.importorskip("mpi4py")
from mpi4py import MPI  # noqa: E402

RANK = MPI.COMM_WORLD.rank


def _fake_result(energy, temperature_mean, lindemann_mean):
    return {
        "total_energy": energy,
        "replica": 0,
        "seed": 1,
        "ok": True,
        "averages": {
            "temperature_mean": temperature_mean,
            "rg_mean": 2.0,
            "lindemann_mean": lindemann_mean,
            "drift_rel": 1e-5,
        },
    }


def test_caloric_table_aggregates_replicas():
    results = [
        _fake_result(-40.0, 0.30, 0.05),
        _fake_result(-40.0, 0.34, 0.06),
        _fake_result(-36.0, 0.60, 0.20),
    ]
    table = caloric_table(results)
    assert len(table["energy"]) == 2
    assert np.isclose(table["temperature_mean"][0], 0.32)
    assert table["max_drift_rel"][0] == 1e-5


def test_heat_capacity_positive_between_energies():
    results = [
        _fake_result(-44.0, 0.05, 0.02),
        _fake_result(-40.0, 0.20, 0.04),
        _fake_result(-36.0, 0.45, 0.08),
        _fake_result(-32.0, 0.46, 0.22),  # plateau -> large C
        _fake_result(-28.0, 0.50, 0.30),
    ]
    table = caloric_table(results)
    c = table["heat_capacity"]
    assert np.isfinite(c).any()
    # the E=-32 point sits on the T-plateau: C must spike there
    i = int(np.argmin(np.abs(table["energy"] - (-32.0))))
    assert c[i] > np.nanmax(np.delete(c, i))


def test_melting_summary_marks_transition():
    results = [
        _fake_result(-44.0, 0.05, 0.02),
        _fake_result(-36.0, 0.20, 0.04),
        _fake_result(-32.0, 0.205, 0.22),
    ]
    table = caloric_table(results)
    summary = melting_summary(table)
    assert summary["lindemann_crossing"] == -32.0
    assert summary["peak_heat_capacity"] == -32.0


def test_viewer_payload_roundtrip():
    """The embedded JSON must survive a round-trip through json.dumps."""
    import json

    payload = {
        "meta": {"line": "test", "dt": 0.005, "cutoff": 2.5},
        "times": [0.0, 0.05, 0.1],
        "temperatures": [0.1, 0.2, 0.3],
        "pe_per_atom": [[-1.0, -2.0], [-1.1, -2.1], [-1.2, -2.2]],
        "frames": [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]]],
    }
    from lj.visualize import write_viewer

    # rank-specific path: pytest runs this on every MPI rank
    out = f"/tmp/test_viewer_{RANK}.html"
    write_viewer(out, payload)
    with open(out) as fh:
        html = fh.read()
    assert "/*__DATA__*/ null" not in html
    assert "LJ_MIN" not in html
    # payload is valid JSON inside the html
    start = html.index("const DATA = ") + len("const DATA = ")
    end = html.index(";\n", start)
    parsed = json.loads(html[start:end])
    assert parsed["frames"] == payload["frames"]
