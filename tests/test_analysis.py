"""Analysis and visualization tests (no MPI needed)."""

import numpy as np
import pytest

from lj.analysis import caloric_table, melting_summary, write_lj_potential_plot

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


def _caloric_profile(energies, e_plateau, t_lo, width, t_hi=None):
    """Monotonic T(E) with a plateau at e_plateau (steep-flat-steep S-curve).

    dT/dE is small on the plateau (T nearly constant) and large elsewhere,
    so C = 1/(dT/dE) peaks at the plateau. Always monotonic increasing.
    ``t_hi`` accepted for backward compatibility (unused).
    """
    dtde = 0.30 - 0.28 * np.exp(-((energies - e_plateau) / width) ** 2)
    t = t_lo + np.concatenate([[0.0], np.cumsum(0.5 * (dtde[1:] + dtde[:-1]) * np.diff(energies))])
    return t


def test_heat_capacity_on_dense_grid_finds_plateau():
    """A 0.01-resolved grid with a clear plateau must put the C peak on it."""
    energies = np.round(np.arange(-34.0, -26.0, 0.01), 2)
    t_vals = _caloric_profile(energies, e_plateau=-31.0, t_lo=0.05, t_hi=None, width=0.3)
    results = [_fake_result(e, t, 0.05) for e, t in zip(energies, t_vals)]
    table = caloric_table(results)
    c = table["heat_capacity"]
    valid = np.isfinite(c)
    assert valid.any()
    peak_e = table["energy"][valid][np.argmax(c[valid])]
    assert -31.5 < peak_e < -30.5, f"C peak {peak_e} not on the plateau"


def test_melting_summary_marks_transition():
    """Dense grid: the C peak and Lindemann crossing land on the transition."""
    energies = np.round(np.arange(-44.0, -28.0, 0.01), 2)
    t_vals = _caloric_profile(energies, e_plateau=-32.0, t_lo=0.02, t_hi=None, width=1.0)
    results = []
    for e, t in zip(energies, t_vals):
        lin = 0.05 if e < -32.0 else 0.20
        results.append(_fake_result(e, t, lin))
    table = caloric_table(results)
    summary = melting_summary(table)
    # grid quantisation: the first point above -32.0 is -31.99
    assert abs(summary["lindemann_crossing"] - (-32.0)) < 0.02
    assert -32.5 < summary["peak_heat_capacity"] < -31.5


def test_viewer_payload_roundtrip():
    """The embedded JSON must survive a round-trip through json.dumps."""
    import json

    payload = {
        "meta": {"line": "test", "dt": 0.005, "cutoff": 2.5},
        "times": [0.0, 0.05, 0.1],
        "temperatures": [0.1, 0.2, 0.3],
        "energies": [-30.0, -30.1, -29.9],
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
    assert "chartCanvas" in html
    assert "energySlider" not in html
    assert "rampChk" not in html
    # payload is valid JSON inside the html
    start = html.index("const DATA = ") + len("const DATA = ")
    end = html.index(";\n", start)
    parsed = json.loads(html[start:end])
    assert parsed["frames"] == payload["frames"]
    assert parsed["energies"] == payload["energies"]
    assert parsed["temperatures"] == payload["temperatures"]


def test_lj_potential_plot_is_written(tmp_path):
    out = tmp_path / "potential.png"
    write_lj_potential_plot(str(out))
    assert out.stat().st_size > 0
