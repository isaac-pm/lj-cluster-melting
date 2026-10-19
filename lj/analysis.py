"""Post-processing: caloric curve, heat capacity, melting diagnostics."""

from __future__ import annotations

import numpy as np


def caloric_table(results: list[dict]) -> dict[str, np.ndarray]:
    """Aggregate per-run results into per-energy means and deviations.

    Each result dict carries ``total_energy``, ``averages`` (with
    temperature_mean, rg_mean, lindemann_mean, drift_rel) and ``replica``.
    Returns one row per distinct energy, sorted by energy.
    """
    by_energy: dict[float, list[dict]] = {}
    for r in results:
        by_energy.setdefault(float(r["total_energy"]), []).append(r)

    energies, t_mean, t_std, rg_mean, lin_mean, drift = [], [], [], [], [], []
    for energy in sorted(by_energy):
        runs = by_energy[energy]
        temps = [r["averages"]["temperature_mean"] for r in runs]
        rgs = [r["averages"]["rg_mean"] for r in runs]
        lins = [r["averages"]["lindemann_mean"] for r in runs]
        drifts = [r["averages"]["drift_rel"] for r in runs]
        energies.append(energy)
        t_mean.append(float(np.mean(temps)))
        t_std.append(float(np.std(temps)))
        rg_mean.append(float(np.mean(rgs)))
        lin_mean.append(float(np.mean(lins)))
        drift.append(float(np.max(drifts)))

    energies = np.asarray(energies)
    t_mean = np.asarray(t_mean)

    # Microcanonical heat capacity C = dE/dT. Raw finite differences of a
    # noisy few-replica caloric curve explode near the plateau (where
    # dT/dE straddles zero), producing spurious "peaks". We instead smooth
    # T(E) with local polynomial fits and differentiate those, which keeps
    # C = 1/(dT/dE) finite and the peak on the actual plateau.
    dtde = _smooth_dtde(energies, t_mean)
    with np.errstate(divide="ignore", invalid="ignore"):
        heat_capacity = np.where(dtde > 1e-9, 1.0 / dtde, np.nan)

    return {
        "energy": energies,
        "temperature_mean": t_mean,
        "temperature_std": np.asarray(t_std),
        "rg_mean": np.asarray(rg_mean),
        "lindemann_mean": np.asarray(lin_mean),
        "heat_capacity": heat_capacity,
        "max_drift_rel": np.asarray(drift),
    }


def _smooth_dtde(energies: np.ndarray, t_mean: np.ndarray, window: int = 9) -> np.ndarray:
    """Smoothed dT/dE via local polynomial fits (handles non-uniform E).

    ``window`` is the number of neighbouring points used for each local fit.
    It is clamped to the available data so sparse grids still work. The
    default of 9 spans ~0.09 sigma_E on the 0.01-resolved scan grid.
    """
    n = len(t_mean)
    window = min(window, n if n % 2 == 1 else n - 1)
    half = max(window // 2, 1)
    dtde = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        xs = energies[lo:hi] - energies[i]
        ys = t_mean[lo:hi]
        deg = min(2, len(xs) - 1)
        coeffs = np.polyfit(xs, ys, deg)
        # derivative at the centre point xs=0 is the linear coefficient
        dtde[i] = coeffs[-2] if len(coeffs) >= 2 else 0.0
    return dtde


def melting_summary(table: dict[str, np.ndarray]) -> dict:
    """Rough transition markers from the caloric curve.

    peak_heat_capacity: energy where the smoothed C = dE/dT is maximal
        (the caloric plateau).
    lindemann_crossing: first energy where the Lindemann index exceeds 0.1
        (scanned from low energy upwards).
    """
    valid = ~np.isnan(table["heat_capacity"])
    summary: dict[str, float | None] = {
        "peak_heat_capacity": None,
        "lindemann_crossing": None,
        "note": (
            "smoothed finite-resolution estimate: the heat-capacity peak and "
            "the Lindemann 0.1 crossing bracket the solid-liquid crossover"
        ),
    }
    if valid.any():
        idx = np.nanargmax(np.where(valid, table["heat_capacity"], -np.inf))
        summary["peak_heat_capacity"] = float(table["energy"][idx])
    above = np.where(table["lindemann_mean"] > 0.1)[0]
    if len(above):
        summary["lindemann_crossing"] = float(table["energy"][above[0]])
    return summary


def write_caloric_plot(table: dict[str, np.ndarray], path: str) -> None:
    """Three-panel PNG: caloric curve, heat capacity, Lindemann index."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.errorbar(
        table["energy"],
        table["temperature_mean"],
        yerr=table["temperature_std"],
        fmt="o-",
        color="#58a6ff",
        capsize=3,
        markersize=3,
    )
    ax.set_xlabel(r"$E^\star$ ($\varepsilon$)")
    ax.set_ylabel(r"$T^\star$ ($\varepsilon/k_B$)")
    ax.set_title("caloric curve")

    ax = axes[1]
    ax.plot(table["energy"], table["heat_capacity"], "-", color="#f778ba", lw=1)
    ax.set_xlabel(r"$E^\star$ ($\varepsilon$)")
    ax.set_ylabel(r"$C^\star = dE/dT$")
    ax.set_title("heat capacity (smoothed)")
    ax.set_ylim(bottom=0)

    ax = axes[2]
    ax.plot(table["energy"], table["lindemann_mean"], "-", color="#3fb950", lw=1)
    ax.axhline(0.1, color="#8b949e", ls="--", lw=1, label="0.1")
    ax.legend()
    ax.set_xlabel(r"$E^\star$ ($\varepsilon$)")
    ax.set_ylabel(r"$\delta_L$")
    ax.set_title("Lindemann index")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
