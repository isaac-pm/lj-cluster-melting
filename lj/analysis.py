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

    # Microcanonical heat capacity C = dE/dT from finite differences of the
    # caloric curve. Diverges at the transition -> the peak is the signal.
    if len(energies) >= 3:
        dtde = np.gradient(t_mean, energies)
        with np.errstate(divide="ignore", invalid="ignore"):
            heat_capacity = np.where(dtde > 1e-9, 1.0 / dtde, np.nan)
    else:
        heat_capacity = np.full_like(energies, np.nan)

    return {
        "energy": energies,
        "temperature_mean": t_mean,
        "temperature_std": np.asarray(t_std),
        "rg_mean": np.asarray(rg_mean),
        "lindemann_mean": np.asarray(lin_mean),
        "heat_capacity": heat_capacity,
        "max_drift_rel": np.asarray(drift),
    }


def melting_summary(table: dict[str, np.ndarray]) -> dict:
    """Rough transition markers from the caloric curve.

    peak_heat_capacity: energy where C peaks (finite-difference estimate of
        the transition region).
    lindemann_crossing: first energy where the Lindemann index exceeds 0.1
        (scanned from low energy upwards).
    """
    valid = ~np.isnan(table["heat_capacity"])
    summary: dict[str, float | None] = {
        "peak_heat_capacity": None,
        "lindemann_crossing": None,
        "note": (
            "finite-resolution estimate: the heat-capacity peak and the "
            "Lindemann 0.1 crossing bracket the solid-liquid crossover"
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
    )
    ax.set_xlabel(r"$E^\star$ ($\varepsilon$)")
    ax.set_ylabel(r"$T^\star$ ($\varepsilon/k_B$)")
    ax.set_title("caloric curve")

    ax = axes[1]
    ax.plot(table["energy"], table["heat_capacity"], "s-", color="#f778ba")
    ax.set_xlabel(r"$E^\star$ ($\varepsilon$)")
    ax.set_ylabel(r"$C^\star = dE/dT$")
    ax.set_title("heat capacity")
    ax.set_ylim(bottom=0)

    ax = axes[2]
    ax.plot(table["energy"], table["lindemann_mean"], "^-", color="#3fb950")
    ax.axhline(0.1, color="#8b949e", ls="--", lw=1, label="0.1")
    ax.legend()
    ax.set_xlabel(r"$E^\star$ ($\varepsilon$)")
    ax.set_ylabel(r"$\delta_L$")
    ax.set_title("Lindemann index")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
