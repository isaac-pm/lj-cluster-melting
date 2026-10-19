"""Cluster geometry builders and initial-condition helpers."""

from __future__ import annotations

import numpy as np

GOLDEN = (1.0 + np.sqrt(5.0)) / 2.0
LJ_MIN_DISTANCE = 2.0 ** (1.0 / 6.0)  # pair-potential minimum, in sigma


def icosahedron_surface() -> np.ndarray:
    """The 12 vertices of an icosahedron, normalised to unit circumradius."""
    raw = []
    for s1 in (-1.0, 1.0):
        for s2 in (-1.0, 1.0):
            raw.append((0.0, s1, s2 * GOLDEN))
            raw.append((s1, s2 * GOLDEN, 0.0))
            raw.append((s1 * GOLDEN, 0.0, s2))
    vertices = np.asarray(raw, dtype=float)
    return vertices / np.linalg.norm(vertices, axis=1, keepdims=True)


def mackay_icosahedron() -> np.ndarray:
    """Positions of the LJ-13 Mackay icosahedron (the global minimum).

    One atom at the centre plus the 12 icosahedral vertices. The global
    minimum is the *regular* icosahedron, but its edge length is slightly
    shorter than the pair minimum 2**(1/6) (compressing the surface lets
    the centre and diagonal pairs gain more than the edges lose), so we
    minimise the total energy over the global scale with a golden-section
    search. The resulting energy is E* ~ -44.3268.
    """
    from .physics import lj_energy_and_forces

    surface = icosahedron_surface()
    delta = surface[:, None, :] - surface[None, :, :]
    distances = np.linalg.norm(delta, axis=-1)
    edge = distances[distances > 1e-12].min()
    unit = np.vstack([np.zeros((1, 3)), surface * (LJ_MIN_DISTANCE / edge)])

    def energy(scale: float) -> float:
        return lj_energy_and_forces(unit * scale, cutoff=10.0)[1]

    # golden-section search for the energy-minimising scale
    lo, hi = 0.97, 1.10
    invphi = (np.sqrt(5.0) - 1.0) / 2.0
    c = hi - invphi * (hi - lo)
    d = lo + invphi * (hi - lo)
    fc, fd = energy(c), energy(d)
    for _ in range(80):
        if abs(hi - lo) < 1e-14:
            break
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - invphi * (hi - lo)
            fc = energy(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + invphi * (hi - lo)
            fd = energy(d)
    scale = 0.5 * (lo + hi)
    positions = unit * scale
    return positions - positions.mean(axis=0)


def bcc_lattice_sites(num_particles: int, lattice_constant: float | None = None) -> np.ndarray:
    """Sites of a body-centred cubic cluster around the origin.

    Keeps the lattice sites closest to the origin, so the cluster grows in
    shells around a central particle. Used as a generic fallback for
    cluster sizes without a known minimum-energy geometry.

    The default lattice constant places the nearest-neighbour distance at
    the LJ minimum 2**(1/6), so the cluster starts in a bound, physical
    configuration rather than an artificially dense one.
    """
    import math

    if lattice_constant is None:
        # BCC nearest-neighbour distance is sqrt(3)/2 * a; set it to the LJ minimum
        lattice_constant = LJ_MIN_DISTANCE * 2.0 / np.sqrt(3.0)
    half_width = math.ceil(num_particles ** (1 / 3))
    sites = []
    for i in range(-half_width, half_width + 1):
        for j in range(-half_width, half_width + 1):
            for k in range(-half_width, half_width + 1):
                for site in ((i, j, k), (i + 0.5, j + 0.5, k + 0.5)):
                    sites.append(site)
    sites = np.asarray(sites, dtype=float) * lattice_constant
    order = np.argsort(np.sum(sites**2, axis=1))
    positions = sites[order][:num_particles]
    return positions - positions.mean(axis=0)


def build_cluster(num_particles: int, kind: str = "icosahedral") -> np.ndarray:
    """Initial positions (N, 3) with the centre of mass at the origin."""
    if kind == "icosahedral":
        if num_particles != 13:
            raise ValueError("icosahedral builder is exact for N=13 only")
        return mackay_icosahedron()
    if kind == "bcc":
        return bcc_lattice_sites(num_particles)
    raise ValueError(f"unknown cluster kind {kind!r}")


def maxwell_boltzmann_velocities(
    num_particles: int, temperature: float, rng: np.random.Generator
) -> np.ndarray:
    """Draw velocities from the Maxwell-Boltzmann distribution (kB = m = 1)."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    return rng.normal(0.0, np.sqrt(temperature), size=(num_particles, 3))


def remove_com_velocity(velocities: np.ndarray) -> np.ndarray:
    """Subtract the centre-of-mass velocity (in place) and return it."""
    velocities -= velocities.mean(axis=0)
    return velocities


def rescale_to_total_energy(
    velocities: np.ndarray, potential_energy: float, target_total_energy: float
) -> np.ndarray:
    """Scale velocities so KE + PE equals the target total energy exactly.

    A target exactly at the potential energy (a T=0 start at the minimum)
    yields zero velocities; a target *below* the potential energy is
    unphysical and raises.
    """
    ke_target = target_total_energy - potential_energy
    if ke_target < -1e-9:
        raise ValueError(
            f"total energy {target_total_energy:.4f} is below the potential "
            f"energy {potential_energy:.4f}: no kinetic energy available"
        )
    if ke_target <= 1e-9:
        return np.zeros_like(velocities)
    ke = 0.5 * float(np.sum(velocities**2))
    return velocities * np.sqrt(ke_target / ke)
