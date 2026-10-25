"""Cluster geometry builders and initial-condition helpers.

This module provides predefined initial configurations for the classic
Lennard-Jones cluster sizes ("magic numbers"), see :data:`CLUSTER_PRESETS`
and ``python main.py --list-clusters``:

====================  =====  =============================================
preset                N      structure / reference minimum E*
====================  =====  =============================================
dimer                 2      D-inf dimer, E* = -1.000
triangle              3      equilateral triangle (D3h), E* = -3.000
tetrahedron           4      regular tetrahedron (Td), E* = -6.000
triangular_bipyramid  5      D3h, E* = -9.103852
octahedron            6      Oh octahedron, E* = -12.712062
pentagonal_bipyramid  7      D5h, E* = -16.505384
icosahedron           13     Mackay icosahedron (Ih), E* = -44.326801
truncated_octahedron  38     fcc truncated octahedron (Oh), E* = -173.928427*
icosahedron55         55     Mackay icosahedron (Ih), E* = -279.248470
icosahedron147        147    Mackay icosahedron (Ih), E* = -876.461207
icosahedron309        309    Mackay icosahedron (Ih), E* = -2007.218985*
icosahedron561        561    Mackay icosahedron (Ih), E* = -3842.393626
====================  =====  =============================================

Reference energies are for the full (unshifted) LJ potential in reduced
units, in agreement with the published global minima (Cambridge Cluster
Database, Wales/Doye & coworkers) for every size; entries marked * were
verified against the database coordinates but are quoted from this code.

Every builder works in Lennard-Jones reduced units (sigma = epsilon = 1)
and leaves the centre of mass at the origin; :func:`build_cluster`
rescales lengths by the requested ``sigma`` and optionally minimises the
geometry with :func:`relax_lj` (FIRE), so a preset start sits at (or very
near) the corresponding local minimum. Sizes without a preset fall back
to a body-centred-cubic cluster (:func:`bcc_lattice_sites`), and
``kind="icosahedral"`` is exact exactly at the Mackay magic numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .physics import lj_energy_and_forces

GOLDEN = (1.0 + np.sqrt(5.0)) / 2.0
LJ_MIN_DISTANCE = 2.0 ** (1.0 / 6.0)  # pair-potential minimum, in sigma
_FULL_POTENTIAL_CUTOFF = 1e6          # large enough to be "no cutoff"

PHI = GOLDEN

# vertices of a regular icosahedron (circumradius sqrt(1 + PHI^2), edge 2)
_ICOSA_VERTICES = np.array(
    [
        [PHI, 0.0, 1.0], [PHI, 0.0, -1.0], [-PHI, 0.0, 1.0], [-PHI, 0.0, -1.0],
        [1.0, PHI, 0.0], [-1.0, PHI, 0.0], [1.0, -PHI, 0.0], [-1.0, -PHI, 0.0],
        [0.0, 1.0, PHI], [0.0, -1.0, PHI], [0.0, 1.0, -PHI], [0.0, -1.0, -PHI],
    ]
)
# the two faces adjacent to vertex k beyond its square-edge partner; the
# 12 vertices split the 20 faces into 6 square (coordinate) edges, 12
# single-triangle and 8 double-triangle groups of the icosa lattice
_SMALL_TRIANGLE_MAP = {
    0: (8, 9), 1: (10, 11), 2: (8, 9), 3: (10, 11),
    4: (0, 1), 5: (2, 3), 6: (0, 1), 7: (2, 3),
    8: (4, 5), 9: (6, 7), 10: (4, 5), 11: (6, 7),
}
_DOUBLE_TRIANGLE_MAP = {0: (9, 6, 8, 4), 1: (11, 6, 10, 4),
                        2: (9, 7, 8, 5), 3: (11, 7, 10, 5)}


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


def _centred(positions: np.ndarray) -> np.ndarray:
    return positions - positions.mean(axis=0)


def _pair_power_sums(positions: np.ndarray) -> tuple[float, float]:
    """Sum of r^-12 and r^-6 over all pairs, in the geometry's own units."""
    delta = positions[:, None, :] - positions[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", delta, delta)
    d2 = d2[np.triu_indices(len(positions), k=1)]
    return float(np.sum(d2**-6.0)), float(np.sum(d2**-3.0))  # (D12, D6)


def optimal_scale(positions: np.ndarray, sigma: float = 1.0) -> float:
    """Uniform dilation factor minimising the full LJ energy of a fixed shape.

    Scaling all distances by ``s`` makes the pair energy a sum of
    ``4 eps [sigma^12 s^-12 r^-12 - sigma^6 s^-6 r^-6]``, i.e. a quadratic
    in ``s^-6``; its minimum is at ``s* = (2 sigma^6 D12 / D6)^(1/6)`` with
    ``D12/D6 = (sum r^-12, sum r^-6)`` over the unscaled pair distances,
    giving the closed-form optimal energy ``E* = -eps D6^2 / D12``.
    """
    d12, d6 = _pair_power_sums(positions)
    return (2.0 * sigma**6 * d12 / d6) ** (1.0 / 6.0)


def relax_lj(
    positions: np.ndarray,
    sigma: float = 1.0,
    cutoff: float | None = None,
    max_iter: int = 20000,
    ftol: float = 1e-9,
) -> np.ndarray:
    """Minimise a configuration into its nearest local minimum with FIRE.

    Deterministic gradient-based local minimisation (Bitzek et al., 2004)
    driven by the analytic LJ forces. ``cutoff=None`` evaluates the full
    unshifted all-pairs potential, so ideal geometries land exactly on the
    literature minima of the LJ cluster; pass the simulation cutoff to
    relax on the shifted potential instead.

    Works in a reduced frame internally and rescales afterwards, so the
    result is independent of ``sigma``/``epsilon`` up to rescaling.
    """
    scale = float(sigma)
    pos = np.asarray(positions, dtype=float) / scale
    if cutoff is None:
        cutoff = _FULL_POTENTIAL_CUTOFF

    forces, energy = lj_energy_and_forces(pos, cutoff)
    if np.abs(forces).max() <= ftol:
        return np.asarray(positions, dtype=float)  # already a stationary point

    v = np.zeros_like(pos)
    dt, dt_max, alpha = 0.002, 0.02, 0.1
    best_energy, best_pos = energy, pos.copy()
    stagnant = 0
    for _ in range(max_iter):
        forces, energy = lj_energy_and_forces(pos, cutoff)
        if energy < best_energy - 1e-12:
            best_energy, best_pos = energy, pos.copy()
            stagnant = 0
        else:
            stagnant += 1
        if np.abs(forces).max() <= ftol or stagnant > 1000:
            break
        p = float((v * forces).sum())
        v = v * (1.0 - alpha) + dt * forces
        if p < 0.0:  # going uphill: reset velocities and shrink step
            v[:] = 0.0
            dt = max(0.5 * dt, 1e-5)
            alpha = 0.1
        else:
            dt = min(1.1 * dt, dt_max)
            alpha = max(0.99 * alpha, 1e-3)
        v += 0.5 * dt * forces
        pos = pos + dt * v
    return best_pos * scale


def mackay_icosahedron(sigma: float = 1.0, num_shells: int = 1) -> np.ndarray:
    """Positions of a multi-shell Mackay icosahedron, centred at the origin.

    ``num_shells`` counts complete icosahedral shells around the centre
    atom: 1 -> the classic LJ-13 icosahedron (the global minimum,
    E* = -44.326801), 2 -> LJ-55, 3 -> LJ-147, 4 -> LJ-309, 5 -> LJ-561,
    following the magic numbers N = (10 s^3 + 15 s^2 + 11 s + 3)/3.

    The construction (Mackay 1962) wraps the 20 tetrahedral wedges of the
    fcc-stacked icosahedral lattice around the centre; it is deterministic
    and exact, then uniformly dilated to the energy-minimising scale, so
    for N=13 this reproduces the global minimum exactly (larger shells
    carry the well-known Mackay strain and remain slightly above their
    relaxed minima until :func:`relax_lj` is applied).
    """
    if num_shells < 1:
        raise ValueError("num_shells must be >= 1")
    points: list[np.ndarray] = [np.zeros(3)]
    for n in range(1, num_shells + 1):
        for k in range(0, 12, 2):
            v1, v2 = _ICOSA_VERTICES[k], _ICOSA_VERTICES[k + 1]
            for i in range(n + 1):
                points.append(i * v1 + (n - i) * v2)
        if n > 1:
            for k in range(12):
                v0 = n * _ICOSA_VERTICES[k]
                v1 = _ICOSA_VERTICES[_SMALL_TRIANGLE_MAP[k][0]] - _ICOSA_VERTICES[k]
                v2 = _ICOSA_VERTICES[_SMALL_TRIANGLE_MAP[k][1]] - _ICOSA_VERTICES[k]
                for i in range(n):
                    for j in range(n - i):
                        if i == 0 and j == 0:
                            continue
                        points.append(v0 + i * v1 + j * v2)
        if n > 2:
            for k in range(4):
                v0 = n * _ICOSA_VERTICES[k]
                edges = [
                    _ICOSA_VERTICES[idx] - _ICOSA_VERTICES[k]
                    for idx in _DOUBLE_TRIANGLE_MAP[k]
                ]
                for i in range(1, n):
                    for j in range(1, n - i):
                        points.append(v0 + i * edges[0] + j * edges[1])
                        points.append(v0 + i * edges[2] + j * edges[3])
    unique = np.unique(np.round(np.asarray(points), 9), axis=0)
    centred = _centred(unique)
    positions = centred * optimal_scale(centred)
    return positions * sigma


def mackay_sizes(limit_shells: int = 20) -> list[int]:
    """Cluster sizes of complete Mackay icosahedra: 13, 55, 147, 309, ..."""
    sizes = []
    for shells in range(1, limit_shells + 1):
        sizes.append((10 * shells**3 + 15 * shells**2 + 11 * shells + 3) // 3)
    return sizes


def _mackay_shells_for_size(num_particles: int) -> int:
    sizes = mackay_sizes(limit_shells=1000)
    try:
        return sizes.index(num_particles) + 1
    except ValueError:
        magic = ", ".join(str(n) for n in sizes[:5])
        raise ValueError(
            f"kind='icosahedral' is exact only at the Mackay magic numbers "
            f"({magic}, ...); got num_particles={num_particles}. "
            f"Use kind='auto' (falls back to a BCC cluster) or another preset."
        ) from None


def dimer() -> np.ndarray:
    """LJ-2 global minimum: one bond at the pair minimum, E* = -1."""
    return _centred(np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]) * LJ_MIN_DISTANCE)


def triangle() -> np.ndarray:
    """LJ-3 global minimum: equilateral triangle, E* = -3."""
    angles = 2.0 * np.pi * np.arange(3) / 3.0
    ring = LJ_MIN_DISTANCE / (2.0 * np.sin(np.pi / 3.0))
    return _centred(np.column_stack([ring * np.cos(angles), ring * np.sin(angles),
                                     np.zeros(3)]))


def tetrahedron() -> np.ndarray:
    """LJ-4 global minimum: regular tetrahedron, E* = -6 exactly."""
    # (±1, ±1, ±1) with an even number of minuses; side length set to the pair minimum
    corners = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0],
                        [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
    return _centred(corners * LJ_MIN_DISTANCE / (2.0 * np.sqrt(2.0)))


def octahedron() -> np.ndarray:
    """LJ-6 global minimum (Oh): the regular octahedron, E* = -12.712062.

    Oh symmetry leaves overall scale as the only degree of freedom, so the
    closed-form dilation reproduces the exact minimum (the nearest-neighbour
    axis contacts sit slightly *beyond* the pair minimum because the
    12 second-shell diagonal pairs pull the other way).
    """
    axes = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
                     [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    positions = _centred(axes)
    return positions * optimal_scale(positions)


def bipyramid(num_ring: int, apex_ratio: float | None = None,
              scan_steps: int = 64) -> np.ndarray:
    """A ``num_ring``-ring bipyramid with the apex heights optimised for LJ.

    Ring atoms sit in the z=0 plane, two apexes above and below. The shape
    has two LJ degrees of freedom (apex-to-ring-height ratio and overall
    scale); the scale is minimised in closed form and the ratio by a
    coarse scan + golden-section refinement, which reproduces the exact
    D3h/D5h-symmetric minima for the N=5 and N=7 global minima.
    """
    if apex_ratio is None:
        def shape_energy(ratio: float) -> float:
            pos = _bipyramid_positions(num_ring, ratio)
            d12, e6 = _pair_power_sums(pos)
            return -e6**2 / d12

        lo, hi = 0.25, 2.5
        grid = np.linspace(lo, hi, scan_steps)
        candidates = [(shape_energy(r), r) for r in grid]
        best_i = min(range(len(candidates)), key=lambda i: candidates[i][0])
        lo = grid[max(0, best_i - 1)]
        hi = grid[min(len(grid) - 1, best_i + 1)]
        invphi = (np.sqrt(5.0) - 1.0) / 2.0
        for _ in range(60):
            m1, m2 = lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0
            if shape_energy(m1) < shape_energy(m2):
                hi = m2
            else:
                lo = m1
        apex_ratio = 0.5 * (lo + hi)
    return _centred(_bipyramid_positions(num_ring, apex_ratio))


def _bipyramid_positions(num_ring: int, apex_ratio: float) -> np.ndarray:
    angles = 2.0 * np.pi * np.arange(num_ring) / num_ring
    radius = 1.0
    ring = np.column_stack([radius * np.cos(angles), radius * np.sin(angles),
                            np.zeros(num_ring)])
    apexes = np.array([[0.0, 0.0, apex_ratio], [0.0, 0.0, -apex_ratio]])
    pos = np.vstack([ring, apexes])
    return _centred(pos) * optimal_scale(_centred(pos))


def truncated_octahedron() -> np.ndarray:
    """LJ-38 global minimum skeleton: the fcc truncated octahedron (Oh).

    All lattice points of the fcc lattice with odd coordinate sum and
    radius r ∈ {1, 2, sqrt(5)} (in units of the lattice spacing / sqrt(2)):
    6 octahedral + 8 cuboctahedral + 24 vertex sites = 38 atoms — the
    famous non-icosahedral exception among LJ global minima (Pillardy &
    Piela 1995; Doye & Wales). A uniform dilation of this ideal shape
    sits slightly above the exact minimum (frozen internal strain);
    :func:`relax_lj` recovers E* = -173.928427.
    """
    sites = [
        (i, j, k)
        for i in (-2, -1, 0, 1, 2)
        for j in (-2, -1, 0, 1, 2)
        for k in (-2, -1, 0, 1, 2)
        if (i + j + k) % 2 == 1 and 0 < i * i + j * j + k * k <= 5
    ]
    positions = np.asarray(sites, dtype=float)
    centred = _centred(positions)
    return centred * optimal_scale(centred)


def bcc_lattice_sites(num_particles: int, lattice_constant: float | None = None) -> np.ndarray:
    """Sites of a body-centred cubic cluster around the origin.

    Keeps the lattice sites closest to the origin, so the cluster grows in
    shells around a central particle. Used as a generic fallback for
    cluster sizes without a known minimum-energy geometry.

    The default lattice constant places the nearest-neighbour distance at
    the LJ minimum 2**(1/6), so the cluster starts in a bound, physical
    configuration rather than an artificially dense one.
    """
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
    return _centred(positions)


@dataclass(frozen=True)
class ClusterPreset:
    """One predefined cluster configuration."""

    num_particles: int
    label: str
    description: str
    builder: Callable[[], np.ndarray]


CLUSTER_PRESETS: dict[str, ClusterPreset] = {
    "dimer": ClusterPreset(
        2, "dimer",
        "LJ-2 global minimum, E* = -1 exactly",
        dimer,
    ),
    "triangle": ClusterPreset(
        3, "equilateral triangle (D3h)",
        "LJ-3 global minimum, E* = -3 exactly",
        triangle,
    ),
    "tetrahedron": ClusterPreset(
        4, "regular tetrahedron (Td)",
        "LJ-4 global minimum, E* = -6 exactly",
        tetrahedron,
    ),
    "triangular_bipyramid": ClusterPreset(
        5, "triangular bipyramid (D3h)",
        "LJ-5 global minimum, E* = -9.103852",
        lambda: bipyramid(3),
    ),
    "octahedron": ClusterPreset(
        6, "octahedron (Oh)",
        "LJ-6 global minimum, E* = -12.712062",
        octahedron,
    ),
    "pentagonal_bipyramid": ClusterPreset(
        7, "pentagonal bipyramid (D5h)",
        "LJ-7 global minimum, E* = -16.505384",
        lambda: bipyramid(5),
    ),
    "icosahedron": ClusterPreset(
        13, "Mackay icosahedron (Ih)",
        "LJ-13 global minimum, E* = -44.326801",
        mackay_icosahedron,
    ),
    "truncated_octahedron": ClusterPreset(
        38, "fcc truncated octahedron (Oh)",
        "LJ-38 global minimum, E* = -173.928427 (relaxed; the famous "
        "non-icosahedral exception)",
        truncated_octahedron,
    ),
    "icosahedron55": ClusterPreset(
        55, "Mackay icosahedron, 2 shells (Ih)",
        "LJ-55 global minimum, E* = -279.248470 (relaxed)",
        lambda: mackay_icosahedron(num_shells=2),
    ),
    "icosahedron147": ClusterPreset(
        147, "Mackay icosahedron, 3 shells (Ih)",
        "LJ-147 global minimum, E* = -876.461207 (relaxed)",
        lambda: mackay_icosahedron(num_shells=3),
    ),
    "icosahedron309": ClusterPreset(
        309, "Mackay icosahedron, 4 shells (Ih)",
        "LJ-309 ground state (Noya & Doye 2006), E* = -2007.218985 (relaxed)",
        lambda: mackay_icosahedron(num_shells=4),
    ),
    "icosahedron561": ClusterPreset(
        561, "Mackay icosahedron, 5 shells (Ih)",
        "LJ-561 global minimum, E* = -3842.393626 (relaxed)",
        lambda: mackay_icosahedron(num_shells=5),
    ),
}


def list_clusters() -> list[tuple[str, ClusterPreset]]:
    """The predefined cluster configurations, in display order."""
    return list(CLUSTER_PRESETS.items())


def preset_for_size(num_particles: int) -> tuple[str, ClusterPreset] | None:
    """The predefined cluster with exactly this many particles, if any."""
    for key, preset in CLUSTER_PRESETS.items():
        if preset.num_particles == num_particles:
            return key, preset
    return None


def _resolve_geometry(num_particles: int, kind: str) -> tuple[np.ndarray, str]:
    """Reduced-unit positions and a label for (size, kind); raises if invalid."""
    if kind in (None, "auto"):
        match = preset_for_size(num_particles)
        kind = match[0] if match else "bcc"
    if kind == "bcc":
        return bcc_lattice_sites(num_particles), "BCC lattice cluster"
    if kind == "icosahedral":
        shells = _mackay_shells_for_size(num_particles)
        return mackay_icosahedron(num_shells=shells), (
            f"Mackay icosahedron ({shells} shell{'s' if shells > 1 else ''})"
        )
    if kind == "truncated_octahedral":
        if num_particles != 38:
            raise ValueError(
                "kind='truncated_octahedral' is exact for N=38 only; "
                f"got num_particles={num_particles}"
            )
        return truncated_octahedron(), "fcc truncated octahedron"
    if kind in CLUSTER_PRESETS:
        preset = CLUSTER_PRESETS[kind]
        if preset.num_particles != num_particles:
            raise ValueError(
                f"preset {kind!r} has N={preset.num_particles}; "
                f"requested N={num_particles}; use kind='auto' or a"
                " matching size"
            )
        return preset.builder(), preset.label
    known = ", ".join(
        ["auto", "icosahedral", "truncated_octahedral", "bcc"] + sorted(CLUSTER_PRESETS)
    )
    raise ValueError(f"unknown cluster kind {kind!r}; known kinds: {known}")


def describe_size(num_particles: int, kind: str = "auto") -> str:
    """Human-readable description of the geometry used for (N, kind)."""
    _, label = _resolve_geometry(num_particles, kind)
    return f"N={num_particles} {label}"


def build_cluster(
    num_particles: int,
    kind: str = "auto",
    sigma: float = 1.0,
    epsilon: float = 1.0,
    relax: bool = True,
) -> np.ndarray:
    """Initial positions (N, 3) with the centre of mass at the origin.

    ``kind`` selects the geometry:

    * ``"auto"`` (default): a predefined cluster preset when ``num_particles``
      matches one (2-7, 13, 38, 55, 147, 309, 561 — see :func:`list_clusters`),
      otherwise the generic BCC cluster;
    * ``"icosahedral"``: Mackay-icosahedral shells, exact exactly at the
      magic numbers 13/55/147/309/561/...;
    * ``"truncated_octahedral"``: the fcc LJ-38 exception;
    * ``"bcc"``: the generic body-centred-cubic fallback for any N;
    * a preset name from :func:`list_clusters` (e.g. ``"pentagonal_bipyramid"``).

    ``relax=True`` (default) minimises the geometry into its nearest local
    minimum with :func:`relax_lj` so runs start from a proper ground state.
    Lengths are rescaled by ``sigma`` at the end; epsilon only governs
    reported energies of the simulation (it never changes the geometry).
    """
    del epsilon  # the geometry depends on sigma only
    positions, _ = _resolve_geometry(num_particles, kind)
    if relax:
        positions = relax_lj(positions)
    return positions * sigma


def suggested_energy_window(
    num_particles: int,
    cutoff: float = 2.5,
    sigma: float = 1.0,
    epsilon: float = 1.0,
    span_per_atom: float = 1.5,
) -> tuple[float, float]:
    """Suggested (cold, hot) total-energy bounds for a melting scan.

    The cold end sits just above the ground-state potential energy of the
    preset cluster (initialised relaxed by :func:`build_cluster`),
    evaluated in the simulation's potential-shifted frame; the hot end
    adds ``span_per_atom`` per particle (the N=13 default sweep spans
    ~19 epsilon over 13 atoms — check/extend the caloric curve as needed).
    """
    positions = build_cluster(num_particles, sigma=sigma)
    _, pe = lj_energy_and_forces(positions, cutoff, sigma, epsilon)
    pe_reduced = pe / epsilon
    floor = pe_reduced + 0.001 * abs(pe_reduced) + 0.001  # hair above the minimum
    e_low = epsilon * math.ceil(10.0 * floor) / 10.0      # round up to 0.1 * epsilon
    e_high = e_low + round(10.0 * span_per_atom * num_particles) / 10.0 * epsilon
    return e_low, e_high


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
