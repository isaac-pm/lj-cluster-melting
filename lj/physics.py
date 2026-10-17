"""Lennard-Jones forces and energies.

All functions are vectorised NumPy and use a potential-shifted cutoff:

    U(r) = 4 * [(1/r)^12 - (1/r)^6] - U(rc)   for r < rc, else 0.

The shift makes the potential continuous at rc (standard practice, e.g.
GROMACS), at the cost of a small force discontinuity at the cutoff.
"""

from __future__ import annotations

import numpy as np

# Strategy pattern: anything implementing compute(positions) -> (forces, pe).


class ForceEvaluator:
    """Interface for force/potential evaluation.

    The integrator and simulation driver only ever call ``compute``, so
    serial, MPI-decomposed or (future) GPU backends drop in transparently.
    """

    def compute(self, positions: np.ndarray) -> tuple[np.ndarray, float]:
        raise NotImplementedError

    def __call__(self, positions: np.ndarray) -> tuple[np.ndarray, float]:
        return self.compute(positions)


class SerialForces(ForceEvaluator):
    """Reference all-pairs evaluator. O(N^2) per step, exact."""

    def __init__(self, cutoff: float = 2.5):
        self.cutoff = float(cutoff)

    def compute(self, positions: np.ndarray) -> tuple[np.ndarray, float]:
        return lj_energy_and_forces(positions, self.cutoff)


class AtomDecomposedForces(ForceEvaluator):
    """MPI atom decomposition of the force loop.

    Every rank integrates the *full* trajectory redundantly (no per-step
    communication, dynamics identical by construction); rank ``r`` computes
    forces only on its contiguous slice of atoms and the partial potential
    energies are summed with one cheap scalar ``allreduce`` per step.

    This pays off when the O(N^2) force loop dominates (large clusters);
    for N=13 the replica-level parallelism in :mod:`lj.mpi_scan` is the
    right tool instead.
    """

    def __init__(self, cutoff: float = 2.5, comm=None):
        self.cutoff = float(cutoff)
        self.comm = comm
        self.rank = comm.rank if comm is not None else 0
        self.size = comm.size if comm is not None else 1

    def _owned_indices(self, n_atoms: int) -> np.ndarray:
        base, rem = divmod(n_atoms, self.size)
        counts = [base + (1 if r < rem else 0) for r in range(self.size)]
        start = sum(counts[: self.rank])
        return np.arange(start, start + counts[self.rank], dtype=int)

    def compute(self, positions: np.ndarray) -> tuple[np.ndarray, float]:
        owned = self._owned_indices(len(positions))
        delta = positions[owned, None, :] - positions[None, :, :]  # (n_owned, N, 3)
        r2 = np.einsum("ijk,ijk->ij", delta, delta)
        r2[np.arange(len(owned)), owned] = np.inf  # exclude self-interaction
        rc2 = self.cutoff**2
        inside = r2 < rc2

        inv_r2 = np.zeros_like(r2)
        np.divide(1.0, r2, out=inv_r2, where=inside)
        inv_r6 = inv_r2**3
        inv_r12 = inv_r6**2

        u = np.where(inside, 4.0 * (inv_r12 - inv_r6), 0.0)
        u_shift = 4.0 * (rc2**-6 - rc2**-3)
        pe_local = 0.5 * float(np.sum(u - np.where(inside, u_shift, 0.0)))

        f_mag = np.where(inside, 24.0 * (2.0 * inv_r12 - inv_r6) * inv_r2, 0.0)
        forces = np.zeros_like(positions)
        forces[owned] = np.einsum("ij,ijk->ik", f_mag, delta)

        pe = pe_local
        if self.comm is not None:
            from mpi4py import MPI

            pe = self.comm.allreduce(pe_local, op=MPI.SUM)
        return forces, pe


def lj_energy_and_forces(
    positions: np.ndarray, cutoff: float = 2.5
) -> tuple[np.ndarray, float]:
    """Full all-pairs forces and potential energy of an LJ configuration."""
    positions = np.asarray(positions, dtype=float)
    delta = positions[:, None, :] - positions[None, :, :]  # (N, N, 3)
    r2 = np.einsum("ijk,ijk->ij", delta, delta)
    np.fill_diagonal(r2, np.inf)  # exclude self-interaction
    rc2 = cutoff * cutoff
    inside = r2 < rc2

    inv_r2 = np.zeros_like(r2)
    np.divide(1.0, r2, out=inv_r2, where=inside)
    inv_r6 = inv_r2**3
    inv_r12 = inv_r6**2

    u_shift = 4.0 * (rc2**-6 - rc2**-3)
    u = np.where(inside, 4.0 * (inv_r12 - inv_r6) - u_shift, 0.0)
    pe = 0.5 * float(u.sum())

    # F_i = sum_j 24 (2/r_ij^12 - 1_ij^6) * r_vec_ij / r_ij^2
    f_mag = np.where(inside, 24.0 * (2.0 * inv_r12 - inv_r6) * inv_r2, 0.0)
    forces = np.einsum("ij,ijk->ik", f_mag, delta)
    return forces, pe


def per_atom_energy(positions: np.ndarray, cutoff: float = 2.5) -> np.ndarray:
    """Potential energy of each atom (half of each pair, summed)."""
    positions = np.asarray(positions, dtype=float)
    delta = positions[:, None, :] - positions[None, :, :]
    r2 = np.einsum("ijk,ijk->ij", delta, delta)
    np.fill_diagonal(r2, np.inf)
    rc2 = cutoff * cutoff
    inside = r2 < rc2
    inv_r2 = np.zeros_like(r2)
    np.divide(1.0, r2, out=inv_r2, where=inside)
    inv_r6 = inv_r2**3
    inv_r12 = inv_r6**2
    u_shift = 4.0 * (rc2**-6 - rc2**-3)
    u = np.where(inside, 4.0 * (inv_r12 - inv_r6) - u_shift, 0.0)
    return 0.5 * u.sum(axis=1)


def pair_distances(positions: np.ndarray) -> np.ndarray:
    """Upper-triangle pair distances, one per (i < j) pair."""
    positions = np.asarray(positions, dtype=float)
    i, j = np.triu_indices(len(positions), k=1)
    return np.linalg.norm(positions[i] - positions[j], axis=1), i, j


def kinetic_energy(velocities: np.ndarray) -> float:
    return 0.5 * float(np.sum(velocities**2))


def temperature(ke: float, num_particles: int) -> float:
    """Instantaneous kinetic temperature, 3N-3 degrees of freedom (COM removed)."""
    return 2.0 * ke / (3 * num_particles - 3)


def gyration_radius(positions: np.ndarray) -> float:
    centred = positions - positions.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum(centred**2, axis=1))))
