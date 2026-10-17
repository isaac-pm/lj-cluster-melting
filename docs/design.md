# Design notes

## Architecture

```
main.py                 CLI; same entry point on laptop and HPC
lj/
  config.py             SimulationConfig (frozen dataclass, LJ reduced units)
  cluster.py            geometry builders, velocities, COM handling
  physics.py            LJ forces; ForceEvaluator interface (serial / MPI)
  integrator.py         velocity Verlet (kick-drift-kick), Berendsen
  simulation.py         trajectory driver, sampling, Lindemann accumulator
  mpi_scan.py           replica-parallel master/worker scan
  analysis.py           caloric curve, heat capacity, melting summary
  visualize.py          self-contained HTML/CSS/JS 3D viewer
tests/                  pytest; MPI tests run under mpiexec
```

Every module is importable and testable on its own; the physics core has
no MPI dependency at all.

## Parallelism: two levels

1. **Replica level (the workhorse for this project)** — `lj/mpi_scan.py`.
   Rank 0 (master) distributes `(total_energy, seed)` work items
   dynamically; each worker runs an independent trajectory and returns a
   downsampled summary. One process per energy window, exactly the layout
   `hpc/job.sh` anticipated. Embarrassingly parallel: near-linear speedup
   until items run out. Scales: laptop core count → full Aion node.
2. **Data level (for large clusters)** — `lj.physics.AtomDecomposedForces`.
   Every rank integrates the full trajectory redundantly (no per-step
   communication); the O(N²) force loop is split across ranks and partial
   energies are summed with one scalar `allreduce` per step. Pays off when
   the force evaluation dominates (N ≳ few hundred). For N=13 the
   replica level is the right tool; the evaluator interface means a future
   GPU/CUPA backend drops in without touching the integrator.

Without `mpi4py`, or with `mpiexec -n 1`, the same code runs serially.

## Physics

- **Units**: Lennard-Jones reduced units, σ = ε = k_B = m = 1.
  Energies in ε, lengths in σ, time in τ = σ√(m/ε).
- **Initial geometry**: Mackay icosahedron (LJ-13 global minimum,
  E* ≈ −44.3268). The 12 vertices are scaled so the 30 surface edges sit
  at the pair minimum 2^(1/6); the centre atom relaxes slightly inside.
- **Integrator**: velocity Verlet in kick–drift–kick form,

      v(t+dt/2) = v(t) + dt/2·F(t)/m
      r(t+dt)   = r(t) + dt·v(t+dt/2)
      F(t+dt), U(t+dt) evaluated
      v(t+dt)   = v(t+dt/2) + dt/2·F(t+dt)/m

  symplectic and time-reversible; energy drift at dt = 0.005 is ≲ 10⁻⁴
  over 10⁵ steps (asserted in the tests).
- **Cutoff**: 2.5σ, potential-shifted (U → 0 at rc, GROMACS-style).
- **Temperature**: T* = 2·KE/(3N−3) with the COM velocity removed every
  50 steps (3N−3 degrees of freedom, translation projected out).
- **Initial velocities**: Maxwell–Boltzmann at a temperature estimated
  from E₀ − U(r₀), rescaled so E_total = E₀ exactly.
- **Thermostat**: optional Berendsen (NVT); default is microcanonical
  (NVE) for the caloric-curve scan.
- **Melting diagnostics**: Lindemann index over the initial
  icosahedral bonds (cutoff 1.5σ), gyration radius, energy drift.
  The solid→liquid crossover is marked by the heat-capacity peak
  C = dE/dT (finite differences of the caloric curve) and the
  Lindemann 0.1 crossing; see `lj/analysis.py` for the caveats.

## Visualization

`lj/visualize.py` writes a single `.html` file with the trajectory
embedded as JSON — no CDN, no WebGL, works from `file://`. Rendering is
pure CSS 3D (`transform-style: preserve-3d`): atoms are divs placed by
`translate3d`, bonds are divs oriented by quaternions, and the browser
composites in true 3D (perspective scaling, depth). On top of that, a
light at a finite view-space position shades every atom each frame —
highlight offset and brightness follow the atom's 3D position, so the
spheres read as lit, rotating solids rather than flat icons. Bond
width/opacity scale with depth, a perspective floor grid rotates with
the scene, and a soft grounding shadow anchors the cluster. Controls:
play/pause (space), frame scrubbing, speed, bond/floor toggles,
auto-rotate, uniform vs local-potential colouring, drag to rotate,
wheel to zoom, intro tumble on load.
