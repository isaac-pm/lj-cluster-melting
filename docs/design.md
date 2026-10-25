# Design notes

## Architecture

```
main.py                 CLI; same entry point on laptop and HPC
lj/
  config.py             SimulationConfig (frozen dataclass, LJ reduced units)
  cluster.py            cluster presets, FIRE relaxation, velocities, COM
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

- **Units**: Lennard-Jones reduced units by default (σ = ε = k_B = m = 1).
  The potential length and energy scales are settable: `--sigma` and
  `--epsilon` enter the potential directly (U grows from 4ε[(σ/r)¹² −
  (σ/r)⁶], cutoff rc = 2.5σ), rescaling all reported lengths by σ and
  energies/temperatures by ε while trajectories remain numerically
  identical to the reduced-unit run of the scaled system.
  Energies in ε, lengths in σ, time in τ = σ√(m/ε).
- **Initial geometry**: twelve predefined LJ cluster configurations
  (`lj/cluster.py`, `--list-clusters`): the global minima for
  N = 2–7 and 13, the LJ-38 truncated octahedron, and multi-shell
  Mackay icosahedra (N = 55, 147, 309, 561), cross-validated against
  the Cambridge Cluster Database reference energies to ~10⁻⁴.
  `--num-particles N` picks a preset automatically (BCC lattice
  fallback for other sizes); `--kind` selects one explicitly and
  rejects size mismatches. Idealised constructions are ready-relaxed
  with a deterministic FIRE minimiser (equivalent results under
  `--sigma` rescaling; idempotent at stationary points; ~µs for
  N = 13 to ~4 s for N = 561), so every run starts at its nearest
  local minimum. In scan mode without explicit energies, each N gets
  a suggested window: a full 0.01-resolved sweep for N = 13
  (E* −43.0 → −23.5), a 0.5-spaced grid elsewhere starting just
  above the relaxed potential energy.
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

On top of the recorded trajectory, the "MD (velocity Verlet)" panel
runs real molecular dynamics in the browser: the same kick-drift-kick
verlet as :mod:`lj.integrator` and the same potential-shifted LJ forces
as :mod:`lj.physics`, starting from the trajectory's minimised geometry
(embedded in the payload) and rescaled to ``E_total = E0``. Press
**Run** and the cluster heats up *in real time* from E0 to E1: each
step the ramp advances the total-energy setpoint, whose implied
temperature T(t) = 2(E(t) - PE(t))/(3N-3) is chased by a Berendsen
weak-coupling thermostat — the same ``berendsen_factor`` as the core —
so heat flows in continuously while the atoms keep moving. When the
ramp reaches E1 the thermostat switches off and the hold is free NVE.
Because icosahedral LJ-13 superheats metastably, the melting event
usually fires *during the hold*: the default E1 = -24 crosses the
superheating barrier ~2 min into watching (faster with the speed
select, which also sets the integration rate); the "confine" toggle
acts as a real reflective wall so evaporated atoms stay in view.
The live observables update everywhere — a Lindemann index over the
initial bonds (computed over a rolling ~10-tau window, so it rises
with heating instead of being flattened by the whole-run history)
with a solid/melting/liquid/evaporating phase label,
bonds that glow and break at the physical 1.5-sigma cutoff, a live
caloric trace (green T*, dashed pink E*) on the chart, and per-atom
local-potential-energy colouring on a scale shared with the
trajectory. Pause/Resume and reset included; the energy readout tracks
setpoint vs actual state.
