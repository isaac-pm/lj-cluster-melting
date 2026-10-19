"""Self-contained HTML/CSS/JS 3D trajectory viewer.

The output is a single .html file with the trajectory embedded as JSON:
no CDN, no WebGL, no dependencies; it works straight from ``file://``.

Rendering is pure CSS 3D (``transform-style: preserve-3d``): atoms are
divs placed with ``translate3d`` inside a rotated scene, bonds are
divs oriented with quaternions, and the browser composites everything
in true 3D (perspective scaling, depth). On top of that:

- a light at a finite view-space position shades every atom each frame
  (highlight offset + brightness follow the 3D position), giving the
  spheres a genuinely three-dimensional, lit appearance;
- bond width/opacity scale with midpoint depth;
- a perspective floor grid rotates with the scene and a soft grounding
  shadow anchors the cluster;
- an intro tumble plays on load until the user grabs the scene;
- a live in-browser MD run: the "MD (velocity Verlet)" panel runs the same
  kick-drift-kick verlet as :mod:`lj.integrator` with the potential-shifted
  LJ forces of :mod:`lj.physics`, and heats the cluster in *real time*
  from a start energy E0 to a target E1: each step, a Berendsen
  weak-coupling thermostat (the same :func:`lj.integrator.berendsen_factor`)
  chases the temperature implied by the ramping setpoint, so heat flows
  in continuously while the atoms keep moving; at E1 the thermostat
  switches off into a free NVE hold, where the metastably superheated
  cluster typically melts (default E1 = -24 crosses the superheating
  barrier within a watchable time). Melting shows up for real:
  a live Lindemann index (mirroring
  :class:`lj.simulation.LindemannAccumulator`, but computed over a
  rolling ~10-tau window so it responds to the melt instead of being
  flattened by the whole-run history), a solid/melting/liquid/
  evaporating phase label, bonds that break at the physical 1.5-sigma
  cutoff, a live caloric trace on the chart, and per-atom
  local-potential-energy colouring. The "confine" toggle acts as a real
  reflective wall during live runs.
"""

from __future__ import annotations

import json
import os

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    margin: 0; display: flex; flex-direction: column; overflow: hidden;
    font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
    background: #0d1117; color: #e6edf3;
  }
  header {
    display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap;
    padding: 10px 16px; border-bottom: 1px solid #21262d;
  }
  header h1 { font-size: 16px; margin: 0; }
  header .meta { color: #8b949e; font-size: 12px; }

  #viewport {
    position: relative; flex: 1; overflow: hidden; touch-action: none;
    perspective: 1100px; cursor: grab;
    background: radial-gradient(ellipse at 50% 38%, #161b22 0%, #0d1117 72%);
  }
  #viewport.grabbing { cursor: grabbing; }
  #camera { position: absolute; left: 50%; top: 50%; width: 0; height: 0;
            transform-style: preserve-3d; }
  #rotor, #zoomer { position: absolute; width: 0; height: 0;
                    transform-style: preserve-3d; }
  .atom, .bond { position: absolute; transform-style: preserve-3d; }
  .atom {
    border-radius: 50%;
    box-shadow: 0 0 16px rgba(0, 0, 0, 0.5);
  }
  .bond {
    border-radius: 2px;
    background: linear-gradient(180deg,
                rgba(40, 46, 54, 0.9), rgba(190, 200, 212, 0.95) 45%,
                rgba(60, 68, 78, 0.9));
    transform-origin: 0 50%;
  }
  #floor {
    position: absolute; transform-style: preserve-3d;
    background-image:
      repeating-linear-gradient(0deg,  rgba(88, 166, 255, 0.22) 0 1px, transparent 1px 26px),
      repeating-linear-gradient(90deg, rgba(88, 166, 255, 0.22) 0 1px, transparent 1px 26px);
    -webkit-mask-image: radial-gradient(circle at 50% 50%, black 0%, transparent 66%);
            mask-image: radial-gradient(circle at 50% 50%, black 0%, transparent 66%);
  }
  #shadow {
    position: absolute; border-radius: 50%; pointer-events: none;
    background: radial-gradient(ellipse at center, rgba(0, 0, 0, 0.55) 0%, transparent 68%);
    filter: blur(7px);
  }
  #pool {
    position: absolute; border-radius: 50%; pointer-events: none;
    background: radial-gradient(ellipse at center,
                rgba(120, 170, 255, 0.14) 0%, rgba(120, 170, 255, 0.05) 45%,
                transparent 70%);
    filter: blur(4px);
  }
  #hud {
    position: absolute; top: 12px; left: 12px; pointer-events: none;
    background: rgba(13, 17, 23, 0.72); border: 1px solid #21262d;
    border-radius: 8px; padding: 8px 12px; font-size: 12px; color: #8b949e;
  }
  #hud b { color: #e6edf3; font-weight: 600; font-variant-numeric: tabular-nums; }

  /* vertical energy axis on the left */
  #energyAxis {
    position: absolute; left: 10px; top: 50%; transform: translateY(-50%);
    display: flex; flex-direction: column; align-items: center; gap: 8px;
    background: rgba(13, 17, 23, 0.72); border: 1px solid #21262d;
    border-radius: 8px; padding: 8px 4px; pointer-events: auto;
  }
  #energyAxis .elabel { font-size: 10px; color: #8b949e; text-align: center; }
  #energyAxis .etitle { font-size: 9px; color: #8b949e; writing-mode: vertical-lr; }
  #energySlider {
    writing-mode: vertical-lr; direction: rtl;
    -webkit-appearance: none; appearance: none;
    width: 220px; height: 6px; border-radius: 3px;
    background: linear-gradient(90deg, #58a6ff, #d29922 60%, #f778ba);
    cursor: pointer; transform: rotate(-90deg); margin: 107px 0;
  }
  #energySlider::-webkit-slider-thumb {
    -webkit-appearance: none; appearance: none;
    width: 14px; height: 14px; border-radius: 50%;
    background: #e6edf3; border: 2px solid #58a6ff;
  }
  #energySlider::-moz-range-thumb {
    width: 12px; height: 12px; border-radius: 50%;
    background: #e6edf3; border: 2px solid #58a6ff;
  }
  #energyReadout { font-size: 9px; color: #e6edf3; text-align: center;
                    font-variant-numeric: tabular-nums; line-height: 1.3; }

  /* velocity-Verlet MD run controls */
  #mdPanel {
    position: absolute; top: 12px; left: 50%; transform: translateX(-50%);
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap; justify-content: center;
    background: rgba(13, 17, 23, 0.78); border: 1px solid #21262d;
    border-radius: 8px; padding: 6px 10px; font-size: 11px; color: #8b949e;
  }
  #mdPanel input[type="number"] {
    width: 58px; background: #21262d; color: #e6edf3;
    border: 1px solid #30363d; border-radius: 4px; padding: 2px 4px;
    font: inherit; font-variant-numeric: tabular-nums;
  }
  #mdPanel button { background: #1f6feb; border: 1px solid #388bfd; color: #e6edf3;
                    border-radius: 4px; padding: 2px 10px; font: inherit; cursor: pointer; }
  #mdPanel button:hover { background: #388bfd; }
  #mdPanel button.running { background: #da3633; border-color: #f85149; }
  #mdPanel button.secondary { background: #30363d; border-color: #3d444d; }
  #mdPanel button.secondary:hover { background: #3d444d; }
  #mdStatus { min-width: 150px; color: #e6edf3; font-variant-numeric: tabular-nums; }

  /* spherical confinement boundary */
  #boundary {
    position: absolute; left: 50%; top: 50%; width: 0; height: 0;
    transform-style: preserve-3d; pointer-events: none;
  }
  .bShell {
    position: absolute; border-radius: 50%;
    border: 1.5px dashed rgba(120, 170, 255, 0.35);
    transform-style: preserve-3d;
  }

  #chart { position: absolute; bottom: 12px; right: 12px; background: rgba(13, 17, 23, 0.72);
           border: 1px solid #21262d; border-radius: 8px; padding: 6px; }
  #chart canvas { display: block; }

  /* ramp toggles under the energy axis */
  #energyAxis label.toggle { font-size: 9px; }
  #rampDuration { width: 44px; font-size: 10px; background: #21262d; color: #e6edf3;
                   border: 1px solid #30363d; border-radius: 4px; padding: 1px 3px; }

  #controls {
    display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
    padding: 10px 16px; border-top: 1px solid #21262d;
  }
  button, select {
    background: #21262d; color: #e6edf3; border: 1px solid #30363d;
    border-radius: 6px; padding: 5px 12px; font: inherit; font-size: 13px;
  }
  button:hover, select:hover { background: #30363d; }
  button { min-width: 74px; }
  input[type="range"] { flex: 1; min-width: 180px; accent-color: #58a6ff; }
  label.toggle { display: flex; align-items: center; gap: 6px;
                 color: #8b949e; font-size: 12px; user-select: none; }
  #tlabel { font-variant-numeric: tabular-nums; color: #8b949e;
            font-size: 12px; min-width: 170px; }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <span class="meta" id="meta"></span>
</header>

<div id="viewport">
  <div id="camera">
    <div id="rotor"><div id="zoomer"><div id="floor"></div></div></div>
    <div id="pool"></div>
    <div id="shadow"></div>
    <div id="boundary"></div>
  </div>
  <div id="hud"></div>
  <div id="mdPanel">
    <span>MD (velocity Verlet)</span>
    <label>E₀ <input type="number" id="mdStart" value="-43.0" step="0.1"></label>
    <label>E₁ <input type="number" id="mdTarget" value="-24.0" step="0.5" title="hot enough that the superheated icosahedron melts during the NVE hold; -24 shows a bound melt, hotter ones also evaporate"></label>
    <label>dt <input type="number" id="mdDt" value="0.005" step="0.001" min="0.001" max="0.02"></label>
    <label>ramp <input type="number" id="mdRamp" value="20000" step="1000" min="500"> steps</label>
    <button id="mdRun">Run</button>
    <button id="mdClear" class="secondary">reset</button>
    <span id="mdStatus"></span>
  </div>
  <div id="energyAxis">
    <span class="etitle">ENERGY</span>
    <input id="energySlider" type="range" step="0.01">
    <span id="energyReadout"></span>
    <label class="toggle"><input type="checkbox" id="rampChk"> heat ramp</label>
    <label class="toggle"><input type="checkbox" id="rampLoop"> loop</label>
    <label class="toggle">&Delta;t <input type="number" id="rampDuration" value="20" min="2" max="300"> s</label>
  </div>
  <div id="chart"><canvas id="chartCanvas" width="360" height="140"></canvas></div>
</div>

<div id="controls">
  <button id="play" type="button">Play</button>
  <input id="scrub" type="range" min="0" value="0" step="1">
  <span id="tlabel"></span>
  <select id="speed" title="playback speed">
    <option value="1">1&times;</option>
    <option value="2">2&times;</option>
    <option value="4" selected>4&times;</option>
    <option value="8">8&times;</option>
  </select>
  <label class="toggle"><input type="checkbox" id="bonds" checked> bonds</label>
  <label class="toggle"><input type="checkbox" id="floorchk" checked> floor</label>
  <label class="toggle"><input type="checkbox" id="spin"> auto-rotate</label>
  <label class="toggle"><input type="checkbox" id="confineChk"> confine</label>
  <select id="colormode" title="atom colour">
    <option value="uniform">uniform colour</option>
    <option value="pe" selected>local potential energy</option>
    <option value="energy">energy axis (melting)</option>
  </select>
  <button id="reset" type="button">reset view</button>
</div>

<script>
"use strict";
/* Trajectory payload, embedded at generation time (no fetch, file:// safe). */
const DATA = /*__DATA__*/ null;

// ---------------------------------------------------------------- state
const viewport  = document.getElementById("viewport");
const camera    = document.getElementById("camera");
const rotor     = document.getElementById("rotor");
const zoomer    = document.getElementById("zoomer");
const floor     = document.getElementById("floor");
const shadow    = document.getElementById("shadow");
const pool      = document.getElementById("pool");
const hud       = document.getElementById("hud");
const playBtn   = document.getElementById("play");
const scrub     = document.getElementById("scrub");
const tlabel    = document.getElementById("tlabel");
const speedSel  = document.getElementById("speed");
const bondsChk  = document.getElementById("bonds");
const floorChk  = document.getElementById("floorchk");
const confineChk = document.getElementById("confineChk");
const boundaryEl = document.getElementById("boundary");

// spherical confinement: atoms bounce off a wall of radius R_CONF (sigma)
const R_CONF = 10;                 // confinement radius
const R_CONF2 = R_CONF * R_CONF;    // squared, for fast distance checks
const spinChk   = document.getElementById("spin");
const colorSel  = document.getElementById("colormode");
const energySlider = document.getElementById("energySlider");
const energyReadout = document.getElementById("energyReadout");
const rampChk = document.getElementById("rampChk");
const rampLoop = document.getElementById("rampLoop");
const rampDuration = document.getElementById("rampDuration");
const resetBtn  = document.getElementById("reset");

// ---------------------------------------------------------------- MD engine
// Real Lennard-Jones molecular dynamics in the browser: potential-shifted
// LJ forces, kick-drift-kick velocity Verlet, and velocity rescaling to ramp
// the total energy from a start value to a target value. Ported directly from
// lj/physics.py and lj/integrator.py (same units: sigma = epsilon = kB = 1).
// Physical LJ parameters ride in the payload (sigma = eps = 1 for the fully
// reduced default). The cutoff is given in units of sigma, as in the Python
// core (config.cutoff in sigma; physical rc = cutoff * sigma).
const LJ_RC =
  ((DATA.meta && DATA.meta.cutoff) || 2.5) * ((DATA.meta && DATA.meta.sigma) || 1.0);
const LJ_SIGMA = (DATA.meta && DATA.meta.sigma) || 1.0;
const LJ_EPS = (DATA.meta && DATA.meta.epsilon) || 1.0;
const LJ_RC2 = LJ_RC * LJ_RC;
const U_SHIFT = 4.0 * LJ_EPS *
  (Math.pow(LJ_SIGMA / LJ_RC, 12) - Math.pow(LJ_SIGMA / LJ_RC, 6));

// Mackay icosahedron vertices (unit circumradius), N = 13
function icosahedronVertices() {
  const phi = (1 + Math.sqrt(5)) / 2;
  const raw = [];
  for (const s1 of [-1, 1]) for (const s2 of [-1, 1]) {
    raw.push([0, s1, s2 * phi], [s1, s2 * phi, 0], [s1 * phi, 0, s2]);
  }
  // normalise to unit circumradius
  return raw.map(v => {
    const n = Math.hypot(v[0], v[1], v[2]);
    return [v[0] / n, v[1] / n, v[2] / n];
  });
}

// Build N-particle initial positions (Mackay icosahedron for 13, else BCC-like)
function buildInitialPositions(n) {
  if (n === 13) {
    const verts = icosahedronVertices();
    // scale so edges sit at the LJ minimum 2^(1/6)
    const edge = Math.hypot(verts[0][0] - verts[1][0], verts[0][1] - verts[1][1], verts[0][2] - verts[1][2]);
    const s = Math.pow(2, 1 / 6) / edge;
    const pos = [[0, 0, 0]];
    for (const v of verts) pos.push([v[0] * s, v[1] * s, v[2] * s]);
    // recentre
    const c = [0, 0, 0];
    for (const p of pos) { c[0] += p[0]; c[1] += p[1]; c[2] += p[2]; }
    for (const p of pos) { p[0] -= c[0] / n; p[1] -= c[1] / n; p[2] -= c[2] / n; }
    return pos;
  }
  // BCC-like fallback: simple cubic shell
  const pos = [];
  const hw = Math.ceil(Math.cbrt(n));
  const a = Math.pow(2, 1 / 6) * 2 / Math.sqrt(3);
  for (let i = -hw; i <= hw && pos.length < n; i++)
    for (let j = -hw; j <= hw && pos.length < n; j++)
      for (let k = -hw; k <= hw && pos.length < n; k++)
        for (const [dx, dy, dz] of [[0, 0, 0], [0.5, 0.5, 0.5]])
          if (pos.length < n)
            pos.push([(i + dx) * a, (j + dy) * a, (k + dz) * a]);
  const c = [0, 0, 0];
  for (const p of pos) { c[0] += p[0]; c[1] += p[1]; c[2] += p[2]; }
  for (const p of pos) { p[0] -= c[0] / n; p[1] -= c[1] / n; p[2] -= c[2] / n; }
  return pos;
}

// Potential-shifted LJ forces on all particles. Returns [forces, potential].
// Pairs are visited exactly once (i < j), so pe accumulates each pair
// energy once — the Python core divides by two only because its N x N
// matrix counts every pair twice (physics.lj_energy_and_forces).
function ljForces(pos) {
  const n = pos.length;
  const forces = Array.from({ length: n }, () => [0, 0, 0]);
  let pe = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const dx = pos[i][0] - pos[j][0];
      const dy = pos[i][1] - pos[j][1];
      const dz = pos[i][2] - pos[j][2];
      const r2 = dx * dx + dy * dy + dz * dz;
      if (r2 >= LJ_RC2) continue;
      const q2 = LJ_SIGMA * LJ_SIGMA / r2;   // (sigma/r)^2
      const q6 = q2 * q2 * q2;
      const q12 = q6 * q6;
      const u = 4 * LJ_EPS * (q12 - q6) - U_SHIFT;
      pe += u;
      const fMag = 24 * LJ_EPS * (2 * q12 - q6) / r2;
      forces[i][0] += fMag * dx; forces[i][1] += fMag * dy; forces[i][2] += fMag * dz;
      forces[j][0] -= fMag * dx; forces[j][1] -= fMag * dy; forces[j][2] -= fMag * dz;
    }
  }
  return [forces, pe];
}

// Velocity Verlet step (kick-drift-kick). Mass = 1.
function verletStep(pos, vel, forces, dt) {
  const n = pos.length;
  for (let i = 0; i < n; i++) {
    vel[i][0] += 0.5 * dt * forces[i][0];
    vel[i][1] += 0.5 * dt * forces[i][1];
    vel[i][2] += 0.5 * dt * forces[i][2];
    pos[i][0] += dt * vel[i][0];
    pos[i][1] += dt * vel[i][1];
    pos[i][2] += dt * vel[i][2];
  }
  const [newForces, newPe] = ljForces(pos);
  for (let i = 0; i < n; i++) {
    vel[i][0] += 0.5 * dt * newForces[i][0];
    vel[i][1] += 0.5 * dt * newForces[i][1];
    vel[i][2] += 0.5 * dt * newForces[i][2];
  }
  return [newForces, newPe];
}

function kineticEnergy(vel) {
  let ke = 0;
  for (const v of vel) ke += 0.5 * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  return ke;
}

function temperatureOf(ke, n) {
  return 2 * ke / (3 * n - 3);
}

// Rescale velocities so total energy hits target exactly
function rescaleToEnergy(pos, vel, pe, eTarget) {
  const keTarget = eTarget - pe;
  if (keTarget <= 0) return;
  const ke = kineticEnergy(vel);
  if (ke <= 1e-12) return;
  const s = Math.sqrt(keTarget / ke);
  for (const v of vel) { v[0] *= s; v[1] *= s; v[2] *= s; }
}

// MD run state
const mdPanel = document.getElementById("mdPanel");
const mdStart = document.getElementById("mdStart");
const mdTarget = document.getElementById("mdTarget");
const mdDt = document.getElementById("mdDt");
const mdRamp = document.getElementById("mdRamp");
const mdRun = document.getElementById("mdRun");
const mdClear = document.getElementById("mdClear");
const mdStatus = document.getElementById("mdStatus");
let mdRunning = false;
let mdDone = false;            // ramp reached E1 and the hold phase elapsed
let mdActive = false;          // a live MD state exists and owns the viewport
let mdPos = null, mdVel = null, mdForces = null, mdPe = 0;
let mdFlat = null;             // live positions as a flat array for the renderer
let mdStepCount = 0;
let mdE0 = 0, mdE1 = 0;
let mdDtVal = 0.005, mdRampSteps = 5000;
let mdRampTarget = 0;          // current annealed total-energy setpoint
let mdRampF = 0;               // ramp progress 0..1
let mdLive = null;             // live observables for HUD / chart / readout
let mdLiveHue = null;          // per-atom colour from live local PE
let mdSamples = [];            // recorded (E, T) samples along the ramp
const MD_TAU = 50;             // COM-removal + sampling stride (com_remove_every)
const MD_TAU_T = 0.1;          // Berendsen coupling time (config.tau_t)
// NVE hold at E1 *after* the ramp: long enough that the metastably
// superheated icosahedron nucleates its melt inside the watchable window
// (measured: E1 = -24 melts ~2.5k steps after switch-off; colder holds can
// stay all-solid for thousands of tau — genuine cluster metastability).
const MD_HOLD_STEPS = 12000;
const MD_BOND_CUTOFF2 = 1.5 * LJ_SIGMA * 1.5 * LJ_SIGMA;   // same as the Python view

// Positions the renderer should draw: the live MD state while it exists,
// otherwise the trajectory frame under the scrubber.
function activePositions() {
  return mdActive && mdFlat ? mdFlat : frames[frame];
}

// Standard normal sample (Box-Muller transform), matching the Maxwell-
// Boltzmann draw of cluster.maxwell_boltzmann_velocities (rng.normal).
function randn() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

// Project out the centre-of-mass velocity (cluster.remove_com_velocity),
// in place.
function removeComVelocity(vel) {
  let cx = 0, cy = 0, cz = 0;
  for (const v of vel) { cx += v[0]; cy += v[1]; cz += v[2]; }
  const n = vel.length;
  for (const v of vel) { v[0] -= cx / n; v[1] -= cy / n; v[2] -= cz / n; }
}

// Per-atom potential energy (half of each pair attributed to each atom),
// mirroring physics.per_atom_energy.
function perAtomPE(pos) {
  const n = pos.length;
  const u = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const dx = pos[i][0] - pos[j][0];
      const dy = pos[i][1] - pos[j][1];
      const dz = pos[i][2] - pos[j][2];
      const r2 = dx * dx + dy * dy + dz * dz;
      if (r2 >= LJ_RC2) continue;
      const inv_r2 = 1 / r2;
      const inv_r6 = inv_r2 * inv_r2 * inv_r2;
      const pair = 0.5 * (4.0 * (inv_r6 * inv_r6 - inv_r6) - U_SHIFT);
      u[i] += pair; u[j] += pair;
    }
  }
  return u;
}

// Lindemann index over the bonds of the initial configuration, mirroring
// simulation.LindemannAccumulator (cutoff 1.5 sigma) — but as a *rolling
// window* over the last MD_LIND_WINDOW samples instead of a whole-run
// average: a live indicator must forget. (A run-cumulative index stays
// flat all run because every solid sample looks alike, and stays diluted
// forever after the melt because the solid history dominates it.)
let mdBonds = [], mdLindSum = null, mdLindSum2 = null, mdLindBuf = [];
const MD_LIND_WINDOW = 200;    // samples at 10-step stride = last 10 tau
function initLindemann(pos) {
  mdBonds = [];
  for (let i = 0; i < pos.length; i++)
    for (let j = i + 1; j < pos.length; j++) {
      const dx = pos[i][0] - pos[j][0];
      const dy = pos[i][1] - pos[j][1];
      const dz = pos[i][2] - pos[j][2];
      if (dx * dx + dy * dy + dz * dz < MD_BOND_CUTOFF2) mdBonds.push([i, j]);
    }
  mdLindSum = new Array(mdBonds.length).fill(0);
  mdLindSum2 = new Array(mdBonds.length).fill(0);
  mdLindBuf = [];
}
function lindemannAdd(pos) {
  const d = new Float64Array(mdBonds.length);
  for (let k = 0; k < mdBonds.length; k++) {
    const [i, j] = mdBonds[k];
    d[k] = Math.hypot(pos[i][0] - pos[j][0],
                      pos[i][1] - pos[j][1],
                      pos[i][2] - pos[j][2]);
    mdLindSum[k] += d[k];
    mdLindSum2[k] += d[k] * d[k];
  }
  mdLindBuf.push(d);
  if (mdLindBuf.length > MD_LIND_WINDOW) {
    // incremental eviction keeps the window means O(1); the divisor is
    // the number of samples actually inside the window
    const old = mdLindBuf.shift();
    for (let k = 0; k < mdBonds.length; k++) {
      mdLindSum[k] -= old[k];
      mdLindSum2[k] -= old[k] * old[k];
    }
  }
}
function lindemannIndex() {
  const n = mdLindBuf.length;
  if (n === 0) return 0;
  let acc = 0;
  for (let k = 0; k < mdBonds.length; k++) {
    const m = mdLindSum[k] / n;
    const variance = Math.max(mdLindSum2[k] / n - m * m, 0);
    acc += Math.sqrt(variance) / m;
  }
  return acc / mdBonds.length;
}

// Microcanonical phase label. The Lindemann 0.2 crossing is the
// solid-liquid marker (as in lj/analysis.py); the gyration radius flags
// actual evaporation, which hot microclusters do instead of melting.
function phaseOf(lind, rg) {
  if (rg > 2.5) return "evaporating";
  if (lind < 0.1) return "solid";
  if (lind < 0.2) return "melting";
  return "liquid";
}

// Exact minimised starting geometry computed by the Python core when the
// payload carries it; the JS icosahedron builder is only the fallback.
function mdInitialPositions(n) {
  if (DATA.meta && Array.isArray(DATA.meta.positions) && DATA.meta.positions.length === 3 * n)
    return Array.from({ length: n }, (_, i) => DATA.meta.positions.slice(3 * i, 3 * i + 3));
  return buildInitialPositions(n);
}

function mdInit() {
  const n = nAtoms;
  mdPos = mdInitialPositions(n);
  const [, pe0] = ljForces(mdPos);
  mdE0 = parseFloat(mdStart.value);
  mdE1 = parseFloat(mdTarget.value);
  mdDtVal = parseFloat(mdDt.value);
  mdRampSteps = Math.max(500, parseFloat(mdRamp.value) || 20000);
  if (!(mdE0 > pe0) || !(mdE1 > pe0)) {
    mdStatus.textContent =
      "e below the cluster minimum PE (" + pe0.toFixed(2) + ") — unphysical";
    return false;
  }
  // Maxwell-Boltzmann at a temperature guessed from E0 - PE, rescaled so
  // E_total = E0 exactly, COM velocity removed first (run_trajectory).
  const tGuess = 2.0 * Math.max(mdE0 - pe0, 0.05) / (3 * n - 3);
  mdVel = Array.from({ length: n }, () => [
    randn() * Math.sqrt(tGuess),
    randn() * Math.sqrt(tGuess),
    randn() * Math.sqrt(tGuess),
  ]);
  removeComVelocity(mdVel);
  rescaleToEnergy(mdPos, mdVel, pe0, mdE0);
  const [f, pe] = ljForces(mdPos);
  mdForces = f; mdPe = pe;
  mdStepCount = 0;
  mdDone = false;
  mdRampTarget = mdE0;
  mdRampF = 0;
  mdActive = true;
  mdSamples = [];
  initLindemann(mdPos);
  lindemannAdd(mdPos);
  mdLiveHue = null;
  mdFlat = [];
  for (const p of mdPos) mdFlat.push(p[0], p[1], p[2]);
  return true;
}

// Simulation playback rate: the viewer's speed multiplier also drives the
// live MD, so 4x plays 4x the physics per real second.
function mdStepsPerFrame() {
  return Math.max(1, Math.round(2 * speed));
}

// One animation tick: mdStepsPerFrame() Verlet steps plus rendering.
function mdAdvance() {
  if (!mdRunning || !mdPos) return;
  const ndof = 3 * nAtoms - 3;
  const confine = confineChk.checked;
  for (let s = 0; s < mdStepsPerFrame(); s++) {
    // annealing ramp: the total-energy setpoint moves linearly E0 -> E1
    const f = Math.min(1, mdStepCount / mdRampSteps);
    mdRampTarget = mdE0 + (mdE1 - mdE0) * f;
    mdRampF = f;
    [mdForces, mdPe] = verletStep(mdPos, mdVel, mdForces, mdDtVal);
    mdStepCount++;
    if (mdStepCount % MD_TAU === 0) removeComVelocity(mdVel);
    // real-time heating: a Berendsen weak-coupling thermostat chases the
    // temperature implied by the setpoint, applied *every step*
    // (berendsen_factor), so heat flows in gently and continuously while
    // the atoms keep moving — no discontinuous velocity kicks. During
    // melting PE rises as the setpoint does, so T* stalls on the caloric
    // plateau just like in the Python scans.
    if (f < 1) {
      const keNow = kineticEnergy(mdVel);
      const tNow = temperatureOf(keNow, nAtoms);
      const tTarget = Math.max(0, 2.0 * (mdRampTarget - mdPe) / ndof);
      if (tTarget > 0 && tNow > 1e-12) {
        const ratio = Math.min(tTarget / tNow, 50.0);
        const fac = Math.sqrt(1.0 + (mdDtVal / MD_TAU_T) * (ratio - 1.0));
        for (const v of mdVel) { v[0] *= fac; v[1] *= fac; v[2] *= fac; }
      }
    }
    // "confine" toggle: a specular reflective wall at R_CONF. During a live
    // run this is a real boundary condition (positions + radial velocity
    // reflected), so hot evaporated atoms stay in the box instead of
    // flying off-screen.
    if (confine) {
      for (let i = 0; i < nAtoms; i++) {
        const p = mdPos[i], v = mdVel[i];
        const r2 = p[0]*p[0] + p[1]*p[1] + p[2]*p[2];
        if (r2 <= R_CONF2) continue;
        const r = Math.sqrt(r2);
        const nx = p[0]/r, ny = p[1]/r, nz = p[2]/r;
        const vn = v[0]*nx + v[1]*ny + v[2]*nz;
        if (vn > 0) {
          v[0] -= 2*vn*nx; v[1] -= 2*vn*ny; v[2] -= 2*vn*nz;
          p[0] = 2*R_CONF*nx - p[0]; p[1] = 2*R_CONF*ny - p[1]; p[2] = 2*R_CONF*nz - p[2];
        }
      }
    }
    // f >= 1: free NVE at the final energy — the thermostat switches off
    if (mdStepCount % MD_TAU === 0) {
      const keS = kineticEnergy(mdVel);
      mdSamples.push({ e: keS + mdPe, t: temperatureOf(keS, nAtoms) });
    }
    if (mdStepCount % 10 === 0) lindemannAdd(mdPos);
  }
  // hold a moment at E1 in pure NVE, then freeze the run (Re-run restarts)
  if (mdStepCount >= mdRampSteps + MD_HOLD_STEPS) {
    mdRunning = false;
    mdDone = true;
    mdRun.textContent = "Re-run";
    mdRun.classList.remove("running");
  }

  // update the live observables and re-render the scene from mdPos
  const ke = kineticEnergy(mdVel);
  const T = temperatureOf(ke, nAtoms);
  const lind = lindemannIndex();
  let srg = 0;
  for (const p of mdPos) srg += p[0]*p[0] + p[1]*p[1] + p[2]*p[2];
  const rg = Math.sqrt(srg / nAtoms);
  const localPE = perAtomPE(mdPos);
  mdLiveHue = localPE.map(u => {
    const t = Math.max(0, Math.min(1, (u - peLo) / Math.max(peHi - peLo, 1e-9)));
    return { h: 215 - 215 * t, s: 72 };
  });
  mdFlat = [];
  for (const p of mdPos) mdFlat.push(p[0], p[1], p[2]);
  const rampF = mdRampF;
  const tTarget = rampF < 1
    ? Math.max(0, 2.0 * (mdRampTarget - mdPe) / ndof) : null;
  mdLive = {
    time: mdStepCount * mdDtVal,
    T: T, ke: ke, pe: mdPe, E: ke + mdPe,
    target: mdRampTarget,
    tTarget: tTarget,
    lind: lind,
    rg: rg,
    phase: phaseOf(lind, rg),
    step: mdStepCount,
    rampF: rampF,
  };
  mdStatus.textContent =
    "step " + mdStepCount +
    " · T*=" + T.toFixed(3) + (tTarget !== null ? " → " + tTarget.toFixed(3) : "") +
    " · E*=" + (ke + mdPe).toFixed(2) +
    " · δL=" + lind.toFixed(3) +
    " · " + mdLive.phase +
    (rampF < 1 ? " · ramp " + Math.round(100 * rampF) + "%" : " · NVE");
  if (hasFramePE) energySel = ke + mdPe;   // pink E*-line shows the actual energy
  render(performance.now());
}

mdRun.addEventListener("click", () => {
  if (mdRunning) {
    mdRunning = false;
    mdRun.textContent = "Resume";
    mdRun.classList.remove("running");
    mdStatus.textContent = "paused (step " + mdStepCount + ")";
    return;
  }
  if (mdDone || !mdPos || mdStepCount === 0) {
    if (!mdInit()) return;
  }
  setPlaying(false);
  thermalActive = false;
  rampChk.checked = false; rampActive = false;
  mdRunning = true;
  mdRun.textContent = "Stop";
  mdRun.classList.add("running");
  mdAdvance();
});

mdClear.addEventListener("click", () => {
  mdRunning = false;
  mdRun.textContent = "Run";
  mdRun.classList.remove("running");
  mdActive = false;
  mdDone = false;
  mdLive = null;
  mdLiveHue = null;
  mdFlat = null;
  mdPos = mdVel = null;
  mdStepCount = 0;
  mdSamples = [];
  mdStatus.textContent = "—";
  render(performance.now());
});

// loop via setInterval so it runs even without trajectory playback
setInterval(() => { if (mdRunning) mdAdvance(); }, 50);

const frames = DATA.frames;               // flat [x1,y1,z1, x2,y2,z2, ...] per frame
const nAtoms = frames[0].length / 3;
const hasPE  = Array.isArray(DATA.pe_per_atom) && DATA.pe_per_atom.length === frames.length;
const hasE   = Array.isArray(DATA.energies) && DATA.energies.length === frames.length;
const hasT   = Array.isArray(DATA.temperatures) && DATA.temperatures.length === frames.length;
const chartCanvas = document.getElementById("chartCanvas");
const chartCtx = chartCanvas.getContext("2d");

let maxExtent = 0;
for (const f of frames)
  for (const c of f) maxExtent = Math.max(maxExtent, Math.abs(c));

let frame = 0, playing = false, speed = parseFloat(speedSel.value);
let rotX = -24, rotY = 38, zoom = 1;
let lastT = 0;

// energy axis: per-frame potential energy and the PE range across frames
const hasFramePE = hasPE;
let framePE = null, peMinAll = 0, peMaxAll = 0;
if (hasFramePE) {
  framePE = DATA.pe_per_atom.map(f => f.reduce((a, b) => a + b, 0));
  peMinAll = Math.min(...framePE);
  peMaxAll = Math.max(...framePE);
}
// energy slider spans from just below the coldest frame PE (deep solid)
// up to well past the hottest frame PE (fully evaporated)
const E_FLOOR = hasFramePE ? peMinAll - 1.0 : -50.0;
const E_CEIL  = hasFramePE ? peMaxAll + 8.0 : -10.0;
let energySel = null;   // null = follow the trajectory's actual E(t)

// Thermal takeover: when the energy slider is active, we drive a live
// melting animation. The slider sets a target total energy E*; the implied
// temperature is T* = 2(E* - PE)/(3N-3). The cluster then heats up in real
// time: atoms vibrate with amplitude ~ sqrt(T*), colors sweep blue->red,
// and bonds break progressively as T* crosses the melting threshold.
const NDOF = 3 * nAtoms - 3;
const T_MELT = 0.15;      // where bonds start breaking (LJ-13 melting)
const T_VAPOR = 0.35;     // where the cluster fully dissociates
let thermalT = 0;          // current animated temperature
let thermalTarget = 0;     // target temperature from the slider
let thermalActive = false; // whether energy mode is driving the view
let rampActive = false;    // heat-ramp mode: auto-sweep the energy slider
let rampStart = 0;         // timestamp when the ramp started (ms)

function impliedTempAt(eSel, i) {
  if (!hasFramePE || eSel === null) return null;
  const pe = framePE[i];
  const ke = eSel - pe;
  return ke > 0 ? 2.0 * ke / NDOF : 0.0;
}
function energyColor(i) {
  // smooth, continuous hue from blue (cold) to red (hot)
  const t = Math.max(0, Math.min(1, thermalT / T_VAPOR));
  return { h: 215 - 215 * t, s: 72 };
}

// energy axis: dragging the vertical slider sets a target total energy E*.
// The implied temperature is T* = 2(E* - PE)/(3N-3); the cluster then heats
// up in real time (thermalT -> thermalTarget), bonds break, and the melting
// process unfolds live on top of the trajectory frame.
if (hasFramePE) {
  energySlider.min = E_FLOOR.toFixed(2);
  energySlider.max = E_CEIL.toFixed(2);
  energySlider.step = "0.01";
  // default: the trajectory's own total energy at frame 0
  energySel = DATA.energies && hasE ? DATA.energies[0] : framePE[0];
  energySlider.value = energySel.toFixed(2);
  const setTarget = () => {
    energySel = parseFloat(energySlider.value);
    thermalTarget = impliedTempAt(energySel, frame) ?? 0;
    thermalActive = true;
  };
  energySlider.addEventListener("input", () => {
    setTarget();
    colorSel.value = "energy";
    render(performance.now());
  });
  colorSel.addEventListener("change", () => {
    if (colorSel.value !== "energy") { thermalActive = false; energySel = null; thermalT = 0; }
    render(performance.now());
  });
  setTarget();
  updateEnergyReadout();

  // heat ramp: auto-sweep the energy slider from E_FLOOR to E_CEIL over
  // rampDuration seconds, driving the full melting crossover hands-free.
  const RAMP_STATES = { IDLE: 0, UP: 1, HOLD_HOT: 2, COOL: 3, HOLD_COLD: 4 };
  let rampState = RAMP_STATES.IDLE;
  let rampStateStart = 0;
  let rampHoldMs = 4000;      // pause at the hot and cold ends
  const rampTargetE = () => parseFloat(energySlider.value);

  function setRampTarget(e) {
    energySel = e;
    energySlider.value = e.toFixed(2);
    thermalTarget = impliedTempAt(e, frame) ?? 0;
    thermalActive = true;
  }

  function advanceRamp(now) {
    if (rampState === RAMP_STATES.IDLE) {
      rampState = RAMP_STATES.UP;
      rampStateStart = now;
    }
    const dur = Math.max(2, parseFloat(rampDuration.value) || 20) * 1000;
    const hold = rampHoldMs;

    if (rampState === RAMP_STATES.UP) {
      const f = Math.min(1, (now - rampStateStart) / dur);
      setRampTarget(E_FLOOR + (E_CEIL - E_FLOOR) * f);
      if (f >= 1) { rampState = RAMP_STATES.HOLD_HOT; rampStateStart = now; }
    } else if (rampState === RAMP_STATES.HOLD_HOT) {
      if (now - rampStateStart > hold) {
        if (rampLoop.checked) { rampState = RAMP_STATES.COOL; rampStateStart = now; }
        else { rampState = RAMP_STATES.IDLE; rampChk.checked = false; thermalActive = false; thermalT = 0; }
      }
    } else if (rampState === RAMP_STATES.COOL) {
      const f = Math.min(1, (now - rampStateStart) / dur);
      setRampTarget(E_CEIL - (E_CEIL - E_FLOOR) * f);
      if (f >= 1) { rampState = RAMP_STATES.HOLD_COLD; rampStateStart = now; }
    } else if (rampState === RAMP_STATES.HOLD_COLD) {
      if (now - rampStateStart > hold) { rampState = RAMP_STATES.UP; rampStateStart = now; }
    }
    render(now);
  }

  rampChk.addEventListener("change", () => {
    rampActive = rampChk.checked;
    colorSel.value = "energy";
    if (rampActive) { advanceRamp(performance.now()); }
    else { rampState = RAMP_STATES.IDLE; thermalActive = false; thermalT = 0; render(performance.now()); }
  });
  rampLoop.addEventListener("change", () => { rampStateStart = performance.now(); });
  rampDuration.addEventListener("input", () => { rampStateStart = performance.now(); });
}
function updateEnergyReadout() {
  // live MD: show the actual state chasing the annealing setpoint
  if (mdActive && mdLive) {
    const tArrow = mdLive.tTarget !== null ? " → " + mdLive.tTarget.toFixed(3) : " · NVE";
    energyReadout.innerHTML =
      "set E* = " + mdLive.target.toFixed(2) + " (ramp " +
        Math.round(100 * mdLive.rampF) + "%)" +
      "<br>E* = " + mdLive.E.toFixed(2) +
      "<br>T* = " + mdLive.T.toFixed(3) + tArrow +
      "<br>&delta;<sub>L</sub> = " + mdLive.lind.toFixed(3);
    return;
  }
  if (!hasFramePE || energySel === null) { energyReadout.textContent = ""; return; }
  const tHere = impliedTempAt(energySel, frame);
  const tStr = tHere === null ? "—" : tHere.toFixed(3);
  const dE = energySel - framePE[frame];
  energyReadout.innerHTML = "E* = " + energySel.toFixed(2) + "<br>T* = " + tStr +
    "<br>&Delta;E = " + (dE >= 0 ? "+" : "") + dE.toFixed(2);
}

// light at a finite position in view space (px, y down positive). Kept
// close to the cluster so the light direction varies from atom to atom —
// that parallax is what makes the spheres read as 3D solids.
const LIGHT = { x: -150, y: -170, z: 230 };
const FLOOR_Y = -1.62;                    // world y of the floor plane (sigma)

// intro tumble: swing the scene around once on load (skip with ?still=1)
const INTRO_FROM = { rotX: 58, rotY: -165 };
const INTRO_TO   = { rotX: -24, rotY: 38 };
const INTRO_MS   = 1600;
const STILL = new URLSearchParams(location.search).has("still");
let introStart = STILL ? -1e18 : performance.now();

// ---------------------------------------------------------------- projection
function fitScale() {
  const w = viewport.clientWidth, h = viewport.clientHeight;
  return Math.min(w, h) * 0.30 / Math.max(maxExtent, 1e-9);
}
let scale = fitScale();

// rotate a world point by the current camera angles (CSS order: X then Y)
function rotateView(p) {
  const cy = Math.cos(rotY * Math.PI / 180), sy = Math.sin(rotY * Math.PI / 180);
  const cx = Math.cos(rotX * Math.PI / 180), sx = Math.sin(rotX * Math.PI / 180);
  const x1 =  p.x * cy + p.z * sy;
  const z1 = -p.x * sy + p.z * cy;
  const y1 = p.y * cx - z1 * sx;
  const z2 = p.y * sx + z1 * cx;
  return { x: x1, y: y1, z: z2 };      // y still world-up here
}
function project(p) {                   // world (sigma) -> screen px
  const v = rotateView(p);
  return { x: v.x * scale, y: -v.y * scale, z: v.z * scale };
}
function applyCamera() {
  rotor.style.transform = "rotateX(" + rotX + "deg) rotateY(" + rotY + "deg)";
}

// ---------------------------------------------------------------- colours
// per-(frame, atom) hue table for the local-PE colour mode, built once.
// peLo/peHi are the normalisation window; the live MD controller reuses it
// so trajectory frames and live-MD atoms share one colour scale.
let hueTable = null;
let peLo = -4.4, peHi = -1.2;   // sensible fallback for N=13 LJ clusters
if (hasPE) {
  let lo = Infinity, hi = -Infinity;
  for (const f of DATA.pe_per_atom)
    for (let k = 0; k < nAtoms; k++) {
      if (f[k] < lo) lo = f[k];
      if (f[k] > hi) hi = f[k];
    }
  peLo = lo;
  peHi = hi;
  hueTable = DATA.pe_per_atom.map(f =>
    f.map(v => {
      const t = hi > lo ? (v - lo) / (hi - lo) : 0.5;
      return { h: 215 - 215 * t, s: 72 };   // blue (cold) -> red (hot)
    })
  );
}

function baseHueSat(i) {
  // live MD colouring wins: per-atom local potential energy, same scale
  if (mdActive && mdLiveHue) return mdLiveHue[i];
  if (colorSel.value === "energy") return energyColor(i);
  if (colorSel.value === "uniform" || !hueTable) return { h: 210, s: 65 };
  return hueTable[frame][i];
}

// Sphere shading: layered gradients per atom, all tracking the light —
// top to bottom: specular dot, terminator crescent (dark side opposite the
// light), body (bright centre offset toward the light fading to a dark rim).
// The gradient centre offsets are fractions of the atom radius, so every
// sphere is lit from the same direction but with per-atom parallax.
function sphereShading(i, screen, r) {
  const dx = LIGHT.x - screen.x, dy = LIGHT.y - screen.y, dz = LIGHT.z - screen.z;
  const dist = Math.hypot(dx, dy, dz) || 1;
  const lx = dx / dist, ly = dy / dist, lz = dz / dist;
  const hx = 0.38 * lx;                       // highlight offset, fraction of r
  const hy = 0.38 * ly;
  const lit = Math.max(-1, Math.min(1, lz));  // 1 = facing the light
  const light = 58 + 14 * lit;
  const hs = baseHueSat(i);
  const c = "hsl(" + hs.h.toFixed(1) + "," + hs.s + "%,";

  const spec = "radial-gradient(circle at " + (50 + hx * 160).toFixed(1) + "% " +
               (50 + hy * 160).toFixed(1) + "%, rgba(255,255,255," +
               (0.95 * Math.max(0, lit)).toFixed(2) + ") 0%, rgba(255,255,255,0) 9%)";

  const umbra = "radial-gradient(circle at " + (50 - hx * 95).toFixed(1) + "% " +
               (50 - hy * 95).toFixed(1) + "%, rgba(0,0,0,0) 42%," +
               "rgba(0,0,0," + (0.30 + 0.25 * (1 - lit) * 0.5).toFixed(2) + ") 100%)";

  const body = "radial-gradient(circle at " + (50 + hx * 105).toFixed(1) + "% " +
               (50 + hy * 105).toFixed(1) + "%," +
               c + (light + 20).toFixed(1) + "%) 0%," +
               c + light.toFixed(1) + "%) 38%," +
               c + (light - 13).toFixed(1) + "%) 70%," +
               c + Math.max(light - 25, 6).toFixed(1) + "%) 100%)";

  // cast shadow: falls opposite the light, always slightly below
  const shx = -hx * r * 1.15, shy = -hy * r * 0.75 + r * 0.38;
  const shadow = shx.toFixed(1) + "px " + shy.toFixed(1) + "px " +
                 (r * 0.45).toFixed(1) + "px rgba(0,0,0,0.55)";

  return {
    background: spec + "," + umbra + "," + body,
    boxShadow: shadow + ", 0 0 " + (r * 0.55).toFixed(1) + "px rgba(0,0,0,0.30)",
  };
}

// ---------------------------------------------------------------- atoms
const atomEls = [];
for (let i = 0; i < nAtoms; i++) {
  const el = document.createElement("div");
  el.className = "atom";
  zoomer.appendChild(el);
  atomEls.push(el);
}

function renderAtoms(tNow) {
  const f = activePositions();
  const r = 0.185 * scale;
  const inEnergy = !mdActive && colorSel.value === "energy" && thermalActive;
  // smooth thermal expansion: cluster grows continuously as T rises
  const meltF = inEnergy ? Math.max(0, Math.min(1, (thermalT - T_MELT) / (T_VAPOR - T_MELT))) : 0;
  const expand = 1 + meltF * 0.45;   // up to 45% linear expansion at vaporisation
  for (let i = 0; i < nAtoms; i++) {
    let px = f[3*i] * expand, py = f[3*i+1] * expand, pz = f[3*i+2] * expand;
    // spherical confinement: reflect atoms that cross the boundary wall
    if (confineChk.checked) {
      const d2 = px*px + py*py + pz*pz;
      if (d2 > R_CONF2) {
        const d = Math.sqrt(d2);
        const s = (2 * R_CONF - d) / d;   // mirror position inside the wall
        px *= s; py *= s; pz *= s;
      }
    }
    const screen = project({ x: px, y: py, z: pz });
    const el = atomEls[i];
    el.style.width = el.style.height = (2 * r) + "px";
    const shading = sphereShading(i, screen, r);
    el.style.background = shading.background;
    el.style.boxShadow = shading.boxShadow;
    // billboard: spheres are rotationally symmetric, so counter-rotate to
    // face the camera — the 3D look comes from the shading, not the plane
    el.style.transform =
      "translate3d(" + (screen.x - r).toFixed(1) + "px," +
                           (screen.y - r).toFixed(1) + "px," +
                           screen.z.toFixed(1) + "px)" +
      " rotateY(" + (-rotY).toFixed(1) + "deg) rotateX(" + (-rotX).toFixed(1) + "deg)";
  }
}

// ---------------------------------------------------------------- bonds
const BOND_CUTOFF = 1.5 * LJ_SIGMA;    // LJ neighbour range
const BOND_CUTOFF2 = BOND_CUTOFF * BOND_CUTOFF;
let bondEls = [];

function ensureBondCapacity(n) {
  while (bondEls.length < n) {
    const el = document.createElement("div");
    el.className = "bond";
    zoomer.appendChild(el);
    bondEls.push(el);
  }
}

function quatFromTo(a, b) {           // rotation taking unit a onto unit b
  const d = a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
  const c = [
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0],
  ];
  if (d < -0.999999) {                // antiparallel: rotate 180 deg about any
    const ax = Math.abs(a[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];  // perpendicular axis
    const cc = [
      a[1]*ax[2] - a[2]*ax[1],
      a[2]*ax[0] - a[0]*ax[2],
      a[0]*ax[1] - a[1]*ax[0],
    ];
    const n = Math.hypot(cc[0], cc[1], cc[2]);
    return [cc[0]/n, cc[1]/n, cc[2]/n, 180];
  }
  const w = 1 + d;
  const n = Math.hypot(c[0], c[1], c[2], w);
  const angle = 2 * Math.atan2(Math.hypot(c[0], c[1], c[2]), w) * 180 / Math.PI;
  return [c[0]/n, c[1]/n, c[2]/n, angle];
}

function renderBonds(tNow) {
  const f = activePositions();
  // fake thermal-takeover melting (energy-slider mode); during a live MD
  // run the physical distances and the 1.5-sigma bond cutoff decide on
  // their own, so no artificial expansion or staged snapping is applied
  const inEnergy = !mdActive && colorSel.value === "energy" && thermalActive;
  const meltF = inEnergy ? Math.max(0, Math.min(1, (thermalT - T_MELT) / (T_VAPOR - T_MELT))) : 0;
  const expand = 1 + meltF * 0.45;
  const confine = confineChk.checked;
  const pairs = [];
  for (let i = 0; i < nAtoms; i++)
    for (let j = i + 1; j < nAtoms; j++) {
      let xi = f[3*i]*expand, yi = f[3*i+1]*expand, zi = f[3*i+2]*expand;
      let xj = f[3*j]*expand, yj = f[3*j+1]*expand, zj = f[3*j+2]*expand;
      if (confine) {
        const di2 = xi*xi + yi*yi + zi*zi;
        if (di2 > R_CONF2) { const d = Math.sqrt(di2); const s = (2 * R_CONF - d) / d; xi *= s; yi *= s; zi *= s; }
        const dj2 = xj*xj + yj*yj + zj*zj;
        if (dj2 > R_CONF2) { const d = Math.sqrt(dj2); const s = (2 * R_CONF - d) / d; xj *= s; yj *= s; zj *= s; }
      }
      const dx = xj - xi, dy = yj - yi, dz = zj - zi;
      const d2 = dx*dx + dy*dy + dz*dz;
      // real bond cutoff (1.5 sigma) in every mode: beyond it there is no
      // bond, so the network breaks apart physically as the cluster melts
      if (d2 >= BOND_CUTOFF2) continue;
      pairs.push([i, j, Math.sqrt(d2)]);
    }
  ensureBondCapacity(pairs.length);
  const zRange = 2 * maxExtent * scale;
  for (let k = 0; k < bondEls.length; k++) {
    const el = bondEls[k];
    if (k >= pairs.length || !bondsChk.checked) { el.style.display = "none"; continue; }
    const [i, j, dist] = pairs[k];
    let xi = f[3*i]*expand, yi = f[3*i+1]*expand, zi = f[3*i+2]*expand;
    let xj = f[3*j]*expand, yj = f[3*j+1]*expand, zj = f[3*j+2]*expand;
    if (confine) {
      const di2 = xi*xi + yi*yi + zi*zi;
      if (di2 > R_CONF2) { const d = Math.sqrt(di2); const s = (2 * R_CONF - d) / d; xi *= s; yi *= s; zi *= s; }
      const dj2 = xj*xj + yj*yj + zj*zj;
      if (dj2 > R_CONF2) { const d = Math.sqrt(dj2); const s = (2 * R_CONF - d) / d; xj *= s; yj *= s; zj *= s; }
    }
    const p1 = project({ x: xi, y: yi, z: zi });
    const p2 = project({ x: xj, y: yj, z: zj });
    const dx = p2.x - p1.x, dy = p2.y - p1.y;
    const lenPx = Math.hypot(dx, dy);
    if (lenPx < 1e-6) { el.style.display = "none"; continue; }
    // depth cue: bonds nearer the camera are thicker and more opaque
    const depthT = Math.max(0, Math.min(1, ((p1.z + p2.z) / 2 + maxExtent * scale) / zRange));
    // strain cue: bonds fade and glow warm as they approach the cutoff
    const strain = Math.min(1, dist / BOND_CUTOFF);
    const q = quatFromTo([1, 0, 0], [dx / lenPx, dy / lenPx, 0]);
    el.style.display = "";
    el.style.width = lenPx + "px";
    el.style.opacity = ((0.22 + 0.6 * depthT) * (1 - strain * 0.45)).toFixed(2);
    if (strain > 0.62) {
      el.style.background = "rgba(255, " + Math.round(180 - 120 * strain) +
        ", 80, " + (0.4 + 0.6 * (strain - 0.62) / 0.38).toFixed(2) + ")";
    } else {
      el.style.background = "";
    }
    el.style.transform =
      "translate3d(" + p1.x.toFixed(1) + "px," + p1.y.toFixed(1) + "px," +
        ((p1.z + p2.z) / 2).toFixed(1) + "px)" +
      " rotate3d(" + q[0] + "," + q[1] + "," + q[2] + "," + q[3] + "deg)";
  }
}

// ---------------------------------------------------------------- floor + shadow
function renderFloor(tNow) {
  if (!floorChk.checked) {
    floor.style.display = "none"; shadow.style.display = "none"; pool.style.display = "none";
    return;
  }
  const size = 9 * scale;
  floor.style.display = "";
  floor.style.width = floor.style.height = size + "px";
  const floorScreenY = -FLOOR_Y * scale;   // world y=FLOOR_Y -> screen y below centre
  floor.style.transform =
    "translate3d(" + (-size / 2) + "px," + floorScreenY + "px,0) rotateX(90deg)";

  // soft grounding shadow + light pool: screen-space ellipses at the floor
  // line, tracking the cluster centroid; fade out as the view goes top-down
  const f = activePositions();
  const inEnergy = !mdActive && colorSel.value === "energy" && thermalActive;
  const meltF = inEnergy ? Math.max(0, Math.min(1, (thermalT - T_MELT) / (T_VAPOR - T_MELT))) : 0;
  const expand = 1 + meltF * 0.45;
  let cx = 0, rx = 0;
  for (let i = 0; i < nAtoms; i++) {
    const px = f[3*i]*expand, py = f[3*i+1]*expand, pz = f[3*i+2]*expand;
    const s = project({ x: px, y: py, z: pz });
    cx += s.x; rx = Math.max(rx, Math.abs(s.x));
  }
  cx /= nAtoms;
  const w = Math.max(50, 2.5 * rx);
  const fade = Math.max(0, Math.cos(rotX * Math.PI / 180));
  pool.style.display = "";
  pool.style.width = (w * 1.5) + "px";
  pool.style.height = (w * 0.62) + "px";
  pool.style.left = (cx - w * 0.75) + "px";
  pool.style.top = (floorScreenY - w * 0.24) + "px";
  shadow.style.display = "";
  shadow.style.opacity = (0.9 * fade * (1 - meltF * 0.5)).toFixed(2);
  shadow.style.width = w + "px";
  shadow.style.height = (w * 0.42) + "px";
  shadow.style.left = (cx - w / 2) + "px";
  shadow.style.top = (floorScreenY - w * 0.16) + "px";
}

// ---------------------------------------------------------------- chart
function renderChart() {
  const W = chartCanvas.width, H = chartCanvas.height;
  chartCtx.clearRect(0, 0, W, H);
  if (!hasE && !hasT) return;

  const padL = 42, padR = 42, padT = 18, padB = 20;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const n = frames.length;
  const tMax = DATA.times ? DATA.times[n - 1] : n - 1;

  // y-axis ranges
  let eLo = Infinity, eHi = -Infinity, tLo = Infinity, tHi = -Infinity;
  if (hasE) for (const e of DATA.energies) { eLo = Math.min(eLo, e); eHi = Math.max(eHi, e); }
  if (hasT) for (const t of DATA.temperatures) { tLo = Math.min(tLo, t); tHi = Math.max(tHi, t); }
  // live MD trace widens the plotted ranges
  const liveTrace = mdActive ? mdSamples : null;
  if (liveTrace && liveTrace.length > 1) {
    for (const s of liveTrace) {
      if (s.e < eLo) eLo = s.e;
      if (s.e > eHi) eHi = s.e;
      if (s.t < tLo) tLo = s.t;
      if (s.t > tHi) tHi = s.t;
    }
  }
  if (hasE) { const m = (eHi - eLo) * 0.08 || 1; eLo -= m; eHi += m; }
  if (hasT) { const m = (tHi - tLo) * 0.15 || 0.01; tLo -= m; tHi += m; }

  const xOf = i => padL + (i / (n - 1)) * plotW;
  const yE = e => padT + plotH - ((e - eLo) / (eHi - eLo)) * plotH;
  const yT = t => padT + plotH - ((t - tLo) / (tHi - tLo)) * plotH;

  // axes
  chartCtx.strokeStyle = "#30363d";
  chartCtx.lineWidth = 1;
  chartCtx.strokeRect(padL, padT, plotW, plotH);
  chartCtx.fillStyle = "#8b949e";
  chartCtx.font = "9px system-ui";
  chartCtx.textAlign = "right";
  if (hasE) {
    chartCtx.fillStyle = "#f778ba";
    chartCtx.fillText(eHi.toFixed(1), padL - 4, padT + 8);
    chartCtx.fillText(eLo.toFixed(1), padL - 4, padT + plotH);
  }
  chartCtx.textAlign = "left";
  if (hasT) {
    chartCtx.fillStyle = "#58a6ff";
    chartCtx.fillText(tHi.toFixed(2), padL + plotW + 4, padT + 8);
    chartCtx.fillText(tLo.toFixed(2), padL + plotW + 4, padT + plotH);
  }
  // x-axis: time
  chartCtx.textAlign = "center";
  chartCtx.fillStyle = "#8b949e";
  chartCtx.fillText("t = 0", padL, H - 6);
  chartCtx.fillText(`t = ${tMax.toFixed(1)}`, padL + plotW, H - 6);

  // curves
  const draw = (series, yFn, color) => {
    chartCtx.strokeStyle = color;
    chartCtx.lineWidth = 1.5;
    chartCtx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = xOf(i), y = yFn(series[i]);
      if (i === 0) chartCtx.moveTo(x, y); else chartCtx.lineTo(x, y);
    }
    chartCtx.stroke();
  };
  if (hasE) draw(DATA.energies, yE, "#f778ba");
  if (hasT) draw(DATA.temperatures, yT, "#58a6ff");

  // legend
  chartCtx.textAlign = "left";
  let lx = padL + 6;
  if (hasE) { chartCtx.fillStyle = "#f778ba"; chartCtx.fillText("E*", lx, padT + 9); lx += 22; }
  if (hasT) { chartCtx.fillStyle = "#58a6ff"; chartCtx.fillText("T*", lx, padT + 9); lx += 22; }

  // live MD traces: the annealing ramp walks out a caloric curve on top of
  // the trajectory curves (green = temperature, dashed pink = total energy)
  if (liveTrace && liveTrace.length > 1) {
    const xLive = i => padL + (i / (liveTrace.length - 1)) * plotW;
    chartCtx.strokeStyle = "#3fb950";
    chartCtx.lineWidth = 2;
    chartCtx.beginPath();
    for (let i = 0; i < liveTrace.length; i++) {
      const x = xLive(i), y = yT(liveTrace[i].t);
      if (i === 0) chartCtx.moveTo(x, y); else chartCtx.lineTo(x, y);
    }
    chartCtx.stroke();
    chartCtx.strokeStyle = "#f778ba";
    chartCtx.lineWidth = 1.5;
    chartCtx.setLineDash([4, 3]);
    chartCtx.beginPath();
    for (let i = 0; i < liveTrace.length; i++) {
      const x = xLive(i), y = yE(liveTrace[i].e);
      if (i === 0) chartCtx.moveTo(x, y); else chartCtx.lineTo(x, y);
    }
    chartCtx.stroke();
    chartCtx.setLineDash([]);
    // marker at the live head of the trace
    const s = liveTrace[liveTrace.length - 1];
    chartCtx.fillStyle = "#3fb950";
    chartCtx.beginPath();
    chartCtx.arc(xLive(liveTrace.length - 1), yT(s.t), 2.5, 0, 2 * Math.PI);
    chartCtx.fill();
    chartCtx.fillStyle = "#3fb950";
    chartCtx.fillText("live T", lx, padT + 9);
    lx += 38;
    chartCtx.fillStyle = "#f778ba";
    chartCtx.fillText("live E*", lx, padT + 9);
  }

  // cursor at current frame
  const cx = xOf(frame);
  chartCtx.strokeStyle = "rgba(230, 237, 243, 0.6)";
  chartCtx.lineWidth = 1;
  chartCtx.setLineDash([3, 3]);
  chartCtx.beginPath();
  chartCtx.moveTo(cx, padT);
  chartCtx.lineTo(cx, padT + plotH);
  chartCtx.stroke();
  chartCtx.setLineDash([]);

  // energy axis: horizontal line at the selected total energy (pink axis)
  if (hasE && energySel !== null && energySel >= eLo && energySel <= eHi) {
    const ey = padT + plotH - ((energySel - eLo) / (eHi - eLo)) * plotH;
    chartCtx.strokeStyle = "#f778ba";
    chartCtx.lineWidth = 1.5;
    chartCtx.beginPath();
    chartCtx.moveTo(padL, ey);
    chartCtx.lineTo(padL + plotW, ey);
    chartCtx.stroke();
    chartCtx.fillStyle = "#f778ba";
    chartCtx.textAlign = "right";
    chartCtx.fillText("E*=" + energySel.toFixed(2), padL - 4, ey - 3);
  }

  // animated temperature marker on the chart's T axis (blue)
  if (colorSel.value === "energy" && thermalActive && hasT) {
    const ty = padT + plotH - ((thermalT - tLo) / (tHi - tLo)) * plotH;
    chartCtx.strokeStyle = "#58a6ff";
    chartCtx.lineWidth = 2;
    chartCtx.beginPath();
    chartCtx.moveTo(padL, ty);
    chartCtx.lineTo(padL + plotW, ty);
    chartCtx.stroke();
    chartCtx.fillStyle = "#58a6ff";
    chartCtx.textAlign = "left";
    chartCtx.fillText("T*=" + thermalT.toFixed(2), padL + plotW + 4, ty - 3);
  }
}

// ---------------------------------------------------------------- HUD
function renderHUD() {
  // live MD run: show the real-time observables of the verlet trajectory
  if (mdActive && mdLive) {
    hud.innerHTML =
      "live MD · t = <b>" + mdLive.time.toFixed(2) + "</b> &tau;" +
      "&nbsp;&nbsp;T = <b>" + mdLive.T.toFixed(3) + "</b> &epsilon;/k<sub>B</sub>" +
      "&nbsp;&nbsp;E* = <b>" + mdLive.E.toFixed(2) + "</b>" +
      "&nbsp;&nbsp;&delta;<sub>L</sub> = <b>" + mdLive.lind.toFixed(3) + "</b>" +
      "&nbsp;(" + mdLive.phase + ")" +
      "&nbsp;&nbsp;step <b>" + mdLive.step + "</b>";
    tlabel.textContent = "live MD · step " + mdLive.step;
    return;
  }
  const t = DATA.times ? DATA.times[frame] : frame;
  const T = DATA.temperatures ? DATA.temperatures[frame] : null;
  // in energy mode, show the animated temperature the cluster is approaching
  const tEff = (colorSel.value === "energy" && thermalActive)
    ? thermalT : T;
  hud.innerHTML =
    "t = <b>" + (typeof t === "number" ? t.toFixed(2) : t) + "</b> &tau;" +
    (tEff === null ? "" : "&nbsp;&nbsp;T = <b>" + tEff.toFixed(3) + "</b> &epsilon;/k<sub>B</sub>") +
    (colorSel.value === "energy"
      ? "&nbsp;&nbsp;E* = <b>" + energySel.toFixed(2) + "</b>" : "") +
    "&nbsp;&nbsp;frame <b>" + frame + " / " + (frames.length - 1) + "</b>";
  tlabel.textContent = "t = " + (typeof t === "number" ? t.toFixed(2) : t) + " tau";
}

// ---------------------------------------------------------------- boundary
function renderBoundary() {
  const on = confineChk.checked;
  boundaryEl.style.display = on ? "" : "none";
  if (!on) return;
  const R = R_CONF * scale;
  const rings = [
    { rx: 0, ry: 0 },      // equator
    { rx: 90, ry: 0 },     // meridian 1
    { rx: 90, ry: 90 },    // meridian 2
  ];
  if (boundaryEl.childElementCount !== rings.length) {
    boundaryEl.innerHTML = "";
    for (let k = 0; k < rings.length; k++) {
      const el = document.createElement("div");
      el.className = "bShell";
      boundaryEl.appendChild(el);
    }
  }
  rings.forEach((ring, k) => {
    const el = boundaryEl.children[k];
    el.style.width = el.style.height = (2 * R) + "px";
    el.style.transform =
      "translate3d(" + (-R) + "px," + (-R) + "px,0)" +
      " rotateX(" + ring.rx + "deg) rotateY(" + ring.ry + "deg)";
  });
}

// ---------------------------------------------------------------- frame loop
function render(tNow) {
  scale = fitScale() * zoom;
  applyCamera();
  renderAtoms(tNow);
  renderBonds(tNow);
  renderFloor(tNow);
  renderBoundary();
  renderHUD();
  renderChart();
  updateEnergyReadout();
  scrub.value = frame;
}

function easeOutCubic(t) { return 1 - Math.pow(1 - t, 3); }

function tick(now) {
  const introT = Math.min(1, (now - introStart) / INTRO_MS);
  if (introT < 1) {                       // intro tumble until user grabs
    const e = easeOutCubic(introT);
    rotX = INTRO_FROM.rotX + (INTRO_TO.rotX - INTRO_FROM.rotX) * e;
    rotY = INTRO_FROM.rotY + (INTRO_TO.rotY - INTRO_FROM.rotY) * e;
    render(now);
  } else {
    // heat ramp mode: energy sweeps automatically, temperature follows
    if (rampActive) {
      advanceRamp(now);
    }
    // thermal takeover: always advance temperature toward target in energy mode
    else if (colorSel.value === "energy" && thermalActive) {
      const dt = Math.min((now - lastT) / 1000, 0.1);
      const rate = 0.06;  // slow, continuous heating (temperature units per second)
      if (thermalT < thermalTarget) thermalT = Math.min(thermalTarget, thermalT + rate * dt);
      else if (thermalT > thermalTarget) thermalT = Math.max(thermalTarget, thermalT - rate * dt);
      render(now);
    } else if (playing) {
      if (spinChk.checked) rotY += 0.35 * speed;
      const dt = Math.min((now - lastT) / 1000, 0.1);
      frame += dt * 30 * speed;             // ~30 frames per tau at 1x
      if (frame >= frames.length - 1) { frame = frames.length - 1; setPlaying(false); }
      frame = Math.floor(frame);
      render(now);
    }
  }
  lastT = now;
  requestAnimationFrame(tick);
}

function setPlaying(p) {
  playing = p;
  playBtn.textContent = p ? "Pause" : "Play";
}

// ---------------------------------------------------------------- interaction
playBtn.addEventListener("click", () => {
  if (mdRunning) { mdRun.click(); return; }   // pause the live MD first
  if (!playing && frame >= frames.length - 1) frame = 0;   // replay at the end
  setPlaying(!playing);
});
scrub.addEventListener("input", () => { frame = parseInt(scrub.value, 10); render(performance.now()); });
speedSel.addEventListener("change", () => { speed = parseFloat(speedSel.value); });
bondsChk.addEventListener("change", () => renderBonds(performance.now()));
floorChk.addEventListener("change", () => renderFloor(performance.now()));
confineChk.addEventListener("change", () => render(performance.now()));
colorSel.addEventListener("change", () => renderAtoms(performance.now()));
resetBtn.addEventListener("click", () => { rotX = INTRO_TO.rotX; rotY = INTRO_TO.rotY; zoom = 1; render(performance.now()); });

let dragging = false, lastX = 0, lastY = 0;
viewport.addEventListener("pointerdown", e => {
  dragging = true; lastX = e.clientX; lastY = e.clientY;
  introStart = -1e18;                        // cancel the intro tumble
  viewport.classList.add("grabbing");
  viewport.setPointerCapture(e.pointerId);
});
viewport.addEventListener("pointermove", e => {
  if (!dragging) return;
  rotY += (e.clientX - lastX) * 0.45;
  rotX = Math.max(-89, Math.min(89, rotX - (e.clientY - lastY) * 0.45));
  lastX = e.clientX; lastY = e.clientY;
  render(performance.now());
});
viewport.addEventListener("pointerup", () => {
  dragging = false;
  viewport.classList.remove("grabbing");
});
viewport.addEventListener("wheel", e => {
  e.preventDefault();
  zoom = Math.max(0.2, Math.min(8, zoom * Math.pow(1.0015, -e.deltaY)));
  render(performance.now());
}, { passive: false });

window.addEventListener("keydown", e => {
  if (e.code === "Space") { e.preventDefault(); playBtn.click(); }
  else if (e.code === "ArrowRight") { frame = Math.min(frames.length - 1, frame + 1); render(performance.now()); }
  else if (e.code === "ArrowLeft")  { frame = Math.max(0, frame - 1); render(performance.now()); }
});

window.addEventListener("resize", () => render(performance.now()));

// ---------------------------------------------------------------- boot
document.getElementById("meta").textContent =
  (DATA.meta && DATA.meta.line ? DATA.meta.line + " · " : "") +
  nAtoms + " atoms · " + frames.length + " frames";
// the MD panel's start energy defaults to this run's own total energy
if (DATA.meta && typeof DATA.meta.energy === "number") {
  mdStart.value = DATA.meta.energy.toFixed(2);
}
scrub.max = frames.length - 1;
render(performance.now());
requestAnimationFrame(t => { lastT = t; requestAnimationFrame(tick); });
</script>
</body>
</html>
"""


def _round_payload(payload: dict, digits: int = 4) -> dict:
    def rnd(obj):
        if isinstance(obj, float):
            return round(obj, digits)
        if isinstance(obj, list):
            return [rnd(v) for v in obj]
        return obj

    return rnd(payload)


def write_viewer(out_path: str, payload: dict, title: str = "LJ cluster") -> str:
    """Write the standalone viewer; trajectory is embedded, no files needed."""
    html = _TEMPLATE.replace("__TITLE__", title).replace(
        "/*__DATA__*/ null", json.dumps(_round_payload(payload), separators=(",", ":"))
    )
    with open(out_path, "w") as fh:
        fh.write(html)
    return out_path


def write_trajectory_outputs(result, outdir: str, title: str = "LJ cluster") -> tuple[str, str]:
    """Write trajectory.json + the HTML viewer for a RunResult."""
    os.makedirs(outdir, exist_ok=True)
    config = result.config

    from .physics import per_atom_energy
    from . import cluster

    kind = "icosahedral" if config.num_particles == 13 else "bcc"
    pe_per_atom = (
        [per_atom_energy(frame, config.cutoff, config.sigma, config.epsilon).tolist()
         for frame in result.frames]
        if result.frames
        else []
    )
    payload = {
        "meta": {
            "line": f"N={config.num_particles} LJ cluster, E*={config.total_energy}",
            "dt": config.dt,
            "cutoff": config.cutoff,
            "sigma": config.sigma,
            "epsilon": config.epsilon,
            "energy": config.total_energy,
            # exact minimised starting geometry (Mackay icosahedron / BCC) so
            # the viewer's live-MD run starts from the same configuration as
            # the Python trajectories
            "positions": cluster.build_cluster(
                config.num_particles, kind=kind, sigma=config.sigma
            )
            .reshape(-1)
            .tolist(),
        },
        "times": result.frame_times,
        "temperatures": result.frame_temperatures,
        "energies": result.frame_energies,
        "pe_per_atom": pe_per_atom,
        # flat [x1,y1,z1, x2,y2,z2, ...] per frame, as the JS expects
        "frames": [frame.reshape(-1).tolist() for frame in result.frames],
    }

    json_path = os.path.join(outdir, "trajectory.json")
    with open(json_path, "w") as fh:
        json.dump(_round_payload(payload), fh)

    html_path = os.path.join(outdir, "cluster.html")
    write_viewer(html_path, payload, title=title)
    return json_path, html_path
