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
- an intro tumble plays on load until the user grabs the scene.
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
  </div>
  <div id="hud"></div>
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
  <select id="colormode" title="atom colour">
    <option value="uniform">uniform colour</option>
    <option value="pe" selected>local potential energy</option>
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
const spinChk   = document.getElementById("spin");
const colorSel  = document.getElementById("colormode");
const resetBtn  = document.getElementById("reset");

const frames = DATA.frames;               // flat [x1,y1,z1, x2,y2,z2, ...] per frame
const nAtoms = frames[0].length / 3;
const hasPE  = Array.isArray(DATA.pe_per_atom) && DATA.pe_per_atom.length === frames.length;

let maxExtent = 0;
for (const f of frames)
  for (const c of f) maxExtent = Math.max(maxExtent, Math.abs(c));

let frame = 0, playing = false, speed = parseFloat(speedSel.value);
let rotX = -24, rotY = 38, zoom = 1;
let lastT = 0;

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
// per-(frame, atom) hue table for the local-PE colour mode, built once
let hueTable = null;
if (hasPE) {
  let lo = Infinity, hi = -Infinity;
  for (const f of DATA.pe_per_atom)
    for (let k = 0; k < nAtoms; k++) {
      if (f[k] < lo) lo = f[k];
      if (f[k] > hi) hi = f[k];
    }
  hueTable = DATA.pe_per_atom.map(f =>
    f.map(v => {
      const t = hi > lo ? (v - lo) / (hi - lo) : 0.5;
      return { h: 215 - 215 * t, s: 72 };   // blue (cold) -> red (hot)
    })
  );
}

function baseHueSat(i) {
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

function renderAtoms() {
  const f = frames[frame];
  const r = 0.185 * scale;
  for (let i = 0; i < nAtoms; i++) {
    const screen = project({ x: f[3*i], y: f[3*i+1], z: f[3*i+2] });
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
const BOND_CUTOFF2 = 1.5 * 1.5;        // sigma^2
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

function renderBonds() {
  const f = frames[frame];
  const pairs = [];
  for (let i = 0; i < nAtoms; i++)
    for (let j = i + 1; j < nAtoms; j++) {
      const dx = f[3*i] - f[3*j], dy = f[3*i+1] - f[3*j+1], dz = f[3*i+2] - f[3*j+2];
      if (dx*dx + dy*dy + dz*dz < BOND_CUTOFF2) pairs.push([i, j]);
    }
  ensureBondCapacity(pairs.length);
  const zRange = 2 * maxExtent * scale;
  for (let k = 0; k < bondEls.length; k++) {
    const el = bondEls[k];
    if (k >= pairs.length || !bondsChk.checked) { el.style.display = "none"; continue; }
    const [i, j] = pairs[k];
    const p1 = project({ x: f[3*i],   y: f[3*i+1],   z: f[3*i+2] });
    const p2 = project({ x: f[3*j],   y: f[3*j+1],   z: f[3*j+2] });
    const dx = p2.x - p1.x, dy = p2.y - p1.y;
    const lenPx = Math.hypot(dx, dy);
    if (lenPx < 1e-6) { el.style.display = "none"; continue; }
    // depth cue: bonds nearer the camera are thicker and more opaque
    const depthT = Math.max(0, Math.min(1, ((p1.z + p2.z) / 2 + maxExtent * scale) / zRange));
    const q = quatFromTo([1, 0, 0], [dx / lenPx, dy / lenPx, 0]);
    el.style.display = "";
    el.style.width = lenPx + "px";
    el.style.opacity = (0.22 + 0.6 * depthT).toFixed(2);
    el.style.transform =
      "translate3d(" + p1.x.toFixed(1) + "px," + p1.y.toFixed(1) + "px," +
        ((p1.z + p2.z) / 2).toFixed(1) + "px)" +
      " rotate3d(" + q[0] + "," + q[1] + "," + q[2] + "," + q[3] + "deg)";
  }
}

// ---------------------------------------------------------------- floor + shadow
function renderFloor() {
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
  const f = frames[frame];
  let cx = 0, rx = 0;
  for (let i = 0; i < nAtoms; i++) {
    const s = project({ x: f[3*i], y: f[3*i+1], z: f[3*i+2] });
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
  shadow.style.opacity = (0.9 * fade).toFixed(2);
  shadow.style.width = w + "px";
  shadow.style.height = (w * 0.42) + "px";
  shadow.style.left = (cx - w / 2) + "px";
  shadow.style.top = (floorScreenY - w * 0.16) + "px";
}

// ---------------------------------------------------------------- HUD
function renderHUD() {
  const t = DATA.times ? DATA.times[frame] : frame;
  const T = DATA.temperatures ? DATA.temperatures[frame] : null;
  hud.innerHTML =
    "t = <b>" + (typeof t === "number" ? t.toFixed(2) : t) + "</b> &tau;" +
    (T === null ? "" : "&nbsp;&nbsp;T = <b>" + T.toFixed(3) + "</b> &epsilon;/k<sub>B</sub>") +
    "&nbsp;&nbsp;frame <b>" + frame + " / " + (frames.length - 1) + "</b>";
  tlabel.textContent = "t = " + (typeof t === "number" ? t.toFixed(2) : t) + " tau";
}

// ---------------------------------------------------------------- frame loop
function render() {
  scale = fitScale() * zoom;
  applyCamera();
  renderAtoms();
  renderBonds();
  renderFloor();
  renderHUD();
  scrub.value = frame;
}

function easeOutCubic(t) { return 1 - Math.pow(1 - t, 3); }

function tick(now) {
  const introT = Math.min(1, (now - introStart) / INTRO_MS);
  if (introT < 1) {                       // intro tumble until user grabs
    const e = easeOutCubic(introT);
    rotX = INTRO_FROM.rotX + (INTRO_TO.rotX - INTRO_FROM.rotX) * e;
    rotY = INTRO_FROM.rotY + (INTRO_TO.rotY - INTRO_FROM.rotY) * e;
    render();
  } else if (playing) {
    if (spinChk.checked) rotY += 0.35 * speed;
    const dt = Math.min((now - lastT) / 1000, 0.1);
    frame += dt * 30 * speed;             // ~30 frames per tau at 1x
    if (frame >= frames.length - 1) { frame = frames.length - 1; setPlaying(false); }
    frame = Math.floor(frame);
    render();
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
  if (!playing && frame >= frames.length - 1) frame = 0;   // replay at the end
  setPlaying(!playing);
});
scrub.addEventListener("input", () => { frame = parseInt(scrub.value, 10); render(); });
speedSel.addEventListener("change", () => { speed = parseFloat(speedSel.value); });
bondsChk.addEventListener("change", renderBonds);
floorChk.addEventListener("change", renderFloor);
colorSel.addEventListener("change", renderAtoms);
resetBtn.addEventListener("click", () => { rotX = INTRO_TO.rotX; rotY = INTRO_TO.rotY; zoom = 1; render(); });

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
  render();
});
viewport.addEventListener("pointerup", () => {
  dragging = false;
  viewport.classList.remove("grabbing");
});
viewport.addEventListener("wheel", e => {
  e.preventDefault();
  zoom = Math.max(0.2, Math.min(8, zoom * Math.pow(1.0015, -e.deltaY)));
  render();
}, { passive: false });

window.addEventListener("keydown", e => {
  if (e.code === "Space") { e.preventDefault(); playBtn.click(); }
  else if (e.code === "ArrowRight") { frame = Math.min(frames.length - 1, frame + 1); render(); }
  else if (e.code === "ArrowLeft")  { frame = Math.max(0, frame - 1); render(); }
});

window.addEventListener("resize", render);

// ---------------------------------------------------------------- boot
document.getElementById("meta").textContent =
  (DATA.meta && DATA.meta.line ? DATA.meta.line + " · " : "") +
  nAtoms + " atoms · " + frames.length + " frames";
scrub.max = frames.length - 1;
render();
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

    pe_per_atom = (
        [per_atom_energy(frame, config.cutoff).tolist() for frame in result.frames]
        if result.frames
        else []
    )
    payload = {
        "meta": {
            "line": f"N={config.num_particles} LJ cluster, E*={config.total_energy}",
            "dt": config.dt,
            "cutoff": config.cutoff,
        },
        "times": result.frame_times,
        "temperatures": result.frame_temperatures,
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
