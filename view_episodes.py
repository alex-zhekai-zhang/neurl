"""Build an HTML replay viewer for the recorded training episodes and open it in the browser.

three_booster/run.py and four_booster/run.py write every 100th episode to <suite>/outputs/<task>_<model>/
episodes.json. This bundles those recordings into one self-contained page: figures/episodes.html.

    python view_episodes.py                         # every run found
    python view_episodes.py four_booster/outputs/2landing_mlp/episodes.json
    python view_episodes.py --no-open               # just write the file

Keys in the viewer: space = play/pause, left/right = previous/next episode, , and . = step one frame.
"""
import argparse
import json
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="recording JSON files (default: */outputs/*/episodes.json, recordings/*.json)")
    ap.add_argument("--out", default=str(ROOT / "figures" / "episodes.html"))
    ap.add_argument("--no-open", action="store_true", help="write the HTML without opening a browser")
    args = ap.parse_args()

    files = [Path(f) for f in args.files] or (
        sorted(ROOT.glob("*/outputs/*/episodes.json")) + sorted((ROOT / "recordings").glob("*.json")))
    if not files:
        raise SystemExit("no recordings found - run three_booster/run.py or four_booster/run.py first")
    runs = [json.loads(f.read_text()) for f in files]

    data = json.dumps(runs, separators=(",", ":")).replace("</", "<\\/")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(TEMPLATE.replace("__DATA__", data))
    n = sum(len(r["episodes"]) for r in runs)
    print(f"wrote {out} ({len(runs)} run(s), {n} episodes, {out.stat().st_size / 1e6:.1f} MB)")
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lander Replays</title>
<style>
:root {
  color-scheme: dark;
  --bg: #0d0e12;
  --panel: #15161b;
  --panel-2: #1c1d23;
  --surface: #1a1a19;
  --line: #2c2c2a;
  --ink: #f0efec;
  --ink-2: #c3c2b7;
  --muted: #898781;
  --series: #3987e5;
  --good: #0ca30c;
  --warning: #fab219;
  --critical: #d03b3b;
  --radius: 12px;
}
* { box-sizing: border-box; }
html, body { margin: 0; background: var(--bg); color: var(--ink);
  font: 14px/1.45 ui-sans-serif, -apple-system, "SF Pro Text", "Segoe UI", Roboto, sans-serif; }
button { font: inherit; color: inherit; }
.num { font-variant-numeric: tabular-nums; }

.page { max-width: 1320px; margin: 0 auto; padding: 20px 16px 40px; }
header { display: flex; flex-wrap: wrap; align-items: end; justify-content: space-between; gap: 12px; margin-bottom: 16px; }
h1 { font-size: 22px; font-weight: 650; letter-spacing: -0.01em; margin: 0; }
.sub { color: var(--muted); margin-top: 2px; }
.runs { display: flex; gap: 6px; flex-wrap: wrap; }
.chip { border: 1px solid var(--line); background: var(--panel); border-radius: 999px; padding: 6px 14px; cursor: pointer; }
.chip[aria-pressed="true"] { border-color: var(--series); background: color-mix(in srgb, var(--series) 18%, var(--panel)); }

.layout { display: grid; grid-template-columns: minmax(0, 1fr) 280px; gap: 16px; }
@media (max-width: 900px) { .layout { grid-template-columns: 1fr; } }

.stage { background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden; }
.viewport { position: relative; aspect-ratio: 3 / 2; background: #05060a; }
.viewport canvas { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }
.hud { position: absolute; inset: 0; pointer-events: none; padding: 14px 16px; display: flex; flex-direction: column; justify-content: space-between; }
.hud-row { display: flex; justify-content: space-between; align-items: start; gap: 12px; }
.hud .big { font-size: 22px; font-weight: 650; letter-spacing: -0.01em; text-shadow: 0 1px 8px rgba(0,0,0,.6); }
.hud .small { color: var(--ink-2); font-size: 12.5px; text-shadow: 0 1px 6px rgba(0,0,0,.7); }
.hud .right { text-align: right; }
.badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: 999px; font-size: 12.5px; font-weight: 600;
  background: rgba(10,10,14,.72); border: 1px solid var(--line); backdrop-filter: blur(6px); }
.badge svg { width: 14px; height: 14px; flex: none; }
.badge.good { color: var(--good); } .badge.critical { color: var(--critical); }
.badge.warning { color: var(--warning); } .badge.muted { color: var(--ink-2); }
.badge .lbl { color: var(--ink); font-weight: 550; }
.hud-bottom { display: flex; gap: 8px; flex-wrap: wrap; }
.outcome { opacity: 0; transform: translateY(-4px); transition: opacity .25s, transform .25s; margin-top: 6px; }
.outcome.show { opacity: 1; transform: none; }

.transport { display: flex; align-items: center; gap: 10px; padding: 10px 12px; border-top: 1px solid var(--line); flex-wrap: wrap; }
.iconbtn { width: 36px; height: 36px; border-radius: 10px; border: 1px solid var(--line); background: var(--panel-2); display: grid; place-items: center; cursor: pointer; }
.iconbtn:hover { border-color: #3d3d44; }
.iconbtn svg { width: 18px; height: 18px; }
.iconbtn.primary { background: var(--series); border-color: var(--series); color: #fff; width: 42px; }
.seg { display: inline-flex; border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
.seg button { border: 0; background: var(--panel-2); padding: 7px 10px; cursor: pointer; color: var(--ink-2); }
.seg button + button { border-left: 1px solid var(--line); }
.seg button[aria-pressed="true"] { background: color-mix(in srgb, var(--series) 22%, var(--panel-2)); color: var(--ink); }
.scrub { flex: 1 1 200px; display: flex; align-items: center; gap: 10px; min-width: 160px; }
input[type=range] { flex: 1; accent-color: var(--series); }
.toggle { display: inline-flex; align-items: center; gap: 6px; color: var(--ink-2); cursor: pointer; user-select: none; }
.toggle input { accent-color: var(--series); }

aside { background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius); display: flex; flex-direction: column; min-height: 0; }
aside h2, .card h2 { font-size: 13px; font-weight: 600; color: var(--ink-2); margin: 0; padding: 12px 14px 8px; letter-spacing: .02em; }
.eplist { overflow-y: auto; max-height: 560px; padding: 0 8px 8px; }
@media (max-width: 900px) { .eplist { max-height: 300px; } }
.ep { width: 100%; display: grid; grid-template-columns: 1fr auto; align-items: center; gap: 2px 8px; text-align: left;
  border: 1px solid transparent; background: transparent; border-radius: 9px; padding: 8px 10px; cursor: pointer; }
.ep:hover { background: var(--panel-2); }
.ep[aria-current="true"] { background: color-mix(in srgb, var(--series) 16%, var(--panel-2)); border-color: color-mix(in srgb, var(--series) 55%, transparent); }
.ep .name { font-weight: 600; }
.ep .ret { text-align: right; font-weight: 600; }
.ep .meta { grid-column: 1 / -1; display: flex; flex-wrap: wrap; gap: 2px 6px; align-items: center; color: var(--ink-2); font-size: 12px; }
.ep .meta svg { width: 12px; height: 12px; }
.ep .tag { color: var(--warning); }
.ep .icon-good { color: var(--good); } .ep .icon-critical { color: var(--critical); }
.ep .icon-warning { color: var(--warning); } .ep .icon-muted { color: var(--muted); }

.card { background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius); margin-top: 16px; position: relative; }
.card .desc { color: var(--muted); font-size: 12.5px; padding: 0 14px; margin-top: -4px; }
.chart { position: relative; padding: 4px 6px 8px; }
.chart svg { display: block; width: 100%; height: 240px; }
.tooltip { position: absolute; pointer-events: none; background: #0b0b0e; border: 1px solid #34343a; border-radius: 8px; padding: 8px 10px;
  font-size: 12.5px; color: var(--ink); white-space: nowrap; opacity: 0; transition: opacity .1s; box-shadow: 0 6px 20px rgba(0,0,0,.4); }
.tooltip .t-muted { color: var(--ink-2); }
.empty { padding: 40px; text-align: center; color: var(--muted); }
kbd { font: 11px ui-monospace, monospace; border: 1px solid var(--line); border-bottom-width: 2px; border-radius: 4px; padding: 0 4px; color: var(--ink-2); }
.keys { color: var(--muted); font-size: 12px; padding: 10px 14px 12px; border-top: 1px solid var(--line); }
</style>
</head>
<body>
<div class="page">
  <header>
    <div>
      <h1>Lander replays</h1>
      <div class="sub" id="subtitle"></div>
    </div>
    <div class="runs" id="runs"></div>
  </header>

  <div class="layout">
    <div class="stage">
      <div class="viewport" id="viewport">
        <canvas id="canvas"></canvas>
        <div class="hud">
          <div class="hud-row">
            <div>
              <div class="big num" id="hud-ep">Episode 0</div>
              <div class="small num" id="hud-step">step 0 / 0</div>
            </div>
            <div class="right">
              <div class="small">return</div>
              <div class="big num" id="hud-ret">0.0</div>
              <div class="outcome" id="hud-outcome"></div>
            </div>
          </div>
          <div class="hud-bottom" id="hud-scales"></div>
        </div>
      </div>
      <div class="transport">
        <button class="iconbtn" id="prev" title="Previous episode (←)" aria-label="Previous episode">
          <svg viewBox="0 0 24 24" fill="currentColor"><path d="M6 5h2v14H6zM20 5v14L9 12z"/></svg></button>
        <button class="iconbtn primary" id="play" title="Play / pause (space)" aria-label="Play or pause"></button>
        <button class="iconbtn" id="next" title="Next episode (→)" aria-label="Next episode">
          <svg viewBox="0 0 24 24" fill="currentColor"><path d="M16 5h2v14h-2zM4 5v14l11-7z"/></svg></button>
        <div class="seg" id="speeds" role="group" aria-label="Playback speed"></div>
        <div class="scrub">
          <input type="range" id="scrub" min="0" max="0" value="0" aria-label="Frame">
        </div>
        <label class="toggle"><input type="checkbox" id="auto" checked> Auto-advance</label>
      </div>
    </div>

    <aside>
      <h2 id="eplist-title">Recorded episodes</h2>
      <div class="eplist" id="eplist"></div>
      <div class="keys"><kbd>space</kbd> play · <kbd>←</kbd><kbd>→</kbd> episode · <kbd>,</kbd><kbd>.</kbd> frame</div>
    </aside>
  </div>

  <div class="card">
    <h2>Return per episode</h2>
    <div class="desc" id="chart-desc">Faint line: every training episode. Bold line: 20-episode mean. Dots: recorded episodes, click to watch.</div>
    <div class="chart" id="chart">
      <svg id="chart-svg" role="img" aria-label="Return per training episode"></svg>
      <div class="tooltip" id="tooltip"></div>
    </div>
  </div>
</div>

<script id="data" type="application/json">__DATA__</script>
<script>
"use strict";
const RUNS = JSON.parse(document.getElementById("data").textContent);
const $ = (id) => document.getElementById(id);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const ICON = {
  play: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M7 4v16l13-8z"/></svg>',
  pause: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M6 4h4v16H6zM14 4h4v16h-4z"/></svg>',
  landed: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M4 12.5l5 5L20 6.5"/></svg>',
  crashed: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round"><path d="M6 6l12 12M18 6L6 18"/></svg>',
  "out of bounds": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><path d="M7 17L17 7M9 7h8v8"/></svg>',
  "timed out": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round"><circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/></svg>',
  warn: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3.5l9.5 16.5h-19z"/><path d="M12 10v4.5M12 17.5v.01"/></svg>',
  flag: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M6 21V4M6 4h11l-2.5 4L17 12H6"/></svg>',
  dash: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round"><path d="M6 12h12"/></svg>',
};
ICON["landed on good pad"] = ICON.landed; ICON["landed on bad pad"] = ICON.warn; ICON["landed off pad"] = ICON.dash;
const OUTCOME_TONE = { landed: "good", "landed on good pad": "good", "landed on bad pad": "critical", "landed off pad": "muted",
  crashed: "critical", "out of bounds": "warning", "timed out": "muted" };
const OUTCOME_LABEL = { landed: "Landed", "landed on good pad": "Landed on good pad", "landed on bad pad": "Landed on bad pad",
  "landed off pad": "Landed off the pads", crashed: "Crashed", "out of bounds": "Flew off-screen", "timed out": "Timed out" };
const PAD_COLOR = { good: "#3ccf6e", bad: "#ef5350", single: "#fab219" };
const SPEEDS = [0.5, 1, 2, 4];

// ---------- state ----------
const S = { run: null, ep: 0, frame: 0, playing: true, speed: 1, acc: 0, hold: 0, auto: true, particles: [], dpr: 1, s: 1 };

function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a);
  t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const fmt = (x) => (x < 0 ? "−" : "") + Math.abs(x).toFixed(1);

// ---------- run setup ----------
function prepareRun(run) {
  const g = run.geometry;
  run.W = g.viewport[0] / g.scale; run.H = g.viewport[1] / g.scale;
  run.param_log = run.param_log || run.scale_log || [];  // older recordings used scale_*
  run.param_labels = run.param_labels || run.scale_labels || {};
  for (const ep of run.episodes) {
    let c = 0; ep.cum = ep.frames.map((f) => (c += f.r));
    ep.params = ep.params || ep.scales || {};
    if (!ep.pads) ep.pads = [[...ep.helipad, "pad", 1]];
  }
  // chart bands: periods where a task parameter differs from its value at the start of training
  const byKey = {};
  for (const [e, k, v] of run.param_log) (byKey[k] = byKey[k] || []).push([e, v]);
  run.bands = [];
  for (const k in byKey) {
    const ch = byKey[k];
    if (ch.length > 41) continue;  // e.g. a random switch every episode: too many to shade
    ch.forEach(([e, v], i) => {
      if (v === ch[0][1]) return;
      run.bands.push({ start: e, end: i + 1 < ch.length ? ch[i + 1][0] : run.returns.length, key: k, value: v });
    });
  }
}
RUNS.forEach(prepareRun);
const label = (run, k) => run.param_labels[k] || k.replace(/_/g, " ");
const paramText = (run, k, v) => typeof v === "number" ? `${label(run, k)} at ${Math.round(v * 100)}%` : `${label(run, k)}: ${v}`;

// ---------- canvas ----------
const canvas = $("canvas"), ctx = canvas.getContext("2d");
let staticLayer = document.createElement("canvas");

function resize() {
  const r = $("viewport").getBoundingClientRect();
  S.dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(r.width * S.dpr); canvas.height = Math.round(r.height * S.dpr);
  S.s = canvas.width / S.run.W;
  buildStatic();
}

function terrainY(ep, x) {
  const t = ep.terrain;
  if (x <= t[0][0]) return t[0][1];
  for (let i = 1; i < t.length; i++) if (x <= t[i][0]) {
    const [x0, y0] = t[i - 1], [x1, y1] = t[i]; return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
  }
  return t[t.length - 1][1];
}

function worldTransform(c) { c.setTransform(S.s, 0, 0, -S.s, 0, S.run.H * S.s); }

function buildStatic() {
  const run = S.run, ep = run.episodes[S.ep]; if (!ep) return;
  staticLayer.width = canvas.width; staticLayer.height = canvas.height;
  const c = staticLayer.getContext("2d"), W = run.W, H = run.H, px = 1 / S.s * S.dpr;
  const rnd = mulberry32(ep.episode * 7919 + 17);
  worldTransform(c);

  // sky
  const sky = c.createLinearGradient(0, H, 0, 0);
  sky.addColorStop(0, "#04050b"); sky.addColorStop(0.65, "#0b1022"); sky.addColorStop(1, "#151a33");
  c.fillStyle = sky; c.fillRect(0, 0, W, H);
  const neb = c.createRadialGradient(W * 0.78, H * 0.78, 0, W * 0.78, H * 0.78, W * 0.45);
  neb.addColorStop(0, "rgba(90,70,190,0.16)"); neb.addColorStop(1, "rgba(90,70,190,0)");
  c.fillStyle = neb; c.fillRect(0, 0, W, H);

  // stars
  S.twinkle = [];
  for (let i = 0; i < 170; i++) {
    const x = rnd() * W, y = rnd() * H; if (y < terrainY(ep, x) + 0.2) continue;
    const r = (0.35 + rnd() * rnd() * 1.3) * px, a = 0.25 + rnd() * 0.7;
    if (i % 6 === 0) { S.twinkle.push({ x, y, r: r * 1.2, a, ph: rnd() * 6.28, sp: 1 + rnd() * 2 }); continue; }
    c.fillStyle = `rgba(230,235,255,${a})`; c.beginPath(); c.arc(x, y, r, 0, 6.283); c.fill();
  }
  // a distant planet
  const pr = 0.9, pxp = W * (0.15 + rnd() * 0.2), pyp = H * 0.8;
  const pg = c.createRadialGradient(pxp - pr * 0.4, pyp + pr * 0.4, pr * 0.1, pxp, pyp, pr);
  pg.addColorStop(0, "#6c7bb8"); pg.addColorStop(1, "#1d2340");
  c.fillStyle = pg; c.beginPath(); c.arc(pxp, pyp, pr, 0, 6.283); c.fill();

  // terrain
  const t = ep.terrain;
  c.beginPath(); c.moveTo(0, 0); for (const [x, y] of t) c.lineTo(x, y); c.lineTo(W, 0); c.closePath();
  const ground = c.createLinearGradient(0, ep.pads[0][2] + 1.5, 0, 0);
  ground.addColorStop(0, "#5a5a63"); ground.addColorStop(0.35, "#3a3a42"); ground.addColorStop(1, "#1b1b20");
  c.fillStyle = ground; c.fill();
  c.save(); c.clip();
  for (let i = 0; i < 46; i++) {  // craters
    const x = rnd() * W, top = terrainY(ep, x), r = 0.08 + rnd() * rnd() * 0.42, y = 0.2 + rnd() * Math.max(0, top - 0.5);
    c.fillStyle = "rgba(0,0,0,0.24)"; c.beginPath(); c.ellipse(x, y, r, r * 0.45, 0, 0, 6.283); c.fill();
    c.strokeStyle = "rgba(255,255,255,0.07)"; c.lineWidth = 1.2 * px;
    c.beginPath(); c.ellipse(x, y - r * 0.06, r, r * 0.45, 0, 3.4, 6.0); c.stroke();
  }
  c.restore();
  c.beginPath(); t.forEach(([x, y], i) => (i ? c.lineTo(x, y) : c.moveTo(x, y)));
  c.strokeStyle = "rgba(235,235,245,0.45)"; c.lineWidth = 1.5 * px; c.stroke();

  // landing pads: amber for the single stock pad; green (good) and red (bad) when there are two
  const sc = run.geometry.scale, labels = [];
  for (const [hx1, hx2, hy, , good] of ep.pads) {
    const col = !ep.two_pads ? PAD_COLOR.single : good ? PAD_COLOR.good : PAD_COLOR.bad;
    const padGlow = c.createLinearGradient(0, hy, 0, hy + 0.6);
    padGlow.addColorStop(0, col + "26"); padGlow.addColorStop(1, col + "00");
    c.fillStyle = padGlow; c.fillRect(hx1, hy, hx2 - hx1, 0.6);
    c.strokeStyle = col; c.lineWidth = 2.5 * px;
    c.beginPath(); c.moveTo(hx1, hy); c.lineTo(hx2, hy); c.stroke();
    for (const x of [hx1, hx2]) {
      c.strokeStyle = "#d8d8e0"; c.lineWidth = 1.5 * px;
      c.beginPath(); c.moveTo(x, hy); c.lineTo(x, hy + 50 / sc); c.stroke();
      c.fillStyle = col;
      c.beginPath(); c.moveTo(x, hy + 50 / sc); c.lineTo(x, hy + 40 / sc); c.lineTo(x + 25 / sc, hy + 45 / sc); c.closePath(); c.fill();
    }
    if (ep.two_pads) labels.push({ x: (hx1 + hx2) / 2, y: hy + 50 / sc, text: good ? "GOOD" : "BAD", col });
  }
  // pad labels, drawn in screen space so the text isn't mirrored
  c.setTransform(1, 0, 0, 1, 0, 0);
  c.font = `600 ${11 * S.dpr}px ui-sans-serif, -apple-system, system-ui, sans-serif`;
  c.textAlign = "center"; c.textBaseline = "bottom";
  for (const l of labels) {
    c.fillStyle = l.col; c.fillText(l.text, l.x * S.s, (run.H - l.y) * S.s - 4 * S.dpr);
  }
}

// ---------- particles ----------
function spawn(ep, i) {
  const f = ep.frames[i], prev = ep.frames[Math.max(0, i - 1)], fps = S.run.geometry.fps;
  const lvx = (f.l[0] - prev.l[0]) * fps, lvy = (f.l[1] - prev.l[1]) * fps;
  for (const [x, y, dx, dy, p, kind] of f.f) {
    const n = Math.hypot(dx, dy) || 1, ux = dx / n, uy = dy / n;
    if (p <= 0) {  // engine fired but scaled to zero power: sputter smoke
      if (Math.random() < 0.35) S.particles.push({ x, y, vx: lvx + ux * 0.6 + (Math.random() - 0.5) * 0.4,
        vy: lvy + uy * 0.6 + (Math.random() - 0.5) * 0.4, age: 0, life: 0.6, smoke: true, size: 0.07 });
      continue;
    }
    const count = kind === 0 ? 3 : 1, sp = kind === 0 ? 5.5 : 3.2;
    for (let k = 0; k < count; k++) {
      const j = (Math.random() - 0.5) * 0.5;
      S.particles.push({ x, y, vx: lvx + (ux + -uy * j) * sp * p * (0.7 + Math.random() * 0.6),
        vy: lvy + (uy + ux * j) * sp * p * (0.7 + Math.random() * 0.6), age: 0,
        life: (kind === 0 ? 0.5 : 0.3) + Math.random() * 0.35, size: kind === 0 ? 0.06 : 0.04 });
    }
  }
  if (S.particles.length > 900) S.particles.splice(0, S.particles.length - 900);
}

function updateParticles(dt) {
  if (dt <= 0) return;
  const ep = S.run.episodes[S.ep];
  for (const q of S.particles) {
    q.age += dt; q.vy -= 2.5 * dt; q.vx *= 1 - 1.2 * dt; q.vy *= 1 - 1.2 * dt;
    q.x += q.vx * dt; q.y += q.vy * dt;
    const g = terrainY(ep, q.x);
    if (q.y < g) { q.y = g; q.vy = Math.abs(q.vy) * 0.25; q.vx *= 0.6; }
  }
  S.particles = S.particles.filter((q) => q.age < q.life);
}

function drawParticles(c) {
  c.globalCompositeOperation = "lighter";
  for (const q of S.particles) {
    const k = q.age / q.life, r = q.size * (1 + k * 1.8);
    if (q.smoke) c.fillStyle = `rgba(150,150,165,${0.35 * (1 - k)})`;
    else if (k < 0.3) c.fillStyle = `rgba(255,235,170,${0.9 * (1 - k)})`;
    else c.fillStyle = `rgba(255,${Math.round(150 - 90 * k)},${Math.round(60 - 40 * k)},${0.75 * (1 - k)})`;
    c.beginPath(); c.arc(q.x, q.y, r, 0, 6.283); c.fill();
  }
  c.globalCompositeOperation = "source-over";
}

// ---------- scene ----------
function drawFlames(c, f, tsec) {
  c.globalCompositeOperation = "lighter";
  for (const [x, y, dx, dy, p, kind] of f.f) {
    if (p <= 0) continue;
    const n = Math.hypot(dx, dy) || 1, ux = dx / n, uy = dy / n, nx = -uy, ny = ux;
    const flick = 0.82 + 0.36 * Math.random();
    const L = (kind === 0 ? 1.0 : 0.5) * p * flick, w = kind === 0 ? 0.14 : 0.075;
    const glow = c.createRadialGradient(x, y, 0, x, y, (kind === 0 ? 0.9 : 0.45) * p);
    glow.addColorStop(0, "rgba(255,170,80,0.38)"); glow.addColorStop(1, "rgba(255,120,40,0)");
    c.fillStyle = glow; c.beginPath(); c.arc(x, y, (kind === 0 ? 0.9 : 0.45) * p, 0, 6.283); c.fill();
    const tx = x + ux * L, ty = y + uy * L;
    const gr = c.createLinearGradient(x, y, tx, ty);
    gr.addColorStop(0, "rgba(255,255,240,0.95)"); gr.addColorStop(0.25, "rgba(255,215,120,0.9)");
    gr.addColorStop(0.6, "rgba(255,120,40,0.55)"); gr.addColorStop(1, "rgba(255,60,20,0)");
    c.fillStyle = gr; c.beginPath();
    c.moveTo(x + nx * w, y + ny * w);
    c.quadraticCurveTo(x + ux * L * 0.35 + nx * w * 1.2, y + uy * L * 0.35 + ny * w * 1.2, tx, ty);
    c.quadraticCurveTo(x + ux * L * 0.35 - nx * w * 1.2, y + uy * L * 0.35 - ny * w * 1.2, x - nx * w, y - ny * w);
    c.closePath(); c.fill();
  }
  c.globalCompositeOperation = "source-over";
}

function drawLander(c, f, px) {
  const g = S.run.geometry, sc = g.scale, [hw, hh] = [g.leg_half[0] / sc, g.leg_half[1] / sc];
  for (const [x, y, a, contact] of f.g) {
    c.save(); c.translate(x, y); c.rotate(a);
    if (contact) { c.shadowColor = "rgba(80,230,200,0.9)"; c.shadowBlur = 12 * S.dpr; }
    c.fillStyle = contact ? "#7fe8d4" : "#6f5fd0"; c.fillRect(-hw, -hh, 2 * hw, 2 * hh);
    c.shadowBlur = 0; c.strokeStyle = "rgba(220,212,255,0.75)"; c.lineWidth = px; c.strokeRect(-hw, -hh, 2 * hw, 2 * hh);
    c.restore();
  }
  const [lx, ly, la] = f.l;
  c.save(); c.translate(lx, ly); c.rotate(la);
  c.beginPath(); g.lander_poly.forEach(([x, y], i) => (i ? c.lineTo(x / sc, y / sc) : c.moveTo(x / sc, y / sc))); c.closePath();
  const body = c.createLinearGradient(0, 17 / sc, 0, -10 / sc);
  body.addColorStop(0, "#a593ff"); body.addColorStop(0.55, "#6656c8"); body.addColorStop(1, "#3a3474");
  c.fillStyle = body; c.fill();
  c.strokeStyle = "rgba(230,224,255,0.95)"; c.lineWidth = 1.3 * px; c.stroke();
  c.beginPath(); c.arc(0, 6 / sc, 4.2 / sc, 0, 6.283);
  const win = c.createRadialGradient(-1.2 / sc, 7.5 / sc, 0, 0, 6 / sc, 4.2 / sc);
  win.addColorStop(0, "#e6f4ff"); win.addColorStop(1, "#5fa4e8");
  c.fillStyle = win; c.fill(); c.strokeStyle = "rgba(20,20,40,0.6)"; c.lineWidth = px; c.stroke();
  c.restore();
}

function draw(tsec) {
  const run = S.run, ep = run.episodes[S.ep];
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  if (!ep) { ctx.fillStyle = "#05060a"; ctx.fillRect(0, 0, canvas.width, canvas.height); return; }
  ctx.drawImage(staticLayer, 0, 0);
  worldTransform(ctx);
  const px = S.dpr / S.s, f = ep.frames[S.frame];
  for (const st of S.twinkle || []) {
    ctx.fillStyle = `rgba(230,235,255,${st.a * (0.55 + 0.45 * Math.sin(tsec * st.sp + st.ph))})`;
    ctx.beginPath(); ctx.arc(st.x, st.y, st.r, 0, 6.283); ctx.fill();
  }
  // ground shadow under the lander
  const gy = terrainY(ep, f.l[0]), alt = Math.max(0, f.l[1] - gy);
  const sa = clamp(0.4 - alt * 0.05, 0, 0.4);
  if (sa > 0) { ctx.fillStyle = `rgba(0,0,0,${sa})`; ctx.beginPath(); ctx.ellipse(f.l[0], gy, 0.75 + alt * 0.06, 0.09, 0, 0, 6.283); ctx.fill(); }
  drawParticles(ctx);
  drawFlames(ctx, f, tsec);
  drawLander(ctx, f, px);
}

// ---------- HUD & lists ----------
function updateHUD() {
  const run = S.run, ep = run.episodes[S.ep]; if (!ep) return;
  const last = ep.frames.length - 1;
  $("hud-ep").textContent = `Episode ${ep.episode}`;
  $("hud-step").textContent = `step ${S.frame} / ${last}`;
  $("hud-ret").textContent = fmt(ep.cum[S.frame]);
  const o = $("hud-outcome");
  if (S.frame >= last) {
    const tone = OUTCOME_TONE[ep.outcome] || "muted";
    const html = `<span class="badge ${tone}">${ICON[ep.outcome] || ""}<span class="lbl">${OUTCOME_LABEL[ep.outcome] || ep.outcome}</span></span>`;
    if (o.innerHTML !== html) o.innerHTML = html;
    o.classList.add("show");
  } else o.classList.remove("show");
  $("scrub").max = last; $("scrub").value = S.frame;
}

// task parameters worth showing for an episode: weakened engines (warning) and which pad is good
function paramTags(run, ep) {
  return Object.entries(ep.params).filter(([, v]) => typeof v !== "number" || v < 1)
    .map(([k, v]) => ({ text: paramText(run, k, v), tone: typeof v === "number" ? "warning" : "muted" }));
}

function renderEpisodeMeta() {
  const run = S.run, ep = run.episodes[S.ep];
  $("hud-scales").innerHTML = ep ? paramTags(run, ep).map((t) =>
    `<span class="badge ${t.tone}">${t.tone === "warning" ? ICON.warn : ICON.flag}<span class="lbl">${t.text}</span></span>`).join("") : "";
  document.querySelectorAll(".ep").forEach((el, i) => el.setAttribute("aria-current", i === S.ep ? "true" : "false"));
  const cur = document.querySelector('.ep[aria-current="true"]');
  if (cur) cur.scrollIntoView({ block: "nearest" });
  drawChartMarkers();
}

function renderList() {
  const run = S.run;
  $("eplist-title").textContent = `Recorded episodes (${run.episodes.length})`;
  $("eplist").innerHTML = run.episodes.map((ep, i) => {
    const tone = OUTCOME_TONE[ep.outcome] || "muted", tags = paramTags(run, ep);
    return `<button class="ep" data-i="${i}">
      <span class="name num">Ep ${ep.episode}</span><span class="ret num">${fmt(ep.return)}</span>
      <span class="meta"><span class="icon-${tone}">${ICON[ep.outcome] || ""}</span>${OUTCOME_LABEL[ep.outcome] || ep.outcome}
      ${tags.map((t) => `<span class="${t.tone === "warning" ? "tag" : ""}">· ${t.text}</span>`).join("")}</span></button>`;
  }).join("") || '<div class="empty">No episodes recorded yet.</div>';
  document.querySelectorAll(".ep").forEach((el) => el.addEventListener("click", () => gotoEpisode(+el.dataset.i)));
}

function gotoEpisode(i, keepPlaying) {
  const n = S.run.episodes.length; if (!n) return;
  S.ep = clamp(i, 0, n - 1); S.frame = 0; S.acc = 0; S.hold = 0; S.particles = [];
  if (!keepPlaying) setPlaying(true);
  buildStatic(); renderEpisodeMeta(); updateHUD();
}

function setPlaying(p) {
  const ep = S.run.episodes[S.ep];
  if (p && ep && S.frame >= ep.frames.length - 1) { S.frame = 0; S.particles = []; S.hold = 0; }
  S.playing = p; $("play").innerHTML = p ? ICON.pause : ICON.play;
}

// ---------- chart ----------
const NS = "http://www.w3.org/2000/svg";
const chart = { };
function el(tag, attrs, parent) { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; }
function niceStep(range, target) { const raw = range / target, p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p;
  return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p; }

function renderChart() {
  const run = S.run, svg = $("chart-svg"); svg.innerHTML = "";
  const R = run.returns, N = R.length;
  if (!N) { $("chart-desc").textContent = "No returns logged yet."; return; }
  const box = svg.getBoundingClientRect(), Wd = box.width, Hd = box.height;
  svg.setAttribute("viewBox", `0 0 ${Wd} ${Hd}`);
  const m = { l: 48, r: 72, t: 22, b: 26 };
  const win = 20, mean = R.map((_, i) => { const a = Math.max(0, i - win + 1); let s = 0; for (let j = a; j <= i; j++) s += R[j]; return s / (i - a + 1); });
  const sorted = [...R].sort((a, b) => a - b), q = (p) => sorted[Math.floor(p * (N - 1))];
  let lo = Math.min(q(0.02), ...mean, 0), hi = Math.max(q(0.995), ...mean, 0);
  const step = niceStep(hi - lo || 1, 5); lo = Math.floor(lo / step) * step; hi = Math.ceil(hi / step) * step;
  const X = (e) => m.l + (Wd - m.l - m.r) * (N > 1 ? e / (N - 1) : 0.5);
  const Y = (v) => m.t + (Hd - m.t - m.b) * (1 - (clamp(v, lo, hi) - lo) / (hi - lo));
  Object.assign(chart, { X, Y, N, mean, m, Wd, Hd });

  // switch bands
  for (const b of run.bands) {
    const x0 = X(b.start), x1 = X(Math.min(b.end, N - 1));
    const isScale = typeof b.value === "number";
    el("rect", { x: x0, y: m.t, width: Math.max(0, x1 - x0), height: Hd - m.t - m.b,
      fill: isScale ? css("--warning") : css("--ink-2"), "fill-opacity": isScale ? 0.08 : 0.06 }, svg);
    const t = el("text", { x: x0 + 6, y: m.t - 7, fill: css("--ink-2"), "font-size": 11.5 }, svg);
    t.textContent = paramText(run, b.key, b.value);
  }
  // grid + y axis
  for (let v = lo; v <= hi + 1e-9; v += step) {
    el("line", { x1: m.l, x2: Wd - m.r, y1: Y(v), y2: Y(v), stroke: v === 0 ? "#45453f" : css("--line"), "stroke-width": 1 }, svg);
    const t = el("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end", fill: css("--muted"), "font-size": 11 }, svg);
    t.textContent = Math.round(v);
  }
  if (200 >= lo && 200 <= hi) {
    el("line", { x1: m.l, x2: Wd - m.r, y1: Y(200), y2: Y(200), stroke: css("--muted"), "stroke-dasharray": "3 4", "stroke-width": 1 }, svg);
    const t = el("text", { x: Wd - m.r + 6, y: Y(200) + 4, fill: css("--muted"), "font-size": 11 }, svg); t.textContent = "solved (200)";
  }
  // x axis
  const xs = niceStep(N, 6);
  for (let e = 0; e < N; e += xs) {
    const t = el("text", { x: X(e), y: Hd - 8, "text-anchor": "middle", fill: css("--muted"), "font-size": 11 }, svg); t.textContent = e;
  }
  const xl = el("text", { x: Wd - m.r, y: Hd - 8, "text-anchor": "end", fill: css("--muted"), "font-size": 11 }, svg); xl.textContent = "episode";
  // lines
  const path = (arr) => arr.map((v, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join("");
  el("path", { d: path(R), fill: "none", stroke: css("--series"), "stroke-opacity": 0.28, "stroke-width": 1 }, svg);
  el("path", { d: path(mean), fill: "none", stroke: css("--series"), "stroke-width": 2, "stroke-linejoin": "round" }, svg);
  let ly = Y(mean[N - 1]) + 4;  // keep the line label clear of the "solved (200)" label
  if (200 >= lo && 200 <= hi && Math.abs(ly - (Y(200) + 4)) < 14) ly = Y(200) + 4 + (ly <= Y(200) + 4 ? -14 : 14);
  const lbl = el("text", { x: Wd - m.r + 6, y: ly, fill: css("--ink-2"), "font-size": 11.5 }, svg); lbl.textContent = "20-ep mean";
  // markers for recorded episodes
  chart.markers = el("g", {}, svg);
  chart.cross = el("line", { y1: m.t, y2: Hd - m.b, stroke: css("--ink-2"), "stroke-width": 1, opacity: 0 }, svg);
  chart.dot = el("circle", { r: 4, fill: css("--series"), stroke: css("--surface"), "stroke-width": 2, opacity: 0 }, svg);
  drawChartMarkers();
}

function drawChartMarkers() {
  if (!chart.markers) return;
  const run = S.run, g = chart.markers; g.innerHTML = "";
  run.episodes.forEach((ep, i) => {
    if (ep.episode >= chart.N) return;
    const cur = i === S.ep;
    el("circle", { cx: chart.X(ep.episode), cy: chart.Y(ep.return), r: cur ? 6 : 4,
      fill: cur ? css("--ink") : css("--panel"), stroke: cur ? css("--series") : css("--ink-2"), "stroke-width": 2 }, g);
  });
}

function nearestRecorded(e) {
  let best = -1, bd = Infinity;
  S.run.episodes.forEach((ep, i) => { const d = Math.abs(ep.episode - e); if (d < bd) { bd = d; best = i; } });
  return { i: best, d: bd };
}

function chartHover(ev) {
  if (!chart.X) return;
  const svg = $("chart-svg"), r = svg.getBoundingClientRect(), x = ev.clientX - r.left;
  const { m, Wd, N } = chart, R = S.run.returns;
  const e = clamp(Math.round((x - m.l) / (Wd - m.l - m.r) * (N - 1)), 0, N - 1);
  const near = nearestRecorded(e), snap = near.i >= 0 && Math.abs(chart.X(S.run.episodes[near.i].episode) - x) < 14;
  const shown = snap ? S.run.episodes[near.i].episode : e;
  chart.cross.setAttribute("x1", chart.X(shown)); chart.cross.setAttribute("x2", chart.X(shown)); chart.cross.setAttribute("opacity", 0.5);
  chart.dot.setAttribute("cx", chart.X(shown)); chart.dot.setAttribute("cy", chart.Y(R[shown])); chart.dot.setAttribute("opacity", 1);
  const tip = $("tooltip");
  tip.innerHTML = `<div><b class="num">Episode ${shown}</b></div>
    <div class="num">return ${fmt(R[shown])} <span class="t-muted">· 20-ep mean ${fmt(chart.mean[shown])}</span></div>
    ${snap ? '<div class="t-muted">recorded · click to watch</div>' : ""}`;
  tip.style.opacity = 1;
  const cr = $("chart").getBoundingClientRect(), tw = tip.offsetWidth;
  let left = ev.clientX - cr.left + 14; if (left + tw > cr.width - 4) left = ev.clientX - cr.left - tw - 14;
  tip.style.left = left + "px"; tip.style.top = Math.max(4, ev.clientY - cr.top - 60) + "px";
  svg.style.cursor = snap ? "pointer" : "default";
  chart.snap = snap ? near.i : -1;
}

// ---------- wiring ----------
function selectRun(k) {
  S.run = RUNS[k]; S.ep = 0;
  document.querySelectorAll(".chip").forEach((c, i) => c.setAttribute("aria-pressed", i === k ? "true" : "false"));
  const run = S.run;
  $("subtitle").textContent = `${run.run} · ${run.env}.py · every ${run.every}th training episode (actions sampled from the policy)`;
  renderList(); resize(); renderChart(); gotoEpisode(0);
}

$("runs").innerHTML = RUNS.length > 1 ? RUNS.map((r, i) => `<button class="chip" data-k="${i}">${r.run}</button>`).join("") : "";
document.querySelectorAll(".chip").forEach((c) => c.addEventListener("click", () => selectRun(+c.dataset.k)));
$("speeds").innerHTML = SPEEDS.map((s) => `<button data-s="${s}" aria-pressed="${s === 1}">${s}×</button>`).join("");
$("speeds").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
  S.speed = +b.dataset.s; $("speeds").querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", x === b ? "true" : "false"));
}));
$("play").addEventListener("click", () => setPlaying(!S.playing));
$("prev").addEventListener("click", () => gotoEpisode(S.ep - 1));
$("next").addEventListener("click", () => gotoEpisode(S.ep + 1));
$("auto").addEventListener("change", (e) => (S.auto = e.target.checked));
$("scrub").addEventListener("input", (e) => { S.frame = +e.target.value; S.particles = []; S.hold = 0; updateHUD(); });
$("chart-svg").addEventListener("mousemove", chartHover);
$("chart-svg").addEventListener("mouseleave", () => { $("tooltip").style.opacity = 0; if (chart.cross) { chart.cross.setAttribute("opacity", 0); chart.dot.setAttribute("opacity", 0); } });
$("chart-svg").addEventListener("click", () => { if (chart.snap >= 0) gotoEpisode(chart.snap); });
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" && e.target.type !== "range" && e.target.type !== "checkbox") return;
  const ep = S.run.episodes[S.ep];
  if (e.key === " ") { e.preventDefault(); setPlaying(!S.playing); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); gotoEpisode(S.ep - 1); }
  else if (e.key === "ArrowRight") { e.preventDefault(); gotoEpisode(S.ep + 1); }
  else if ((e.key === "," || e.key === ".") && ep) {
    setPlaying(false); S.particles = [];
    S.frame = clamp(S.frame + (e.key === "." ? 1 : -1), 0, ep.frames.length - 1); updateHUD();
  }
});
new ResizeObserver(() => { resize(); renderChart(); }).observe($("viewport"));

let last = performance.now();
function tick(now) {
  const dt = Math.min(0.05, (now - last) / 1000); last = now;
  const ep = S.run.episodes[S.ep];
  if (ep && S.playing) {
    const lastF = ep.frames.length - 1;
    if (S.frame < lastF) {
      S.acc += dt * S.run.geometry.fps * S.speed;
      while (S.acc >= 1 && S.frame < lastF) { S.acc -= 1; S.frame++; spawn(ep, S.frame); }
    } else {
      S.hold += dt;
      if (S.hold > 1.4) {
        if (S.auto && S.ep < S.run.episodes.length - 1) gotoEpisode(S.ep + 1, true);
        else setPlaying(false);
      }
    }
  }
  updateParticles(S.playing ? dt * S.speed : 0);
  draw(now / 1000); updateHUD();
  requestAnimationFrame(tick);
}

selectRun(0);
setPlaying(true);
requestAnimationFrame(tick);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
