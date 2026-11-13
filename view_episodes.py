"""Build an HTML replay viewer for the recorded training episodes and open it in the browser.

three_booster/run.py and four_booster/run.py write every 100th episode to <suite>/outputs/<task>_<model>/
episodes.json. This bundles those recordings into one self-contained page: figures/episodes.html.

    python view_episodes.py                         # every run found
    python view_episodes.py four_booster/outputs/2landing_mlp/episodes.json
    python view_episodes.py --no-open               # just write the file

Keys in the viewer: space = play/pause, left/right = previous/next episode, , and . = step one frame.
Open episodes.html#run=1&ep=40&frame=200 to start at a run (0-based), recorded episode (0-based) and frame, paused.
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


# Theme follows the lunar lander reel: hue = how far into training (violet -> gold), rose = crash and nothing
# else, near-white = the pad and a touchdown, slate = the moon.
TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lander Replays</title>
<style>
:root {
  color-scheme: dark;
  --bg: #07080c;
  --panel: #0b0d13;
  --panel-2: #12151d;
  --line: #1c212d;
  --line-2: #2c3343;
  --ink: #f2f5fa;
  --ink-2: #8c97ac;
  --muted: #5c6577;
  --pad: #d6ecf8;
  --crash: #f43f5e;
  --radius: 6px;
  --mono: ui-monospace, "SF Mono", "JetBrains Mono", Menlo, Consolas, monospace;
  --sans: ui-sans-serif, -apple-system, "Helvetica Neue", Arial, sans-serif;
}
* { box-sizing: border-box; }
html, body { margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.45 var(--sans); }
button { font: inherit; color: inherit; }
.num { font-variant-numeric: tabular-nums; }
.caps { font-family: var(--mono); text-transform: uppercase; letter-spacing: 0.06em; }

.page { max-width: 1320px; margin: 0 auto; padding: 24px 16px 40px; }
header { display: flex; flex-wrap: wrap; align-items: end; justify-content: space-between; gap: 12px; margin-bottom: 18px; }
h1 { font: 800 clamp(22px, 3.4vw, 32px)/1.1 var(--sans); text-transform: uppercase; letter-spacing: 0.01em; margin: 0; }
.sub { color: var(--ink-2); margin-top: 6px; }
.runs { display: flex; gap: 6px; flex-wrap: wrap; }
.chip { border: 1px solid var(--line); background: transparent; border-radius: var(--radius); padding: 6px 12px; cursor: pointer;
  font: 12px var(--mono); text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }
.chip:hover { color: var(--ink-2); border-color: var(--line-2); }
.chip[aria-pressed="true"] { border-color: var(--ink); color: var(--ink); }

.layout { display: grid; grid-template-columns: minmax(0, 1fr) 280px; gap: 16px; }
@media (max-width: 900px) { .layout { grid-template-columns: 1fr; } }

.stage { background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden; }
.viewport { position: relative; background: var(--bg); }
.viewport canvas { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }
.hud { position: absolute; inset: 0; pointer-events: none; padding: 14px 16px; display: flex; flex-direction: column; justify-content: space-between; }
.hud-row { display: flex; justify-content: space-between; align-items: start; gap: 12px; }
.hud .big { font: 26px/1.15 var(--mono); font-variant-numeric: tabular-nums; }
.hud .small { color: var(--muted); font: 11.5px var(--mono); text-transform: uppercase; letter-spacing: 0.06em; }
.hud .right { text-align: right; }
.badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 9px; border-radius: var(--radius);
  font: 11.5px var(--mono); text-transform: uppercase; letter-spacing: 0.06em;
  background: rgba(7, 8, 12, .78); border: 1px solid var(--line-2); }
.badge svg { width: 13px; height: 13px; flex: none; }
.badge.good { color: var(--pad); } .badge.critical { color: var(--crash); }
.badge.warning { color: var(--ink); } .badge.muted { color: var(--ink-2); }
.badge .lbl { color: inherit; }
.badge.warning .lbl { color: var(--ink); } .badge.warning svg { color: var(--ink-2); }
.hud-bottom { display: flex; gap: 8px; flex-wrap: wrap; }
.outcome { opacity: 0; transform: translateY(-4px); transition: opacity .25s, transform .25s; margin-top: 6px; }
.outcome.show { opacity: 1; transform: none; }

.transport { display: flex; align-items: center; gap: 10px; padding: 10px 12px; border-top: 1px solid var(--line); flex-wrap: wrap; }
.iconbtn { width: 36px; height: 36px; border-radius: var(--radius); border: 1px solid var(--line); background: transparent;
  color: var(--ink-2); display: grid; place-items: center; cursor: pointer; }
.iconbtn:hover { border-color: var(--line-2); color: var(--ink); }
.iconbtn svg { width: 16px; height: 16px; }
.iconbtn.primary { background: var(--ink); border-color: var(--ink); color: var(--bg); width: 42px; }
.seg { display: inline-flex; border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden; }
.seg button { border: 0; background: transparent; padding: 7px 10px; cursor: pointer; color: var(--muted);
  font: 12px var(--mono); letter-spacing: 0.04em; }
.seg button + button { border-left: 1px solid var(--line); }
.seg button[aria-pressed="true"] { background: var(--panel-2); color: var(--ink); }
.scrub { flex: 1 1 200px; display: flex; align-items: center; gap: 10px; min-width: 160px; }
input[type=range] { flex: 1; appearance: none; -webkit-appearance: none; background: transparent; height: 16px; margin: 0; }
input[type=range]::-webkit-slider-runnable-track { height: 1px; background: var(--line-2); }
input[type=range]::-moz-range-track { height: 1px; background: var(--line-2); }
input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 10px; height: 10px; border-radius: 50%; background: var(--ink); margin-top: -4.5px; }
input[type=range]::-moz-range-thumb { width: 10px; height: 10px; border: 0; border-radius: 50%; background: var(--ink); }
.toggle { display: inline-flex; align-items: center; gap: 6px; color: var(--ink-2); cursor: pointer; user-select: none;
  font: 12px var(--mono); text-transform: uppercase; letter-spacing: 0.06em; }
.toggle input { accent-color: var(--ink); }

aside { background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius); display: flex; flex-direction: column; min-height: 0; }
aside h2, .card h2 { font: 12px var(--mono); text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin: 0; padding: 12px 14px 8px; }
.eplist { overflow-y: auto; max-height: 560px; padding: 0 8px 8px; }
@media (max-width: 900px) { .eplist { max-height: 300px; } }
.ep { width: 100%; display: grid; grid-template-columns: 1fr auto; align-items: center; gap: 2px 8px; text-align: left;
  border: 0; border-left: 2px solid transparent; background: transparent; border-radius: 0 var(--radius) var(--radius) 0; padding: 8px 10px; cursor: pointer; }
.ep:hover { background: var(--panel-2); }
.ep[aria-current="true"] { background: var(--panel-2); border-left-color: var(--c); }
.ep .name { font: 13px var(--mono); letter-spacing: 0.02em; }
.ep .ret { text-align: right; font: 13px var(--mono); color: var(--c); }
.ep .meta { grid-column: 1 / -1; display: flex; flex-wrap: wrap; gap: 2px 6px; align-items: center; color: var(--muted); font-size: 12px; }
.ep .meta svg { width: 12px; height: 12px; }
.ep .tag { color: var(--ink-2); }
.ep .icon-good { color: var(--pad); } .ep .icon-critical { color: var(--crash); }
.ep .icon-warning { color: var(--ink-2); } .ep .icon-muted { color: var(--muted); }

.card { background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius); margin-top: 16px; position: relative; }
.card .desc { color: var(--muted); font-size: 12.5px; padding: 0 14px; margin-top: -4px; }
.chart { position: relative; padding: 4px 6px 8px; }
.chart svg { display: block; width: 100%; height: 240px; font-family: var(--mono); }
.tooltip { position: absolute; pointer-events: none; background: var(--bg); border: 1px solid var(--line-2); border-radius: var(--radius); padding: 8px 10px;
  font: 12px var(--mono); color: var(--ink); white-space: nowrap; opacity: 0; transition: opacity .1s; }
.tooltip .t-muted { color: var(--ink-2); }
.empty { padding: 40px; text-align: center; color: var(--muted); }
kbd { font: 11px var(--mono); border: 1px solid var(--line-2); border-bottom-width: 2px; border-radius: 4px; padding: 0 4px; color: var(--ink-2); }
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
              <div class="small">Episode</div>
              <div class="big num" id="hud-ep">0</div>
              <div class="small num" id="hud-step">step 0 / 0</div>
              <div class="small num" id="hud-each"></div>
            </div>
            <div class="right">
              <div class="small" id="hud-ret-label">This attempt</div>
              <div class="big num" id="hud-ret">0</div>
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
    <div class="desc" id="chart-desc">Faint line: every training episode. Bold line: 20-episode mean. Colour: how far into training. Dots: recorded episodes, click to watch.</div>
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
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const fmt = (x) => (x < 0 ? "−" : "+") + Math.abs(x).toFixed(1);

// ---------- colour (from the reel's palette.py) ----------
const C = { bg: [7, 8, 12], white: [242, 245, 250], gray: [140, 151, 172], dim: [92, 101, 119], ground: [24, 30, 42],
  ridge: [104, 140, 184], pad: [214, 236, 248], crash: [244, 63, 94] };
const RAMP = [[0, [129, 60, 245]], [0.18, [59, 130, 246]], [0.36, [34, 211, 238]], [0.54, [52, 211, 153]],
  [0.72, [250, 204, 21]], [0.88, [249, 115, 22]], [1, [244, 63, 94]]];
const PROGRESS_END = 0.8;  // training progress stops at gold: rose is reserved for crashes
function ramp(u) {
  u = clamp(u, 0, 1);
  for (let i = 1; i < RAMP.length; i++) {
    const [u0, c0] = RAMP[i - 1], [u1, c1] = RAMP[i];
    if (u <= u1) { const t = (u - u0) / (u1 - u0); return c0.map((v, k) => Math.round(v + (c1[k] - v) * t)); }
  }
  return RAMP[RAMP.length - 1][1];
}
const progress = (run, e) => ramp(PROGRESS_END * e / Math.max(1, run.returns.length - 1));
const rgba = (c, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
const shade = (c, f) => c.map((v) => Math.round(v * f));

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
ICON["landed on good pad"] = ICON.landed; ICON["landed on targets"] = ICON.landed; ICON["landed on bad pad"] = ICON.warn; ICON["landed off pad"] = ICON.dash;
// rose only for a crash; near-white for a landing; everything else gray
const OUTCOME_TONE = { landed: "good", "landed on good pad": "good", "landed on targets": "good", "landed on bad pad": "muted", "landed off pad": "muted",
  crashed: "critical", "out of bounds": "muted", "timed out": "muted" };
const OUTCOME_LABEL = { landed: "Landed", "landed on good pad": "Landed on good pad", "landed on targets": "Landed on targets", "landed on bad pad": "Landed on bad pad",
  "landed off pad": "Landed off the pads", crashed: "Crashed", "out of bounds": "Flew off-screen", "timed out": "Timed out" };
const SPEEDS = [0.5, 1, 2, 4];
const HEADROOM = 2.0;  // world units of sky above the env's viewport, where landers start

// ---------- state ----------
const S = { run: null, ep: 0, frame: 0, playing: true, speed: 1, acc: 0, hold: 0, auto: true, dpr: 1, s: 1 };

// ---------- run setup ----------
function prepareRun(run) {
  const g = run.geometry;
  run.W = g.viewport[0] / g.scale; run.H = g.viewport[1] / g.scale + HEADROOM;
  run.param_log = run.param_log || run.scale_log || [];  // older recordings used scale_*
  run.param_labels = run.param_labels || run.scale_labels || {};
  for (const ep of run.episodes) {
    for (const f of ep.frames) {  // one list entry per lander; single-lander recordings have one
      if (f.L) { f.F = f.F.flat(); f.R = f.r; } else { f.L = [f.l]; f.G = [f.g]; f.F = f.f; f.R = [f.r]; }
    }
    ep.n = ep.frames[0].L.length;
    const acc = new Array(ep.n).fill(0);
    ep.cums = ep.frames.map((f) => f.R.map((v, k) => (acc[k] += v)));
    ep.cum = ep.cums.map((a) => a.reduce((x, y) => x + y, 0) / a.length);
    ep.params = ep.params || ep.scales || {};
    if (!ep.pads) ep.pads = [[...ep.helipad, "pad", 1]];
    ep.color = progress(run, ep.episode);
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
RUNS.forEach((run) => { run.multi = run.episodes.some((ep) => ep.n > 1); });
const label = (run, k) => run.param_labels[k] || k.replace(/_/g, " ");
const paramText = (run, k, v) => typeof v === "number" ? `${label(run, k)} at ${Math.round(v * 100)}%` : `${label(run, k)}: ${v}`;

// ---------- scene ----------
const canvas = $("canvas"), ctx = canvas.getContext("2d");
const staticLayer = document.createElement("canvas");

function resize() {
  const vp = $("viewport"), ar = `${S.run.W} / ${S.run.H}`;
  if (vp.style.aspectRatio !== ar) vp.style.aspectRatio = ar;
  const r = vp.getBoundingClientRect();
  S.dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(r.width * S.dpr); canvas.height = Math.round(r.height * S.dpr);
  S.s = canvas.width / S.run.W;
  buildStatic();
}
const worldTransform = (c) => c.setTransform(S.s, 0, 0, -S.s, 0, S.run.H * S.s);

function trail(c, ep, k, upto, color, alpha, px) {
  const fr = ep.frames, n = Math.min(upto, fr.length - 1);
  if (n < 1) return;
  c.beginPath(); c.moveTo(fr[0].L[k][0], fr[0].L[k][1]);
  for (let i = 1; i <= n; i++) c.lineTo(fr[i].L[k][0], fr[i].L[k][1]);
  c.strokeStyle = rgba(color, alpha); c.lineWidth = 1.5 * px; c.lineJoin = "round"; c.stroke();
}

// sky, the paths of every earlier recorded episode, the moon and the pads
function buildStatic() {
  const run = S.run, ep = run.episodes[S.ep];
  staticLayer.width = canvas.width; staticLayer.height = canvas.height;
  const c = staticLayer.getContext("2d"), px = S.dpr / S.s;
  c.fillStyle = rgba(C.bg); c.fillRect(0, 0, staticLayer.width, staticLayer.height);
  if (!ep) return;
  worldTransform(c);
  run.episodes.slice(0, S.ep).forEach((p, i) => {
    const a = 0.10 + 0.22 * (i + 1) / S.ep;  // older paths fainter
    for (let k = 0; k < p.n; k++) {
      trail(c, p, k, Infinity, p.color, a, px);
      const f = p.frames[p.frames.length - 1].L[k], sq = 2.5 * px;
      c.fillStyle = rgba(p.outcome === "crashed" ? C.crash : p.color, Math.min(1, a * 2));
      c.fillRect(f[0] - sq, f[1] - sq, 2 * sq, 2 * sq);
    }
  });
  const t = ep.terrain;
  c.beginPath(); c.moveTo(0, -1); for (const [x, y] of t) c.lineTo(x, y); c.lineTo(run.W, -1); c.closePath();
  c.fillStyle = rgba(C.ground); c.fill();
  c.beginPath(); t.forEach(([x, y], i) => (i ? c.lineTo(x, y) : c.moveTo(x, y)));
  c.strokeStyle = rgba(C.ridge); c.lineWidth = 1.5 * px; c.stroke();
  // pads: near-white; on the two-pad task the bad pad is dimmed
  const sc = run.geometry.scale, labels = [];
  for (const [x1, x2, y, , good] of ep.pads) {
    const col = !ep.two_pads || good ? C.pad : C.dim;
    c.strokeStyle = rgba(col); c.lineWidth = 2.5 * px;
    c.beginPath(); c.moveTo(x1, y); c.lineTo(x2, y); c.stroke();
    for (const x of [x1, x2]) {
      c.lineWidth = 2 * px; c.beginPath(); c.moveTo(x, y); c.lineTo(x, y + 34 / sc); c.stroke();
      c.fillStyle = rgba(col); c.beginPath();
      c.moveTo(x, y + 34 / sc); c.lineTo(x, y + 25 / sc); c.lineTo(x + 18 / sc, y + 29.5 / sc); c.closePath(); c.fill();
    }
    if (ep.two_pads) labels.push({ x: (x1 + x2) / 2, y: y + 34 / sc, text: good ? "GOOD" : "BAD", col });
  }
  for (const x of ep.targets || []) {  // each lander's target spot on the shared pad
    c.strokeStyle = rgba(C.pad, 0.8); c.lineWidth = 2 * px;
    c.beginPath(); c.moveTo(x, ep.pads[0][2]); c.lineTo(x, ep.pads[0][2] + 12 / sc); c.stroke();
  }
  c.setTransform(1, 0, 0, 1, 0, 0);  // pad labels in screen space so the text isn't mirrored
  c.font = `${11 * S.dpr}px ${css("--mono")}`;
  c.textAlign = "center"; c.textBaseline = "bottom";
  for (const l of labels) { c.fillStyle = rgba(l.col); c.fillText(l.text, l.x * S.s, (run.H - l.y) * S.s - 4 * S.dpr); }
}

function drawFlames(c, f, color, px) {
  c.globalCompositeOperation = "lighter"; c.lineCap = "round";
  for (const [x, y, dx, dy, p, kind] of f.F) {
    const n = Math.hypot(dx, dy) || 1, ux = dx / n, uy = dy / n;
    if (p <= 0) {  // fired, but the engine is broken: a dim stub
      c.strokeStyle = rgba(C.dim, 0.8); c.lineWidth = 1.5 * px;
      c.beginPath(); c.moveTo(x, y); c.lineTo(x + ux * 0.12, y + uy * 0.12); c.stroke();
      continue;
    }
    const L = (kind === 0 ? 0.75 : 0.38) * p * (0.85 + 0.3 * Math.random());
    const g = c.createLinearGradient(x, y, x + ux * L, y + uy * L);
    g.addColorStop(0, rgba(C.white, 0.95)); g.addColorStop(0.35, rgba(color, 0.8)); g.addColorStop(1, rgba(color, 0));
    c.strokeStyle = g; c.lineWidth = (kind === 0 ? 4 : 2.5) * px;
    c.beginPath(); c.moveTo(x, y); c.lineTo(x + ux * L, y + uy * L); c.stroke();
  }
  c.globalCompositeOperation = "source-over"; c.lineCap = "butt";
}

function drawLander(c, f, k, color, px) {
  const g = S.run.geometry, sc = g.scale, hw = g.leg_half[0] / sc, hh = g.leg_half[1] / sc;
  for (const [x, y, a, contact] of f.G[k]) {
    c.save(); c.translate(x, y); c.rotate(a);
    c.fillStyle = rgba(color); c.fillRect(-hw, -hh, 2 * hw, 2 * hh);
    c.restore();
    if (contact) { c.fillStyle = rgba(C.pad); c.beginPath(); c.arc(x, y - hh, 3.5 * px, 0, 6.283); c.fill(); }
  }
  const [lx, ly, la] = f.L[k];
  c.save(); c.translate(lx, ly); c.rotate(la);
  c.beginPath(); g.lander_poly.forEach(([x, y], i) => (i ? c.lineTo(x / sc, y / sc) : c.moveTo(x / sc, y / sc))); c.closePath();
  c.fillStyle = rgba(shade(color, 0.45)); c.fill();
  c.strokeStyle = rgba(color); c.lineWidth = 1.5 * px; c.stroke();
  c.restore();
}

function draw() {
  const run = S.run, ep = run.episodes[S.ep];
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.drawImage(staticLayer, 0, 0);
  if (!ep) return;
  worldTransform(ctx);
  const px = S.dpr / S.s, f = ep.frames[S.frame], end = S.frame >= ep.frames.length - 1;
  const color = end && ep.outcome === "crashed" ? C.crash : ep.color;
  for (let k = 0; k < ep.n; k++) trail(ctx, ep, k, S.frame, ep.color, 0.75, px);
  const rope = run.geometry.tether_length;
  if (rope && ep.n > 1) {  // the tether between the two body centres: white when taut
    const [a, b] = f.L, taut = Math.hypot(a[0] - b[0], a[1] - b[1]) > rope - 0.03;
    ctx.strokeStyle = taut ? rgba(C.white, 0.9) : rgba(C.dim); ctx.lineWidth = 1.5 * px;
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
  }
  drawFlames(ctx, f, ep.color, px);
  for (let k = 0; k < ep.n; k++) drawLander(ctx, f, k, color, px);
}

// ---------- HUD & lists ----------
function updateHUD() {
  const run = S.run, ep = run.episodes[S.ep]; if (!ep) return;
  const last = ep.frames.length - 1, end = S.frame >= last;
  $("hud-ep").textContent = ep.episode.toLocaleString("en-US");
  $("hud-step").textContent = `step ${S.frame} / ${last}`;
  $("hud-each").textContent = ep.n > 1 ? ep.cums[S.frame].map((v, k) => `lander ${k + 1} ${fmt(v)}`).join(" · ") : "";
  $("hud-ret").textContent = fmt(ep.cum[S.frame]);
  $("hud-ret").style.color = rgba(end && ep.outcome === "crashed" ? C.crash : ep.color);
  const o = $("hud-outcome");
  if (end) {
    const tone = OUTCOME_TONE[ep.outcome] || "muted";
    const html = `<span class="badge ${tone}">${ICON[ep.outcome] || ""}<span class="lbl">${OUTCOME_LABEL[ep.outcome] || ep.outcome}</span></span>`;
    if (o.innerHTML !== html) o.innerHTML = html;
    o.classList.add("show");
  } else o.classList.remove("show");
  $("scrub").max = last; $("scrub").value = S.frame;
}

// task parameters worth showing for an episode: weakened engines and which pad is good
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
    return `<button class="ep" data-i="${i}" style="--c:${rgba(ep.color)}">
      <span class="name num">Ep ${ep.episode.toLocaleString("en-US")}</span><span class="ret num">${fmt(ep.return)}</span>
      <span class="meta"><span class="icon-${tone}">${ICON[ep.outcome] || ""}</span>${OUTCOME_LABEL[ep.outcome] || ep.outcome}
      ${tags.map((t) => `<span class="${t.tone === "warning" ? "tag" : ""}">· ${t.text}</span>`).join("")}</span></button>`;
  }).join("") || '<div class="empty">No episodes recorded yet.</div>';
  document.querySelectorAll(".ep").forEach((el) => el.addEventListener("click", () => gotoEpisode(+el.dataset.i)));
}

function gotoEpisode(i, keepPlaying) {
  const n = S.run.episodes.length; if (!n) return;
  S.ep = clamp(i, 0, n - 1); S.frame = 0; S.acc = 0; S.hold = 0;
  if (!keepPlaying) setPlaying(true);
  buildStatic(); renderEpisodeMeta(); updateHUD();
}

function setPlaying(p) {
  const ep = S.run.episodes[S.ep];
  if (p && ep && S.frame >= ep.frames.length - 1) { S.frame = 0; S.hold = 0; }
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
  const m = { l: 48, r: 96, t: 22, b: 26 };
  const win = 20, mean = R.map((_, i) => { const a = Math.max(0, i - win + 1); let s = 0; for (let j = a; j <= i; j++) s += R[j]; return s / (i - a + 1); });
  const sorted = [...R].sort((a, b) => a - b), q = (p) => sorted[Math.floor(p * (N - 1))];
  let lo = Math.min(q(0.02), ...mean, 0), hi = Math.max(q(0.995), ...mean, 0);
  const step = niceStep(hi - lo || 1, 5); lo = Math.floor(lo / step) * step; hi = Math.ceil(hi / step) * step;
  const X = (e) => m.l + (Wd - m.l - m.r) * (N > 1 ? e / (N - 1) : 0.5);
  const Y = (v) => m.t + (Hd - m.t - m.b) * (1 - (clamp(v, lo, hi) - lo) / (hi - lo));
  Object.assign(chart, { X, Y, N, mean, m, Wd, Hd });

  // horizontal gradient = training progress, so the curve has the same hue as the episodes it summarises
  const grad = el("linearGradient", { id: "prog", gradientUnits: "userSpaceOnUse", x1: X(0), x2: X(N - 1), y1: 0, y2: 0 }, el("defs", {}, svg));
  for (let k = 0; k <= 10; k++) el("stop", { offset: k / 10, "stop-color": rgba(ramp(PROGRESS_END * k / 10)) }, grad);

  // switch bands
  for (const b of run.bands) {
    const x0 = X(b.start), x1 = X(Math.min(b.end, N - 1));
    el("rect", { x: x0, y: m.t, width: Math.max(0, x1 - x0), height: Hd - m.t - m.b, fill: css("--ink"), "fill-opacity": 0.04 }, svg);
    const t = el("text", { x: x0 + 6, y: m.t - 7, fill: css("--ink-2"), "font-size": 11 }, svg);
    t.textContent = paramText(run, b.key, b.value).toUpperCase();
  }
  // grid + y axis
  for (let v = lo; v <= hi + 1e-9; v += step) {
    el("line", { x1: m.l, x2: Wd - m.r, y1: Y(v), y2: Y(v), stroke: v === 0 ? css("--line-2") : css("--line"), "stroke-width": 1 }, svg);
    const t = el("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end", fill: css("--muted"), "font-size": 11 }, svg);
    t.textContent = Math.round(v);
  }
  if (200 >= lo && 200 <= hi) {
    el("line", { x1: m.l, x2: Wd - m.r, y1: Y(200), y2: Y(200), stroke: css("--ink-2"), "stroke-dasharray": "5 5", "stroke-width": 1 }, svg);
    const t = el("text", { x: Wd - m.r + 6, y: Y(200) + 4, fill: css("--ink"), "font-size": 11 }, svg); t.textContent = "+200 SOLVED";
  }
  // x axis
  const xs = niceStep(N, 6);
  for (let e = 0; e < N; e += xs) {
    const t = el("text", { x: X(e), y: Hd - 8, "text-anchor": "middle", fill: css("--muted"), "font-size": 11 }, svg); t.textContent = e;
  }
  const xl = el("text", { x: Wd - m.r, y: Hd - 8, "text-anchor": "end", fill: css("--muted"), "font-size": 11 }, svg); xl.textContent = "EPISODE";
  // lines
  const path = (arr) => arr.map((v, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join("");
  el("path", { d: path(R), fill: "none", stroke: "url(#prog)", "stroke-opacity": 0.3, "stroke-width": 1 }, svg);
  el("path", { d: path(mean), fill: "none", stroke: "url(#prog)", "stroke-width": 2, "stroke-linejoin": "round" }, svg);
  let ly = Y(mean[N - 1]) + 4;  // keep the line label clear of the "+200" label
  if (200 >= lo && 200 <= hi && Math.abs(ly - (Y(200) + 4)) < 14) ly = Y(200) + 4 + (ly <= Y(200) + 4 ? -14 : 14);
  const lbl = el("text", { x: Wd - m.r + 6, y: ly, fill: css("--ink-2"), "font-size": 11 }, svg); lbl.textContent = "20-EP MEAN";
  // markers for recorded episodes
  chart.markers = el("g", {}, svg);
  chart.cross = el("line", { y1: m.t, y2: Hd - m.b, stroke: css("--ink-2"), "stroke-width": 1, opacity: 0 }, svg);
  chart.dot = el("circle", { r: 4, fill: css("--ink"), stroke: css("--bg"), "stroke-width": 2, opacity: 0 }, svg);
  drawChartMarkers();
}

function drawChartMarkers() {
  if (!chart.markers) return;
  const run = S.run, g = chart.markers; g.innerHTML = "";
  run.episodes.forEach((ep, i) => {
    if (ep.episode >= chart.N) return;
    const cur = i === S.ep;
    el("circle", { cx: chart.X(ep.episode), cy: chart.Y(ep.return), r: cur ? 6 : 3.5,
      fill: cur ? css("--ink") : css("--bg"), stroke: cur ? rgba(ep.color) : css("--muted"), "stroke-width": 2 }, g);
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
  tip.innerHTML = `<div><b class="num">EPISODE ${shown.toLocaleString("en-US")}</b></div>
    <div class="num">RETURN ${fmt(R[shown])} <span class="t-muted">· 20-EP MEAN ${fmt(chart.mean[shown])}</span></div>
    ${snap ? '<div class="t-muted">RECORDED · CLICK TO WATCH</div>' : ""}`;
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
  $("hud-ret-label").textContent = run.multi ? "This attempt (mean of landers)" : "This attempt";
  $("chart-desc").textContent = "Faint line: every training episode. Bold line: 20-episode mean. Colour: how far into training. "
    + "Dots: recorded episodes, click to watch." + (run.multi ? " Return: mean of the landers' returns." : "");
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
$("scrub").addEventListener("input", (e) => { S.frame = +e.target.value; S.hold = 0; updateHUD(); });
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
    setPlaying(false);
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
      while (S.acc >= 1 && S.frame < lastF) { S.acc -= 1; S.frame++; }
    } else {
      S.hold += dt;
      if (S.hold > 1.4) {
        if (S.auto && S.ep < S.run.episodes.length - 1) gotoEpisode(S.ep + 1, true);
        else setPlaying(false);
      }
    }
  }
  draw(); updateHUD();
  requestAnimationFrame(tick);
}

const hash = Object.fromEntries(new URLSearchParams(location.hash.slice(1)));
selectRun(clamp(+hash.run || 0, 0, RUNS.length - 1));
setPlaying(true);
if (hash.ep !== undefined) gotoEpisode(+hash.ep);
if (hash.frame !== undefined && S.run.episodes[S.ep]) {
  setPlaying(false); S.frame = clamp(+hash.frame, 0, S.run.episodes[S.ep].frames.length - 1); updateHUD();
}
requestAnimationFrame(tick);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
