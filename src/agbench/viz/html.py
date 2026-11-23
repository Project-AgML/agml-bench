"""Self-contained HTML report: one file, data inlined, charts drawn in the browser (no server, no network).

Tabs: overview (accuracy per task with 95% CI, table), agricultural tasks, datasets (filterable, sortable), runs
(benchmark, hash, judge, coverage, comparable).
"""

import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from agbench.viz.aggregate import AG, CLS_FORMATS, COLS, Entry
from agbench.viz.figures import MAX_MODELS


def _num(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(x, 2)


def _cell(c):
    return {"m": _num(c[0]), "lo": _num(c[1]), "hi": _num(c[2]), "n": c[3]}


def _run_info(meta: dict, d: str) -> dict:
    m = meta.get("method") or {}
    bm = meta.get("benchmark") if isinstance(meta.get("benchmark"), dict) and "name" in meta.get("benchmark", {}) else \
        (meta.get("benchmark") or {}).get("benchmark")
    cov = meta.get("coverage") or {}
    return {"dir": d, "model": m.get("model_id") or m.get("name"), "effort": m.get("effort"), "thinking": m.get("thinking"),
            "gen_seed": m.get("gen_seed"), "benchmark": (bm or {}).get("name"), "hash": (bm or {}).get("hash"),
            "variant": (bm or {}).get("variant"), "judge": (meta.get("judge") or {}).get("model"),
            "comparable": meta.get("comparable"), "why": meta.get("not_comparable") or [],
            "answered": cov.get("answered"), "items": cov.get("items"), "complete": cov.get("complete")}


def payload(entries: list[Entry], title: str) -> dict:
    models = [e for e in entries if e.kind == "model"]
    if len(models) > MAX_MODELS:
        raise ValueError(f"{len(models)} models: the report shows at most {MAX_MODELS} (pass fewer runs)")
    ds = sorted({k for e in entries for k in e.datasets})
    return {
        "title": title, "written": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "cols": [{"key": k, "label": h} for k, h in COLS], "ag": [{"key": a, "label": h} for a, h in AG],
        "formats": [{"key": f, "label": h} for f, h in CLS_FORMATS],
        "entries": [{"label": e.label, "kind": e.kind, "replicates": e.replicates,
                     "cells": {k: _cell(v) for k, v in e.cells.items()},
                     "ag": {f"{f}|{a}": _cell(v) for (f, a), v in e.ag.items()},
                     "fuzzy": {a: _num(v) for a, v in e.fuzzy.items()}, "itt_fmt": {q: _num(v) for q, v in e.itt_fmt.items()},
                     "pending_judge": sum(r.pending_judge for r in e.runs),
                     "runs": [_run_info(m, r.dir) for m, r in zip(e.meta, e.runs)]} for e in entries],
        "datasets": [{"col": c, "dataset": d, "scores": [_num(e.datasets.get((c, d))) for e in entries]} for c, d in ds],
    }


def write(entries: list[Entry], path, title: str = "AgML benchmark results") -> Path:
    data = json.dumps(payload(entries, title), separators=(",", ":")).replace("</", "<\\/")
    page = TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__DATA__", data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page)
    return path


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4; --s6: #008300; --s7: #6250d6; --s8: #e34948;
  --yolo: #d9d9d9; --yolo-edge: #555555; --ref: #222222; --warn: #8a5a00; --ok: #006300; --accent: #2a78d6;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --muted: #898781; --grid: #2c2c2a; --axis: #383835;
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #0ca30c; --s7: #9085e9; --s8: #e66767;
    --yolo: #4a4a48; --yolo-edge: #b5b4ad; --ref: #e0dfda; --warn: #fab219; --ok: #0ca30c; --accent: #3987e5; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --muted: #898781; --grid: #2c2c2a; --axis: #383835;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #0ca30c; --s7: #9085e9; --s8: #e66767;
  --yolo: #4a4a48; --yolo-edge: #b5b4ad; --ref: #e0dfda; --warn: #fab219; --ok: #0ca30c; --accent: #3987e5; color-scheme: dark;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink); font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
header { padding: 20px 16px 0; max-width: 1180px; margin: 0 auto; display: flex; gap: 12px; align-items: baseline; flex-wrap: wrap; }
h1 { font-size: 20px; margin: 0; font-weight: 600; }
.sub { color: var(--ink2); font-size: 13px; }
.theme { margin-left: auto; background: none; border: 1px solid var(--axis); color: var(--ink2); border-radius: 6px; padding: 3px 10px; cursor: pointer; }
nav { max-width: 1180px; margin: 12px auto 0; padding: 0 16px; display: flex; gap: 4px; border-bottom: 1px solid var(--grid); overflow-x: auto; }
nav button { background: none; border: 0; border-bottom: 2px solid transparent; padding: 8px 12px; color: var(--ink2); font: inherit; cursor: pointer; white-space: nowrap; }
nav button[aria-selected="true"] { color: var(--ink); border-bottom-color: var(--accent); font-weight: 600; }
main { max-width: 1180px; margin: 0 auto; padding: 16px; }
section[hidden] { display: none; }
.card { background: var(--surface); border: 1px solid var(--grid); border-radius: 10px; padding: 16px; margin-bottom: 16px; }
.card h2 { font-size: 15px; margin: 0 0 4px; font-weight: 600; }
.card p.note { margin: 0 0 12px; color: var(--ink2); font-size: 13px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; margin: 8px 0 4px; font-size: 12.5px; color: var(--ink2); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.sw { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
.chart { width: 100%; overflow-x: auto; }
svg text { fill: var(--ink2); font-size: 11px; }
svg .axis { stroke: var(--axis); }
svg .gridline { stroke: var(--grid); }
svg .bar:hover, svg .bar.hl { filter: brightness(1.12); }
.panels { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 12px; }
.panel h3 { font-size: 13px; margin: 0 0 4px; font-weight: 600; color: var(--ink2); }
.tip { position: fixed; pointer-events: none; background: var(--surface); color: var(--ink); border: 1px solid var(--axis);
       border-radius: 8px; padding: 8px 10px; font-size: 12.5px; box-shadow: 0 4px 14px rgba(0,0,0,.18); z-index: 10; max-width: 300px; }
.tip b { font-weight: 600; } .tip .k { color: var(--ink2); }
.tablewrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; font-variant-numeric: tabular-nums; }
th, td { padding: 6px 8px; border-bottom: 1px solid var(--grid); text-align: right; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--ink2); font-weight: 600; position: sticky; top: 0; background: var(--surface); }
th.sort { cursor: pointer; } th.sort:hover { color: var(--ink); }
td .ci { color: var(--muted); font-size: 11.5px; }
td.best { font-weight: 700; }
td .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; vertical-align: 0; }
.controls { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; }
.controls select, .controls input { font: inherit; padding: 5px 8px; border: 1px solid var(--axis); border-radius: 6px; background: var(--surface); color: var(--ink); }
.controls input { flex: 1; min-width: 160px; }
.flag { font-size: 12px; padding: 1px 7px; border-radius: 10px; border: 1px solid currentColor; }
.flag.ok { color: var(--ok); } .flag.no { color: var(--warn); }
.why { color: var(--ink2); font-size: 12px; white-space: normal; text-align: left; }
.empty { color: var(--muted); padding: 20px 0; text-align: center; }
@media (max-width: 640px) { header { padding-top: 14px; } h1 { font-size: 17px; } th, td { padding: 5px 6px; } }
</style>
</head>
<body>
<header>
  <h1 id="title"></h1><span class="sub" id="subtitle"></span>
  <button class="theme" id="theme" type="button">Theme</button>
</header>
<nav role="tablist" id="tabs">
  <button role="tab" data-tab="overview" aria-selected="true">Overview</button>
  <button role="tab" data-tab="ag">Agricultural tasks</button>
  <button role="tab" data-tab="datasets">Datasets</button>
  <button role="tab" data-tab="runs">Runs</button>
</nav>
<main>
<section id="overview">
  <div class="card">
    <h2>Accuracy by task</h2>
    <p class="note">Mean over datasets of each dataset's accuracy (averaged over repeated runs); bars show the 95% bootstrap CI over datasets. YOLO is scored by top-1 accuracy in every classification column. Dashed lines: chance.</p>
    <div class="legend" id="legend-main"></div>
    <div class="chart" id="chart-main"></div>
  </div>
  <div class="card"><h2>Table</h2><div class="tablewrap" id="table-main"></div></div>
</section>
<section id="ag" hidden>
  <div class="card">
    <h2>Classification by agricultural task</h2>
    <p class="note">Same aggregation, per agricultural task and question format. In the OEQ panel the dark tick marks the fuzzy-match score.</p>
    <div class="legend" id="legend-ag"></div>
    <div class="panels" id="chart-ag"></div>
  </div>
  <div class="card"><h2>Table</h2><div class="tablewrap" id="table-ag"></div></div>
</section>
<section id="datasets" hidden>
  <div class="card">
    <h2>Per dataset</h2>
    <p class="note">Accuracy (%) per dataset and task, averaged over each entry's runs. Click a column header to sort.</p>
    <div class="controls"><select id="ds-col"></select><input id="ds-q" type="search" placeholder="Filter datasets"></div>
    <div class="tablewrap" id="table-ds"></div>
  </div>
</section>
<section id="runs" hidden>
  <div class="card">
    <h2>Runs</h2>
    <p class="note">Comparable: frozen benchmark at its recorded hash, standard items, reference judge (agbench eval check).</p>
    <div class="tablewrap" id="table-runs"></div>
  </div>
</section>
</main>
<div class="tip" id="tip" hidden></div>
<script>
const D = __DATA__;
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const fmt = (v) => v == null ? "–" : v.toFixed(1);
const NS = "http://www.w3.org/2000/svg";
const bars = D.entries.filter((e) => e.kind !== "chance");
const chance = D.entries.find((e) => e.kind === "chance");
let slot = 0;
for (const e of D.entries) e.color = e.kind === "model" ? `var(--s${++slot})` : e.kind === "yolo" ? "url(#hatch)" : "var(--ref)";

$("#title").textContent = D.title;
const nRuns = D.entries.reduce((a, e) => a + e.runs.length, 0);
$("#subtitle").textContent = `${D.entries.length} entries, ${nRuns} runs · ${D.written}`;

// theme
const root = document.documentElement;
try { const t = localStorage.getItem("agbench-theme"); if (t) root.dataset.theme = t; } catch (e) {}
$("#theme").onclick = () => {
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("agbench-theme", root.dataset.theme); } catch (e) {}
};
// tabs
for (const b of document.querySelectorAll("#tabs button")) b.onclick = () => {
  for (const x of document.querySelectorAll("#tabs button")) x.setAttribute("aria-selected", x === b);
  for (const s of document.querySelectorAll("main section")) s.hidden = s.id !== b.dataset.tab;
};
// tooltip
const tip = $("#tip");
function showTip(ev, html) { tip.innerHTML = html; tip.hidden = false; moveTip(ev); }
function moveTip(ev) {
  const r = tip.getBoundingClientRect(), x = ev.clientX + 14, y = ev.clientY + 14;
  tip.style.left = Math.min(x, innerWidth - r.width - 8) + "px"; tip.style.top = Math.min(y, innerHeight - r.height - 8) + "px";
}
const hideTip = () => { tip.hidden = true; };

function el(tag, attrs, parent) {
  const n = document.createElementNS(NS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(n);
  return n;
}
function legend(id, withChance, extra) {
  const L = $(id);
  for (const e of bars) {
    const s = e.kind === "yolo" ? `background: repeating-linear-gradient(45deg, var(--yolo-edge) 0 1.5px, var(--yolo) 1.5px 4px)` : `background:${e.color}`;
    L.insertAdjacentHTML("beforeend", `<span><i class="sw" style="${s}"></i>${esc(e.label)}${e.replicates > 1 ? ` (${e.replicates} runs)` : ""}</span>`);
  }
  if (withChance && chance) L.insertAdjacentHTML("beforeend", `<span><svg width="18" height="8"><line x1="0" y1="4" x2="18" y2="4" stroke="var(--ref)" stroke-dasharray="3 2" stroke-width="1.5"/></svg>Chance (random answer)</span>`);
  if (extra) L.insertAdjacentHTML("beforeend", extra);
}

// grouped bars: groups [{label, sub}], value(entry, gi) -> {m, lo, hi, n} | null
function barChart(host, groups, value, opts = {}) {
  const W = opts.width || Math.max(560, 140 * groups.length), H = opts.height || 300, m = {l: 40, r: 8, t: 10, b: opts.sub ? 40 : 28};
  const svg = el("svg", {viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": opts.aria || "bar chart"}, null);
  svg.style.minWidth = (opts.minWidth || 480) + "px";
  const defs = el("defs", {}, svg), pat = el("pattern", {id: "hatch", width: 5, height: 5, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)"}, defs);
  el("rect", {width: 5, height: 5, fill: "var(--yolo)"}, pat); el("line", {x1: 0, y1: 0, x2: 0, y2: 5, stroke: "var(--yolo-edge)", "stroke-width": 1.5}, pat);
  const pw = W - m.l - m.r, ph = H - m.t - m.b, y = (v) => m.t + ph * (1 - v / 100);
  for (let t = 0; t <= 100; t += 20) {
    el("line", {x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), class: t ? "gridline" : "axis"}, svg);
    el("text", {x: m.l - 6, y: y(t) + 4, "text-anchor": "end"}, svg).textContent = t;
  }
  const gw = pw / groups.length, n = bars.length, bw = Math.min(26, (gw * 0.82) / n), gap = 2;
  groups.forEach((g, gi) => {
    const cx = m.l + gw * (gi + 0.5);
    const t = el("text", {x: cx, y: H - m.b + 16, "text-anchor": "middle"}, svg); t.textContent = g.label;
    if (opts.sub && g.sub) { const s = el("text", {x: cx, y: H - m.b + 30, "text-anchor": "middle"}, svg); s.textContent = g.sub; s.style.fill = "var(--muted)"; }
    bars.forEach((e, i) => {
      const v = value(e, gi); if (!v || v.m == null) return;
      const x = cx + (i - n / 2) * bw + gap / 2, w = bw - gap, top = y(v.m), r = Math.min(4, w / 2, y(0) - top);
      const d = `M${x},${y(0)} V${top + r} Q${x},${top} ${x + r},${top} H${x + w - r} Q${x + w},${top} ${x + w},${top + r} V${y(0)} Z`;
      const p = el("path", {d, fill: e.color, class: "bar"}, svg);
      if (e.kind === "yolo") p.setAttribute("stroke", "var(--yolo-edge)");
      if (v.lo != null && v.hi != null) {
        const xm = x + w / 2;
        el("line", {x1: xm, x2: xm, y1: y(v.lo), y2: y(v.hi), stroke: "var(--ink2)", "stroke-width": 1}, svg);
        for (const yy of [v.lo, v.hi]) el("line", {x1: xm - 2, x2: xm + 2, y1: y(yy), y2: y(yy), stroke: "var(--ink2)", "stroke-width": 1}, svg);
      }
      if (opts.tick) { const tv = opts.tick(e, gi); if (tv != null) el("line", {x1: x, x2: x + w, y1: y(tv), y2: y(tv), stroke: "var(--ref)", "stroke-width": 1.5}, svg); }
      const hit = el("rect", {x: x - gap / 2, y: m.t, width: bw, height: ph, fill: "transparent"}, svg);
      const html = `<b>${esc(e.label)}</b><br>${esc(g.label)}${g.sub ? " · " + esc(g.sub) : ""}<br>${fmt(v.m)}%` +
        (v.lo != null ? ` <span class="k">[${fmt(v.lo)}, ${fmt(v.hi)}]</span>` : "") + `<br><span class="k">${v.n} datasets` +
        (e.replicates > 1 ? `, ${e.replicates} runs` : "") + `</span>` + (opts.tipExtra ? opts.tipExtra(e, gi) : "");
      hit.onmouseenter = (ev) => { p.classList.add("hl"); showTip(ev, html); };
      hit.onmousemove = moveTip; hit.onmouseleave = () => { p.classList.remove("hl"); hideTip(); };
    });
    if (opts.chance && chance) {
      const v = opts.chance(chance, gi);
      if (v && v.m != null) el("line", {x1: cx - gw * 0.44, x2: cx + gw * 0.44, y1: y(v.m), y2: y(v.m), stroke: "var(--ref)", "stroke-width": 1.5, "stroke-dasharray": "4 3"}, svg);
    }
  });
  host.appendChild(svg);
}

// cells: html string, or {h, best}
function table(host, head, rows) {
  let h = "<table><thead><tr>" + head.map((x) => `<th>${x}</th>`).join("") + "</tr></thead><tbody>";
  for (const r of rows) h += "<tr>" + r.map((c) => typeof c === "object" ? `<td${c.best ? ' class="best"' : ""}>${c.h}</td>` : `<td>${c}</td>`).join("") + "</tr>";
  host.innerHTML = h + "</tbody></table>";
}
const name = (e) => `<i class="dot" style="background:${e.kind === "yolo" ? "var(--yolo-edge)" : e.color}"></i>${esc(e.label)}`;
const ci = (c) => c && c.m != null ? `${fmt(c.m)} <span class="ci">[${fmt(c.lo)}, ${fmt(c.hi)}]</span>` : "–";

// overview
const cols = D.cols.filter((c) => bars.some((e) => e.cells[c.key].n));
legend("#legend-main", true);
const nDs = (k) => Math.max(0, ...bars.map((e) => e.cells[k].n));
barChart($("#chart-main"), cols.map((c) => ({label: c.label, sub: `n=${nDs(c.key)}`})), (e, gi) => e.cells[cols[gi].key],
  {sub: true, chance: (e, gi) => e.cells[cols[gi].key], aria: "Accuracy by task"});
const bestBy = (vals) => Math.max(...vals.filter((v) => v != null));
const mainBest = cols.map((c) => bestBy(D.entries.filter((e) => e.kind === "model").map((e) => e.cells[c.key].m)));
table($("#table-main"), ["Entry", ...cols.map((c) => `${c.label}<br><span class="ci">n=${nDs(c.key)}</span>`)],
  D.entries.map((e) => [name(e) + (e.pending_judge ? ' <span class="flag no">not judged</span>' : ""),
    ...cols.map((c, j) => ({h: ci(e.cells[c.key]), best: e.kind === "model" && e.cells[c.key].m === mainBest[j]}))]));

// agricultural tasks
const ags = D.ag.filter((a) => bars.some((e) => D.formats.some((f) => (e.ag[`${f.key}|${a.key}`] || {}).n)));
legend("#legend-ag", false, `<span><svg width="18" height="8"><line x1="0" y1="4" x2="18" y2="4" stroke="var(--ref)" stroke-width="1.5"/></svg>OEQ fuzzy match</span>`);
for (const f of D.formats) {
  const p = document.createElement("div"); p.className = "panel"; p.innerHTML = `<h3>${esc(f.label)}</h3>`; $("#chart-ag").appendChild(p);
  barChart(p, ags.map((a) => ({label: a.label})), (e, gi) => e.ag[`${f.key}|${ags[gi].key}`],
    {width: 380, height: 260, minWidth: 300, aria: f.label,
     tick: f.key === "oeq" ? (e, gi) => e.kind === "model" ? e.fuzzy[ags[gi].key] : null : null,
     tipExtra: f.key === "oeq" ? (e, gi) => e.kind === "model" ? `<br><span class="k">fuzzy match ${fmt(e.fuzzy[ags[gi].key])}%</span>` : "" : null});
}
table($("#table-ag"), ["Entry", ...D.formats.flatMap((f) => ags.map((a) => `${f.label}<br><span class="ci">${a.label}</span>`))],
  D.entries.filter((e) => e.kind !== "chance").map((e) => [name(e), ...D.formats.flatMap((f) => ags.map((a) => fmt((e.ag[`${f.key}|${a.key}`] || {}).m)))]));

// datasets
const sel = $("#ds-col");
sel.innerHTML = `<option value="">All tasks</option>` + D.cols.filter((c) => D.datasets.some((d) => d.col === c.key)).map((c) => `<option value="${c.key}">${c.label}</option>`).join("");
const colLabel = Object.fromEntries(D.cols.map((c) => [c.key, c.label]));
let sortKey = -2, sortDir = 1;
function drawDs() {
  const q = $("#ds-q").value.toLowerCase(), c = sel.value;
  let rows = D.datasets.filter((d) => (!c || d.col === c) && (!q || d.dataset.toLowerCase().includes(q)));
  rows.sort((a, b) => sortKey === -2 ? a.dataset.localeCompare(b.dataset) * sortDir : sortKey === -1 ? a.col.localeCompare(b.col) * sortDir :
    ((a.scores[sortKey] ?? -1) - (b.scores[sortKey] ?? -1)) * sortDir);
  const host = $("#table-ds");
  if (!rows.length) { host.innerHTML = `<div class="empty">No datasets match.</div>`; return; }
  let h = `<table><thead><tr><th class="sort" data-k="-2">Dataset</th><th class="sort" data-k="-1">Task</th>` +
    D.entries.map((e, i) => `<th class="sort" data-k="${i}">${esc(e.label)}</th>`).join("") + "</tr></thead><tbody>";
  for (const r of rows.slice(0, 2000)) {
    const mx = bestBy(r.scores.filter((v, i) => D.entries[i].kind === "model"));
    h += `<tr><td>${esc(r.dataset)}</td><td>${esc(colLabel[r.col] || r.col)}</td>` +
      r.scores.map((v, i) => `<td${v != null && v === mx && D.entries[i].kind === "model" ? ' class="best"' : ""}>${fmt(v)}</td>`).join("") + "</tr>";
  }
  host.innerHTML = h + "</tbody></table>";
  for (const th of host.querySelectorAll("th.sort")) th.onclick = () => {
    const k = +th.dataset.k; sortDir = sortKey === k ? -sortDir : (k >= 0 ? -1 : 1); sortKey = k; drawDs();
  };
}
sel.onchange = drawDs; $("#ds-q").oninput = drawDs; drawDs();

// runs
table($("#table-runs"), ["Entry", "Run", "Model", "Seed", "Benchmark", "Hash", "Judge", "Answered", "Comparable"],
  D.entries.flatMap((e) => e.runs.map((r) => [name(e), esc(r.dir.split("/").slice(-2).join("/")), esc(r.model), r.gen_seed ?? "–",
    esc(r.benchmark || "–") + (r.variant ? ` (${esc(r.variant)})` : ""), `<code>${esc(r.hash || "–")}</code>`, esc(r.judge || "–"),
    r.answered != null ? `${r.answered}/${r.items}` : "–",
    r.comparable == null ? "–" : r.comparable ? '<span class="flag ok">yes</span>' : `<span class="flag no">no</span><div class="why">${r.why.map(esc).join("<br>")}</div>`])));
</script>
</body>
</html>
"""
