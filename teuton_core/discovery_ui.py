"""Small browser UI for bucket-backed discovery."""
from __future__ import annotations

import json
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from teuton_core.visualizer import VisualizerConfig, artifact_metadata, discover_run_ids, job_detail, visualizer_snapshot
from teuton_runtime.discovery import DiscoveryRecord, scan_bucket_discovery_records
from teuton_runtime.storage import ObjectStore


LOGO_PATH = Path(__file__).resolve().parent / "_assets" / "teutonic.png"


def _load_logo_bytes() -> bytes:
    try:
        return LOGO_PATH.read_bytes()
    except OSError:
        return b""


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Cache-Control" content="no-cache, must-revalidate">
<title>Teutonic</title>
<link rel="icon" type="image/png" href="/teutonic.png">
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

:root {
    --paper: #fff;
    --ink: #000;
    --ink-inv: #fff;
    --ink-muted: #999;
    --border: 1px dashed #000;
    --border-faint: 1px dotted rgba(0,0,0,0.15);
    --font: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    --mono: 'Space Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
}

[data-theme="dark"] {
    --paper: #111;
    --ink: #ddd;
    --ink-inv: #111;
    --ink-muted: #777;
    --border: 1px dashed #ddd;
    --border-faint: 1px dotted rgba(255,255,255,0.15);
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    background: var(--paper);
    color: var(--ink);
    font: 13px/1.5 var(--font);
    display: flex;
    justify-content: center;
    min-height: 100vh;
    padding: 0;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-rendering: optimizeLegibility;
}

.page {
    width: 100%;
    max-width: 1400px;
    background: var(--paper);
    position: relative;
    padding: 32px 48px;
    display: flex;
    flex-direction: column;
    gap: 24px;
}

.section-label {
    text-transform: uppercase;
    font-weight: bold;
    font-size: 11px;
    border-bottom: var(--border);
    padding-bottom: 4px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
}
.section-label .count { font-weight: normal; opacity: 0.6; }

.scroll-box {
    overflow-y: auto;
    overflow-x: auto;
    scrollbar-width: thin;
    max-height: 60vh;
}
.scroll-box::-webkit-scrollbar { width: 4px; height: 4px; }
.scroll-box::-webkit-scrollbar-thumb { background: var(--ink); }

table {
    width: 100%;
    border-collapse: collapse;
    font-size: 11px;
    table-layout: fixed;
}
th {
    text-transform: uppercase;
    font-weight: 600;
    font-size: 10px;
    letter-spacing: 0.04em;
    text-align: left;
    padding: 3px 6px 3px 0;
    border-bottom: var(--border);
    background: var(--paper);
    position: sticky;
    top: 0;
    z-index: 1;
}
td {
    padding: 2px 6px 2px 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
tr { border-bottom: var(--border-faint); }
tr:hover td {
    background: var(--ink);
    color: var(--ink-inv);
    cursor: crosshair;
}
th:last-child, td:last-child { text-align: right; }
code { font-family: var(--mono); }

.miners-table, .jobs-table { min-width: 0; }
.miners-table th:nth-child(1), .miners-table td:nth-child(1),
.jobs-table th:nth-child(1), .jobs-table td:nth-child(1) { width: 4%; }
/* Miners: # | Status | M/H/W | GPU | UID | Emit | Inflight | Rcpts | Ping */
.miners-table th:nth-child(2), .miners-table td:nth-child(2) { width: 8%; }
.miners-table th:nth-child(3), .miners-table td:nth-child(3) { width: 18%; }
.miners-table th:nth-child(4), .miners-table td:nth-child(4) { width: 14%; }
.miners-table th:nth-child(5), .miners-table td:nth-child(5) { width: 6%; text-align: right; }
.miners-table th:nth-child(6), .miners-table td:nth-child(6) { width: 9%; text-align: right; }
.miners-table th:nth-child(7), .miners-table td:nth-child(7) { width: 18%; text-align: right; }
.miners-table th:nth-child(8), .miners-table td:nth-child(8) { width: 8%; text-align: right; }
.miners-table th:nth-child(9), .miners-table td:nth-child(9) { width: 8%; text-align: right; }
/* Outstanding jobs: # | Kind | Miner | Age | Attempt | Deadline */
.outstanding-table th:nth-child(1), .outstanding-table td:nth-child(1) { width: 4%; }
.outstanding-table th:nth-child(2), .outstanding-table td:nth-child(2) { width: 24%; }
.outstanding-table th:nth-child(3), .outstanding-table td:nth-child(3) { width: 14%; }
.outstanding-table th:nth-child(4), .outstanding-table td:nth-child(4) { width: 14%; text-align: right; }
.outstanding-table th:nth-child(5), .outstanding-table td:nth-child(5) { width: 14%; text-align: right; }
.outstanding-table th:nth-child(6), .outstanding-table td:nth-child(6) { width: 30%; text-align: right; }
/* Completed jobs: # | Kind | Status | Miner | Finished | Latency | I/O */
.jobs-table th:nth-child(2), .jobs-table td:nth-child(2) { width: 20%; }
.jobs-table th:nth-child(3), .jobs-table td:nth-child(3) { width: 12%; }
.jobs-table th:nth-child(4), .jobs-table td:nth-child(4) { width: 12%; }
.jobs-table th:nth-child(5), .jobs-table td:nth-child(5) { width: 13%; text-align: right; }
.jobs-table th:nth-child(6), .jobs-table td:nth-child(6) { width: 11%; text-align: right; }
.jobs-table th:nth-child(7), .jobs-table td:nth-child(7) { width: 24%; text-align: right; }

#theme-toggle {
    background: none;
    border: var(--border);
    color: var(--ink);
    font: 600 11px/1 var(--font);
    text-transform: uppercase;
    padding: 3px 8px;
    cursor: pointer;
    letter-spacing: 0.05em;
}
#theme-toggle:hover { background: var(--ink); color: var(--ink-inv); }

.filter-bar {
    display: flex;
    gap: 6px;
    margin-bottom: 8px;
    flex-wrap: wrap;
}
.filter-btn {
    background: none;
    border: var(--border);
    color: var(--ink);
    font: 600 11px/1 var(--font);
    text-transform: uppercase;
    padding: 4px 10px;
    cursor: pointer;
    letter-spacing: 0.05em;
}
.filter-btn:hover { background: var(--ink); color: var(--ink-inv); }
.filter-btn.active { background: var(--ink); color: var(--ink-inv); }
.filter-btn .badge { opacity: 0.6; margin-left: 6px; font-weight: 400; }
.filter-btn.active .badge { opacity: 0.9; }

.status-pill {
    display: inline-block;
    font: 600 10px/1 var(--mono);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 2px 6px;
    border: var(--border);
    white-space: nowrap;
}
.status-live   { background: var(--ink); color: var(--ink-inv); }
.status-stale  { opacity: 0.7; }
.status-seen   { opacity: 0.55; }
.status-assigned { opacity: 0.4; font-style: italic; }

#error { border: var(--border); padding: 8px; font-size: 11px; }

#compute-chart-wrap {
    position: relative;
    border-bottom: var(--border-faint);
    padding: 4px 0 16px;
}
#compute-chart {
    width: 100%;
    height: 140px;
    display: block;
    overflow: visible;
}
.compute-line {
    fill: none;
    stroke: var(--ink);
    stroke-width: 1.6;
    vector-effect: non-scaling-stroke;
    stroke-linejoin: round;
    stroke-linecap: round;
}
.compute-line-raw {
    fill: none;
    stroke: var(--ink);
    stroke-width: 0.8;
    opacity: 0.25;
    vector-effect: non-scaling-stroke;
    stroke-linejoin: round;
    stroke-linecap: round;
}
.compute-area {
    fill: var(--ink);
    opacity: 0.05;
}
.compute-baseline {
    stroke: var(--ink);
    stroke-dasharray: 2 4;
    opacity: 0.25;
    vector-effect: non-scaling-stroke;
}
.compute-cursor {
    stroke: var(--ink);
    opacity: 0.4;
    vector-effect: non-scaling-stroke;
    pointer-events: none;
}
.compute-empty {
    font: 600 11px/1 var(--mono);
    fill: var(--ink-muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.compute-axis {
    font: 10px/1 var(--mono);
    color: var(--ink-muted);
    letter-spacing: 0.04em;
    text-transform: uppercase;
    pointer-events: none;
}
.compute-axis-ymax { position: absolute; top: 4px; right: 2px; }
.compute-axis-xstart { position: absolute; bottom: 0; left: 0; }
.compute-axis-xend { position: absolute; bottom: 0; right: 0; }
.compute-tooltip {
    position: absolute;
    display: none;
    pointer-events: none;
    padding: 2px 6px;
    background: var(--paper);
    border: var(--border-faint);
    white-space: nowrap;
    z-index: 2;
    color: var(--ink);
}
.graph-label {
    align-items: center;
}
.metric-tabs {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    justify-content: center;
}
.metric-tab {
    background: none;
    border: var(--border);
    color: var(--ink);
    font: 600 10px/1 var(--font);
    text-transform: uppercase;
    padding: 3px 8px;
    cursor: pointer;
    letter-spacing: 0.05em;
}
.metric-tab:hover { background: var(--ink); color: var(--ink-inv); }
.metric-tab.active { background: var(--ink); color: var(--ink-inv); }

/* ---------------------------------------------------------------------
   Queue panel: depth headline + sparkline + per-miner inflight grid.
   Sized to render between the hero and the compute chart without
   needing its own scroll container.
   --------------------------------------------------------------------- */
.queue-headline {
    display: flex;
    align-items: baseline;
    gap: 16px;
    flex-wrap: wrap;
    padding: 4px 0 8px;
    border-bottom: var(--border-faint);
}
.queue-depth-big {
    font: 700 28px/1 var(--mono);
    letter-spacing: 0.04em;
}
.queue-depth-cap {
    font: 600 12px/1 var(--mono);
    opacity: 0.55;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.queue-meta {
    font: 600 10px/1.4 var(--mono);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    opacity: 0.7;
}
.queue-meta strong { font-weight: 700; opacity: 1; }
.queue-meta .alert { color: var(--ink); font-weight: 700; }

#queue-spark-wrap {
    position: relative;
    padding: 4px 0 8px;
    border-bottom: var(--border-faint);
}
#queue-spark {
    width: 100%;
    height: 80px;
    display: block;
    overflow: visible;
}
.queue-spark-line {
    fill: none;
    stroke: var(--ink);
    stroke-width: 1.4;
    vector-effect: non-scaling-stroke;
    stroke-linejoin: round;
}
.queue-spark-area {
    fill: var(--ink);
    opacity: 0.05;
}
.queue-spark-baseline {
    stroke: var(--ink);
    opacity: 0.2;
    stroke-dasharray: 2 4;
    vector-effect: non-scaling-stroke;
}
.queue-empty {
    font: 600 11px/1 var(--mono);
    fill: var(--ink-muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.queue-axis {
    position: absolute;
    font: 10px/1 var(--mono);
    color: var(--ink-muted);
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.queue-axis-xstart { bottom: 8px; left: 0; }
.queue-axis-xend { bottom: 8px; right: 0; }
.queue-axis-ymax { top: 4px; right: 0; }

#queue-bars {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 6px 16px;
    padding-top: 8px;
}
.queue-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    font: 11px/1 var(--mono);
}
.queue-bar-id { width: 96px; flex-shrink: 0; opacity: 0.85; }
.queue-bar-track {
    flex: 1;
    height: 8px;
    border: var(--border-faint);
    position: relative;
}
.queue-bar-fill {
    height: 100%;
    background: var(--ink);
    opacity: 0.55;
}
.queue-bar.at-cap .queue-bar-fill { opacity: 1; }
.queue-bar-num {
    width: 44px;
    text-align: right;
    font-variant-numeric: tabular-nums;
}
.queue-bar.at-cap .queue-bar-num { font-weight: 700; }

/* ---------------------------------------------------------------------
   Inline inflight bar inside the Miners table.
   --------------------------------------------------------------------- */
.inflight-cell {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font: 600 11px/1 var(--mono);
}
.inflight-track {
    width: 56px;
    height: 6px;
    border: var(--border-faint);
    position: relative;
}
.inflight-fill {
    height: 100%;
    background: var(--ink);
    opacity: 0.55;
}
.inflight-cell.at-cap .inflight-fill { opacity: 1; }
.inflight-cell.at-cap { font-weight: 700; }

/* Completed-job status pill colors layered on the existing status-pill base. */
.status-completed { background: var(--ink); color: var(--ink-inv); opacity: 0.6; }
.status-verified  { background: var(--ink); color: var(--ink-inv); }
.status-failed    { opacity: 0.7; font-style: italic; }

/* Deadline countdown column: red when expired. */
.deadline-expired { font-weight: 700; }
.attempt-retry { font-weight: 700; }

@media (max-width: 640px) {
    .page { padding: 16px 12px; gap: 16px; }
    #top-bar { font-size: 11px !important; }
    .scroll-box table { min-width: 520px; }
}
@media (max-width: 1100px) {
    .page { padding: 20px 22px; }
}
</style>
</head>
<body>
<div class="page">

    <div id="top-bar" style="display:flex;justify-content:flex-end;align-items:center;border-bottom:var(--border);padding:6px 0;font-size:12px;text-transform:uppercase;gap:12px;flex-wrap:wrap;">
        <button id="theme-toggle">DARK</button>
    </div>

    <div class="hero" style="text-align:center;margin-bottom:8px;">
        <img src="/teutonic.png" alt="Teutonic" class="logo-img" style="width:80px;height:80px;margin-bottom:8px;">
        <div class="header-title" style="font-size:20px;font-weight:bold;letter-spacing:0.1em;text-transform:uppercase;">Teutonic</div>
        <div style="font-size:11px;margin-top:4px;display:flex;flex-direction:column;align-items:center;gap:4px;text-transform:uppercase;opacity:0.75;">
            <span id="hero-meta">--</span>
        </div>
    </div>

    <div id="error" style="display:none"></div>

    <div>
        <div class="section-label">
            <span>Queue</span>
            <span class="count" id="queue-meta-line">--</span>
        </div>
        <div class="queue-headline">
            <div>
                <div class="queue-depth-big" id="queue-depth-big">--</div>
                <div class="queue-depth-cap" id="queue-depth-cap">outstanding / cap</div>
            </div>
            <div class="queue-meta">
                <div>BACKPRESSURE <strong id="queue-backpressure">--</strong></div>
                <div>OLDEST <strong id="queue-oldest">--</strong></div>
                <div>SNAPSHOT <strong id="queue-snapshot-age">--</strong></div>
            </div>
        </div>
        <div id="queue-spark-wrap">
            <svg id="queue-spark" viewBox="0 0 1000 80" preserveAspectRatio="none"></svg>
            <span id="queue-spark-y-max" class="queue-axis queue-axis-ymax"></span>
            <span class="queue-axis queue-axis-xstart">-30m</span>
            <span class="queue-axis queue-axis-xend">now</span>
        </div>
        <div id="queue-bars"></div>
    </div>

    <div>
        <div class="section-label graph-label">
            <span id="graph-title">Metrics</span>
            <div class="metric-tabs" id="graph-tabs"></div>
            <span class="count" id="compute-rate">--</span>
        </div>
        <div id="compute-chart-wrap">
            <svg id="compute-chart" viewBox="0 0 1000 140" preserveAspectRatio="none"></svg>
            <span id="compute-y-max" class="compute-axis compute-axis-ymax"></span>
            <span class="compute-axis compute-axis-xstart">-30m</span>
            <span class="compute-axis compute-axis-xend">now</span>
            <span id="compute-tooltip" class="compute-axis compute-tooltip"></span>
        </div>
    </div>

    <div>
        <div class="section-label">
            <span>Miners</span>
            <span class="count" id="miners-count">--</span>
        </div>
        <div class="filter-bar" id="miners-filter"></div>
        <div class="scroll-box">
            <table class="data-table miners-table">
                <thead><tr>
                    <th>#</th>
                    <th data-miner-sort="status">Status</th>
                    <th data-miner-sort="identity">M/H/W</th>
                    <th>GPU</th>
                    <th data-miner-sort="uid">UID</th>
                    <th data-miner-sort="emission">Emit</th>
                    <th data-miner-sort="inflight">Inflight</th>
                    <th data-miner-sort="receipts">Rcpts</th>
                    <th data-miner-sort="ping">Ping</th>
                </tr></thead>
                <tbody id="miners"></tbody>
            </table>
        </div>
    </div>

    <div>
        <div class="section-label">
            <span>Outstanding</span>
            <span class="count" id="outstanding-count">--</span>
        </div>
        <div class="scroll-box">
            <table class="data-table outstanding-table">
                <thead><tr>
                    <th>#</th>
                    <th>Kind</th>
                    <th>Miner</th>
                    <th>Age</th>
                    <th>Attempt</th>
                    <th>Deadline</th>
                </tr></thead>
                <tbody id="outstanding"></tbody>
            </table>
        </div>
    </div>

    <div>
        <div class="section-label">
            <span>Completed</span>
            <span class="count" id="jobs-count">--</span>
        </div>
        <div class="filter-bar" id="jobs-filter"></div>
        <div class="scroll-box">
            <table class="data-table jobs-table">
                <thead><tr>
                    <th>#</th>
                    <th>Kind</th>
                    <th>Status</th>
                    <th>Miner</th>
                    <th>Finished</th>
                    <th>Latency</th>
                    <th>I/O</th>
                </tr></thead>
                <tbody id="jobs"></tbody>
            </table>
        </div>
    </div>

</div>

<script>
var POLL_MS = __REFRESH_MS__;
var JOB_STATUS_FILTERS = [
    { id: "all", label: "All" },
    { id: "completed", label: "Completed" },
    { id: "verified", label: "Verified" },
    { id: "failed", label: "Failed" }
];
var MINER_STATUS_FILTERS = [
    { id: "all", label: "All" },
    { id: "live", label: "Live" },
    { id: "stale", label: "Stale" },
    { id: "at-cap", label: "At-cap" }
];
var GRAPH_METRICS = [
    { id: "compute", label: "Compute", unit: "CU/s", empty: "Awaiting first compute receipt" },
    { id: "bandwidth", label: "Bandwidth", unit: "B/s", empty: "Awaiting first I/O receipt" },
    { id: "jobs", label: "Jobs", unit: "jobs/s", empty: "Awaiting first completed job" },
    { id: "latency", label: "Latency", unit: "avg", empty: "Awaiting first timed job" }
];
var DEFAULT_JOBS_FILTER = "all";
var DEFAULT_MINERS_FILTER = "all";
var DEFAULT_GRAPH_METRIC = "compute";

(function maybeResetFilters() {
    try {
        var params = new URLSearchParams(window.location.search);
        if (params.get("reset") === "1") {
            localStorage.removeItem("jobsFilter");
            localStorage.removeItem("minersFilter");
            localStorage.removeItem("graphMetric");
        }
    } catch (e) {}
})();

function _validFilter(stored, valid, fallback) {
    if (!stored) return fallback;
    for (var i = 0; i < valid.length; i++) {
        if (valid[i].id === stored) return stored;
    }
    return fallback;
}

var COMPUTE_WINDOW_SEC = 30 * 60;
var COMPUTE_BIN_SEC = 30;
var COMPUTE_BINS = COMPUTE_WINDOW_SEC / COMPUTE_BIN_SEC;
var state = {
    snapshot: null,
    jobsFilter: _validFilter(localStorage.getItem("jobsFilter"), JOB_STATUS_FILTERS, DEFAULT_JOBS_FILTER),
    minersFilter: _validFilter(localStorage.getItem("minersFilter"), MINER_STATUS_FILTERS, DEFAULT_MINERS_FILTER),
    graphMetric: _validFilter(localStorage.getItem("graphMetric"), GRAPH_METRICS, DEFAULT_GRAPH_METRIC),
    minersSort: { key: localStorage.getItem("minersSortKey") || "inflight", dir: localStorage.getItem("minersSortDir") || "desc" },
    computeRates: null,
    graphRawRates: null,
    graphMetricDef: null,
    computeNowUnix: 0
};

(function initTheme() {
    var saved = localStorage.getItem("theme");
    var dark = saved ? saved === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.dataset.theme = dark ? "dark" : "light";
})();

function esc(value) {
    return String(value == null ? "" : value).replace(/[&<>"']/g, function(ch) {
        return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch];
    });
}

function short(s, n) {
    if (!s) return "--";
    return s.length > n ? s.slice(0, n) + "\u2026" : s;
}

function trim5(s) {
    if (!s) return "--";
    return String(s).slice(0, 5);
}

function fmtBytes(n) {
    n = Number(n || 0);
    if (n < 1024) return n + " B";
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
    if (n < 1024 * 1024 * 1024) return (n / 1024 / 1024).toFixed(2) + " MB";
    return (n / 1024 / 1024 / 1024).toFixed(2) + " GB";
}

function fmtDurationSec(n) {
    n = Number(n);
    if (!isFinite(n) || n < 0) return "--";
    if (n < 1) return Math.round(n * 1000) + "ms";
    if (n < 60) return (n < 10 ? n.toFixed(1) : Math.round(n)) + "s";

    var total = Math.round(n);
    var days = Math.floor(total / 86400);
    total -= days * 86400;
    var hours = Math.floor(total / 3600);
    total -= hours * 3600;
    var minutes = Math.floor(total / 60);
    var seconds = total - minutes * 60;

    if (days > 0) return days + "d" + (hours > 0 ? " " + hours + "h" : "");
    if (hours > 0) return hours + "h" + (minutes > 0 ? " " + minutes + "m" : "");
    return minutes + "m" + (seconds > 0 ? " " + seconds + "s" : "");
}

function fmtDurationMs(n) {
    n = Number(n);
    if (!isFinite(n) || n < 0) return "--";
    return fmtDurationSec(n / 1000);
}

function fmtSec(n) {
    return fmtDurationSec(n);
}

function fmtPoints(n) {
    n = Number(n || 0);
    if (n >= 1000) return (n / 1000).toFixed(1) + "k";
    if (n >= 100) return n.toFixed(0);
    return n.toFixed(2);
}

function fmtTime(ts) {
    return ts ? new Date(ts * 1000).toLocaleTimeString() : "--";
}

function ageLabel(seconds) {
    seconds = Number(seconds || 0);
    if (seconds < 60) return Math.round(seconds) + "s ago";
    if (seconds < 3600) return Math.round(seconds / 60) + "m ago";
    if (seconds < 86400) return Math.round(seconds / 3600) + "h ago";
    return Math.round(seconds / 86400) + "d ago";
}

function allWorkers() {
    var out = [];
    var machines = (state.snapshot && state.snapshot.machines) || [];
    for (var m = 0; m < machines.length; m++) {
        var rows = machines[m].workers || [];
        for (var r = 0; r < rows.length; r++) {
            out.push({ machine: machines[m], row: rows[r] });
        }
    }
    return out;
}

function renderMinersFilter(counts) {
    var bar = document.getElementById("miners-filter");
    bar.innerHTML = MINER_STATUS_FILTERS.map(function(f) {
        var n = counts[f.id] || 0;
        var cls = "filter-btn" + (state.minersFilter === f.id ? " active" : "");
        return '<button type="button" class="' + cls + '" data-filter="' + esc(f.id) + '">' +
            esc(f.label) + ' <span class="badge">' + n + '</span></button>';
    }).join("");
}

function renderMiners() {
    var rows = allWorkers();
    var counts = { all: rows.length };
    MINER_STATUS_FILTERS.forEach(function(f) { if (f.id !== "all") counts[f.id] = 0; });
    for (var k = 0; k < rows.length; k++) {
        var st = rows[k].row.status || "live";
        if (counts[st] !== undefined) counts[st] += 1;
        if (rows[k].row.at_cap) counts["at-cap"] = (counts["at-cap"] || 0) + 1;
    }
    renderMinersFilter(counts);
    var filter = state.minersFilter;
    var filtered;
    if (filter === "all") {
        filtered = rows.slice();
    } else if (filter === "at-cap") {
        filtered = rows.filter(function(r) { return !!r.row.at_cap; });
    } else {
        filtered = rows.filter(function(r) { return (r.row.status || "live") === filter; });
    }
    filtered.sort(minerSortComparator(state.minersSort.key, state.minersSort.dir));
    document.getElementById("miners-count").textContent = filtered.length + " of " + rows.length;
    var body = document.getElementById("miners");
    if (!filtered.length) {
        body.innerHTML = '<tr><td colspan="9" style="text-align:center;padding:16px">No miners match this filter.</td></tr>';
        return;
    }
    renderMinerSortHeaders();
    body.innerHTML = filtered.map(function(item, i) {
        var w = item.row.worker || {};
        var cap = w.capabilities || {};
        var chain = item.row.chain || (item.row.miner && item.row.miner.chain) || {};
        var gpu = cap.gpu_name || cap.gpu_class || (w.gpu_index != null ? "gpu" + w.gpu_index : "--");
        var status = item.row.status || "live";
        var pillCls = "status-pill status-" + status;
        var sources = (item.row.sources || []).join("+") || "";
        var pingMs = cap.rtt_to_bucket_ms;
        var pingStr = (pingMs == null) ? "--" : fmtDurationMs(pingMs);
        var identityTitle = [
            "miner=" + (w.hotkey_ss58 || "--"),
            "host=" + (w.host_id || "--"),
            "worker=" + (w.worker_id || "--")
        ].join(" ");
        var identity = trim5(w.hotkey_ss58) + "/" + trim5(w.host_id) + "/" + trim5(w.worker_id);
        var depth = Number(item.row.queue_depth || 0);
        var qcap = Number(item.row.queue_cap || 0);
        var atCap = !!item.row.at_cap;
        var pct = (qcap > 0) ? Math.min(100, (depth / qcap) * 100) : 0;
        var inflightHtml = (qcap > 0)
            ? '<span class="inflight-cell' + (atCap ? ' at-cap' : '') + '">' +
              '<span class="inflight-track"><span class="inflight-fill" style="width:' + pct.toFixed(0) + '%"></span></span>' +
              '<span>' + depth + '/' + qcap + '</span></span>'
            : '<span class="inflight-cell"><span>' + depth + '</span></span>';
        return '<tr>' +
            '<td>' + (i + 1) + '</td>' +
            '<td><span class="' + pillCls + '" title="' + esc(sources) + '">' + esc(status) + '</span></td>' +
            '<td><code title="' + esc(identityTitle) + '">' + esc(identity) + '</code></td>' +
            '<td>' + esc(gpu) + '</td>' +
            '<td style="text-align:right">' + esc(chain.uid == null ? "--" : chain.uid) + '</td>' +
            '<td style="text-align:right" title="' + esc(chain.emission == null ? "" : chain.emission) + '">' + esc(chain.emission == null ? "--" : fmtPoints(chain.emission)) + '</td>' +
            '<td style="text-align:right">' + inflightHtml + '</td>' +
            '<td style="text-align:right">' + esc(item.row.n_receipts || 0) + '</td>' +
            '<td style="text-align:right">' + esc(pingStr) + '</td>' +
            '</tr>';
    }).join("");
}

function minerSortValue(item, key) {
    var w = item.row.worker || {};
    var cap = w.capabilities || {};
    var chain = item.row.chain || (item.row.miner && item.row.miner.chain) || {};
    if (key === "status") return item.row.status || "";
    if (key === "identity") return (w.hotkey_ss58 || "") + "/" + (w.host_id || "") + "/" + (w.worker_id || "");
    if (key === "uid") return chain.uid == null ? -1 : Number(chain.uid);
    if (key === "emission") return chain.emission == null ? -1 : Number(chain.emission);
    if (key === "receipts") return Number(item.row.n_receipts || 0);
    if (key === "inflight") return Number(item.row.queue_depth || 0);
    if (key === "ping") return cap.rtt_to_bucket_ms == null ? Number.POSITIVE_INFINITY : Number(cap.rtt_to_bucket_ms);
    return Number(item.row.queue_depth || 0);
}

function minerSortComparator(key, dir) {
    var factor = dir === "asc" ? 1 : -1;
    return function(a, b) {
        var av = minerSortValue(a, key);
        var bv = minerSortValue(b, key);
        if (typeof av === "number" && typeof bv === "number") {
            if (av !== bv) return (av - bv) * factor;
        } else {
            var cmp = String(av).localeCompare(String(bv));
            if (cmp !== 0) return cmp * factor;
        }
        return String(minerSortValue(a, "identity")).localeCompare(String(minerSortValue(b, "identity")));
    };
}

function renderMinerSortHeaders() {
    var headers = document.querySelectorAll("[data-miner-sort]");
    for (var i = 0; i < headers.length; i++) {
        var key = headers[i].getAttribute("data-miner-sort");
        var label = headers[i].getAttribute("data-label") || headers[i].textContent.replace(/[▲▼]/g, "").trim();
        headers[i].setAttribute("data-label", label);
        headers[i].style.cursor = "pointer";
        headers[i].textContent = label + (state.minersSort.key === key ? (state.minersSort.dir === "asc" ? " ▲" : " ▼") : "");
    }
}

function renderJobsFilter(counts) {
    var bar = document.getElementById("jobs-filter");
    bar.innerHTML = JOB_STATUS_FILTERS.map(function(f) {
        var n = counts[f.id] || 0;
        var cls = "filter-btn" + (state.jobsFilter === f.id ? " active" : "");
        return '<button type="button" class="' + cls + '" data-filter="' + esc(f.id) + '">' +
            esc(f.label) + ' <span class="badge">' + n + '</span></button>';
    }).join("");
}

function jobStatusClass(status) {
    return "status-pill status-" + status;
}

function completedJobs() {
    var jobs = (state.snapshot && state.snapshot.jobs);
    if (!jobs) return [];
    // Legacy dev-dashboard shape: jobs is a flat array of mixed states. Show
    // anything with a receipt/verdict as "completed".
    if (Array.isArray(jobs)) {
        return jobs.filter(function(j) {
            return j.status === "completed" || j.status === "verified" || j.status === "failed";
        });
    }
    return jobs.completed || [];
}

function outstandingJobs() {
    var jobs = (state.snapshot && state.snapshot.jobs);
    if (!jobs) return [];
    if (Array.isArray(jobs)) {
        // Legacy: best-effort -- treat the "created" status as outstanding so
        // the dev dashboard still has something to show in the new section.
        return jobs.filter(function(j) { return j.status === "created"; }).map(function(j) {
            return {
                job_id: j.job_id, kind: j.kind, assigned_hotkey: j.assigned_hotkey,
                assigned_worker: j.assigned_worker, attempt: j.attempt || 0,
                created_unix: j.created_unix, deadline_unix: j.deadline_unix
            };
        });
    }
    return jobs.outstanding || [];
}

function renderOutstandingJobs() {
    var jobs = outstandingJobs();
    document.getElementById("outstanding-count").textContent = jobs.length + " outstanding";
    var body = document.getElementById("outstanding");
    if (!jobs.length) {
        body.innerHTML = '<tr><td colspan="6" style="text-align:center;padding:16px">Queue is empty.</td></tr>';
        return;
    }
    var nowUnix = (state.snapshot && state.snapshot.meta && state.snapshot.meta.generated_unix) || (Date.now() / 1000);
    body.innerHTML = jobs.map(function(j, i) {
        var worker = j.assigned_worker || j.assigned_hotkey || "--";
        var workerTitle = "miner=" + (j.assigned_hotkey || "--") + " worker=" + (j.assigned_worker || "--");
        var age = j.created_unix ? Math.max(0, nowUnix - j.created_unix) : null;
        var ageLabel = age != null ? fmtDurationSec(age) : "--";
        var attemptCls = (Number(j.attempt || 0) > 0) ? "attempt-retry" : "";
        var deadlineLeft = j.deadline_unix ? (j.deadline_unix - nowUnix) : null;
        var deadlineLabel;
        var deadlineCls = "";
        if (deadlineLeft == null) {
            deadlineLabel = "--";
        } else if (deadlineLeft < 0) {
            deadlineLabel = "-" + fmtDurationSec(-deadlineLeft);
            deadlineCls = "deadline-expired";
        } else {
            deadlineLabel = fmtDurationSec(deadlineLeft);
        }
        return '<tr>' +
            '<td>' + (i + 1) + '</td>' +
            '<td>' + esc(j.kind || "--") + '</td>' +
            '<td><code title="' + esc(workerTitle) + '">' + esc(trim5(worker)) + '</code></td>' +
            '<td style="text-align:right">' + esc(ageLabel) + '</td>' +
            '<td style="text-align:right" class="' + attemptCls + '">' + esc(String(j.attempt || 0)) + '</td>' +
            '<td style="text-align:right" class="' + deadlineCls + '">' + esc(deadlineLabel) + '</td>' +
            '</tr>';
    }).join("");
}

function renderJobs() {
    var jobs = completedJobs();
    var counts = {};
    JOB_STATUS_FILTERS.forEach(function(f) { counts[f.id] = 0; });
    counts.all = jobs.length;
    for (var k = 0; k < jobs.length; k++) {
        var s = jobs[k].status;
        if (counts[s] !== undefined) counts[s] += 1;
    }
    renderJobsFilter(counts);
    var filter = state.jobsFilter;
    var filtered = filter === "all" ? jobs.slice() : jobs.filter(function(j) { return j.status === filter; });
    document.getElementById("jobs-count").textContent = filtered.length + " of " + jobs.length + " completed";
    var body = document.getElementById("jobs");
    if (!filtered.length) {
        var emptyLabel = filter === "all" ? "" : esc(filter) + " ";
        body.innerHTML = '<tr><td colspan="7" style="text-align:center;padding:16px">No ' + emptyLabel + 'completed jobs in this run.</td></tr>';
        return;
    }
    function activityUnix(j) {
        return Number(j.checked_unix || j.finished_unix || 0);
    }
    var ordered = filtered.slice().sort(function(a, b) { return activityUnix(b) - activityUnix(a); });
    var nowUnix = (state.snapshot && state.snapshot.meta && state.snapshot.meta.generated_unix) || (Date.now() / 1000);
    body.innerHTML = ordered.map(function(j, i) {
        var finishedAge = j.finished_unix ? Math.max(0, nowUnix - j.finished_unix) : null;
        var finishedLabel = finishedAge != null ? fmtDurationSec(finishedAge) + " ago" : "--";
        var finishedTitle = j.finished_unix ? fmtTime(j.finished_unix) : "";
        var latency = (j.duration_sec != null) ? fmtDurationSec(j.duration_sec) : "--";
        var ioBytes = Number(j.bytes_read || 0) + Number(j.bytes_written || 0);
        var io = ioBytes > 0 ? fmtBytes(ioBytes) : "--";
        var ioTitle = "read " + fmtBytes(j.bytes_read) + " / wrote " + fmtBytes(j.bytes_written);
        var worker = j.assigned_worker || j.assigned_hotkey || "--";
        return '<tr>' +
            '<td>' + (i + 1) + '</td>' +
            '<td>' + esc(j.kind || "--") + '</td>' +
            '<td><span class="' + jobStatusClass(j.status) + '">' + esc(j.status) + '</span></td>' +
            '<td><code title="' + esc(worker) + '">' + esc(trim5(worker)) + '</code></td>' +
            '<td style="text-align:right" title="' + esc(finishedTitle) + '">' + esc(finishedLabel) + '</td>' +
            '<td style="text-align:right">' + esc(latency) + '</td>' +
            '<td style="text-align:right" title="' + esc(ioTitle) + '">' + esc(io) + '</td>' +
            '</tr>';
    }).join("");
}

function renderQueue() {
    var snap = state.snapshot && state.snapshot.queue;
    var depthBigEl = document.getElementById("queue-depth-big");
    var depthCapEl = document.getElementById("queue-depth-cap");
    var bpEl = document.getElementById("queue-backpressure");
    var oldestEl = document.getElementById("queue-oldest");
    var snapAgeEl = document.getElementById("queue-snapshot-age");
    var metaLineEl = document.getElementById("queue-meta-line");
    var barsEl = document.getElementById("queue-bars");
    var spark = document.getElementById("queue-spark");
    var sparkYMax = document.getElementById("queue-spark-y-max");
    if (!snap) {
        depthBigEl.textContent = "--";
        depthCapEl.textContent = "no queue yet";
        bpEl.textContent = "--";
        oldestEl.textContent = "--";
        snapAgeEl.textContent = "--";
        metaLineEl.textContent = "no snapshot";
        barsEl.innerHTML = "";
        spark.innerHTML = '<text class="queue-empty" x="500" y="40" text-anchor="middle">Awaiting first queue snapshot</text>';
        sparkYMax.textContent = "";
        return;
    }

    var nowUnix = (state.snapshot.meta && state.snapshot.meta.generated_unix) || (Date.now() / 1000);
    var depth = Number(snap.depth_total || 0);
    var cap = Number(snap.max_inflight_per_hotkey || 0);
    var byHotkey = snap.depth_by_hotkey || {};
    var minerCount = Object.keys(byHotkey).length;
    var maxNetCap = cap > 0 ? cap * Math.max(minerCount, 1) : 0;
    depthBigEl.textContent = depth + (maxNetCap > 0 ? " / " + maxNetCap : "");
    depthCapEl.textContent = cap > 0
        ? "outstanding / cap (" + cap + " per miner * " + minerCount + " miners)"
        : "outstanding entries";

    var atCap = Number(snap.at_cap_count || 0);
    var bpFraction = minerCount > 0 ? (atCap / minerCount) : 0;
    var bpStr = (bpFraction * 100).toFixed(0) + "% (" + atCap + "/" + minerCount + " miners)";
    bpEl.textContent = bpStr;
    bpEl.className = bpFraction >= 0.5 ? "alert" : "";

    if (snap.oldest_entry_age_sec != null) {
        oldestEl.textContent = fmtDurationSec(snap.oldest_entry_age_sec) +
            (snap.oldest_job_id ? " (" + short(snap.oldest_job_id, 20) + ")" : "");
    } else {
        oldestEl.textContent = "--";
    }

    var snapAge = snap.snapshot_unix ? Math.max(0, nowUnix - snap.snapshot_unix) : null;
    snapAgeEl.textContent = snapAge != null ? fmtDurationSec(snapAge) + " ago (id=" + (snap.snapshot_id || 0) + ")" : "--";
    metaLineEl.textContent = "id=" + (snap.snapshot_id || 0);

    // Per-miner inflight bars.
    var hotkeys = Object.keys(byHotkey).sort(function(a, b) { return byHotkey[b] - byHotkey[a]; });
    if (!hotkeys.length) {
        barsEl.innerHTML = '<div class="queue-bar"><span class="queue-bar-id">--</span><span class="queue-bar-num">no entries</span></div>';
    } else {
        barsEl.innerHTML = hotkeys.map(function(hk) {
            var d = Number(byHotkey[hk]);
            var pct = (cap > 0) ? Math.min(100, (d / cap) * 100) : Math.min(100, d * 10);
            var isAtCap = (cap > 0 && d >= cap);
            return '<div class="queue-bar' + (isAtCap ? ' at-cap' : '') + '" title="' + esc(hk) + '">' +
                '<span class="queue-bar-id"><code>' + esc(short(hk, 12)) + '</code></span>' +
                '<span class="queue-bar-track"><span class="queue-bar-fill" style="width:' + pct.toFixed(0) + '%"></span></span>' +
                '<span class="queue-bar-num">' + d + (cap > 0 ? "/" + cap : "") + '</span>' +
                '</div>';
        }).join("");
    }

    renderQueueSpark(snap, nowUnix);
}

function renderQueueSpark(snap, nowUnix) {
    var svg = document.getElementById("queue-spark");
    var ymaxEl = document.getElementById("queue-spark-y-max");
    var hist = snap.history || [];
    if (!hist.length) {
        svg.innerHTML = '<text class="queue-empty" x="500" y="40" text-anchor="middle">Queue depth history will appear within ~1 minute</text>';
        ymaxEl.textContent = "";
        return;
    }
    var W = 1000, H = 80, PAD = 6;
    var INNER_H = H - PAD * 2;
    var WINDOW_SEC = 30 * 60;
    var startUnix = nowUnix - WINDOW_SEC;
    var maxDepth = 0;
    for (var i = 0; i < hist.length; i++) {
        if (hist[i].depth_total > maxDepth) maxDepth = hist[i].depth_total;
    }
    if (maxDepth <= 0) maxDepth = 1;
    function xFor(ts) { return ((ts - startUnix) / WINDOW_SEC) * W; }
    function yFor(d) { return PAD + INNER_H - (d / maxDepth) * INNER_H; }
    var pts = hist.map(function(p) { return xFor(p.ts).toFixed(2) + "," + yFor(p.depth_total).toFixed(2); });
    var lineStr = pts.join(" ");
    var areaStr = "0," + (H - PAD) + " " + lineStr + " " + W + "," + (H - PAD);
    svg.innerHTML =
        '<polygon class="queue-spark-area" points="' + areaStr + '"/>' +
        '<line class="queue-spark-baseline" x1="0" x2="' + W + '" y1="' + (H - PAD) + '" y2="' + (H - PAD) + '"/>' +
        '<polyline class="queue-spark-line" points="' + lineStr + '"/>';
    ymaxEl.textContent = "MAX " + maxDepth;
}

function renderHero() {
    var meta = (state.snapshot && state.snapshot.meta) || {};
    var health = meta.health || {};
    var chain = health.chain || {};
    var states = health.states || {};
    var parts = [];
    if (meta.netuid != null) parts.push("NETUID " + meta.netuid);
    if (meta.bucket) parts.push("BUCKET " + meta.bucket);
    if (meta.run_id) parts.push("RUN " + short(meta.run_id, 36));
    if (meta.source) parts.push("SOURCE " + meta.source);
    if (chain.current_block != null) parts.push("BLOCK " + chain.current_block);
    if (states.bucket && states.bucket.updated_unix) parts.push("BUCKET SCAN " + ageLabel(Math.max(0, meta.generated_unix - states.bucket.updated_unix)));
    if (states.chain && states.chain.updated_unix) parts.push("CHAIN SCAN " + ageLabel(Math.max(0, meta.generated_unix - states.chain.updated_unix)));
    var el = document.getElementById("hero-meta");
    if (el) el.textContent = parts.length ? parts.join(" \u00B7 ") : "--";
}

function fmtCompute(n) {
    n = Number(n || 0);
    if (n >= 1000) return (n / 1000).toFixed(1) + "k";
    if (n >= 100) return n.toFixed(0);
    return n.toFixed(1);
}

function fmtMetricValue(metricId, n) {
    n = Number(n || 0);
    if (metricId === "bandwidth") return fmtBytes(n) + "/s";
    if (metricId === "jobs") {
        if (n >= 10) return n.toFixed(1) + " jobs/s";
        if (n >= 1) return n.toFixed(2) + " jobs/s";
        return n.toFixed(3) + " jobs/s";
    }
    if (metricId === "latency") return fmtDurationSec(n);
    return fmtCompute(n) + " CU/s";
}

function fmtMetricTotal(metricId, total, count) {
    total = Number(total || 0);
    count = Number(count || 0);
    if (metricId === "bandwidth") return fmtBytes(total) + " over 30m";
    if (metricId === "jobs") return count + " jobs over 30m";
    if (metricId === "latency") return count ? "avg " + fmtDurationSec(total / count) + " over 30m" : "avg -- over 30m";
    return fmtCompute(total) + " CU over 30m";
}

function graphMetricDef() {
    for (var i = 0; i < GRAPH_METRICS.length; i++) {
        if (GRAPH_METRICS[i].id === state.graphMetric) return GRAPH_METRICS[i];
    }
    return GRAPH_METRICS[0];
}

function renderGraphTabs() {
    var tabs = document.getElementById("graph-tabs");
    if (!tabs) return;
    tabs.innerHTML = GRAPH_METRICS.map(function(metric) {
        var cls = "metric-tab" + (metric.id === state.graphMetric ? " active" : "");
        return '<button type="button" class="' + cls + '" data-graph-metric="' + esc(metric.id) + '">' +
            esc(metric.label) + '</button>';
    }).join("");
}

function buildGraphSeries(jobs, nowUnix, metricId) {
    var totals = new Array(COMPUTE_BINS);
    var counts = new Array(COMPUTE_BINS);
    for (var b = 0; b < COMPUTE_BINS; b++) {
        totals[b] = 0;
        counts[b] = 0;
    }
    var total = 0;
    var count = 0;

    for (var i = 0; i < jobs.length; i++) {
        var j = jobs[i];
        var finished = Number(j.finished_unix);
        if (!finished) continue;
        var ageSec = nowUnix - finished;
        if (ageSec < 0 || ageSec >= COMPUTE_WINDOW_SEC) continue;
        var idx = COMPUTE_BINS - 1 - Math.floor(ageSec / COMPUTE_BIN_SEC);
        if (idx < 0 || idx >= COMPUTE_BINS) continue;

        var value = 0;
        if (metricId === "bandwidth") {
            value = Number(j.bytes_read || 0) + Number(j.bytes_written || 0);
        } else if (metricId === "jobs") {
            value = 1;
        } else if (metricId === "latency") {
            value = Number(j.duration_sec || 0);
        } else {
            value = Number(j.compute_sec || 0);
        }
        if (!isFinite(value) || value <= 0) continue;
        totals[idx] += value;
        counts[idx] += 1;
        total += value;
        count += 1;
    }

    var rates = new Array(COMPUTE_BINS);
    for (var k = 0; k < COMPUTE_BINS; k++) {
        if (metricId === "latency") {
            rates[k] = counts[k] ? totals[k] / counts[k] : 0;
        } else {
            rates[k] = totals[k] / COMPUTE_BIN_SEC;
        }
    }
    return { rates: rates, total: total, count: count };
}

function renderCompute() {
    renderGraphTabs();
    var svg = document.getElementById("compute-chart");
    if (!svg) return;
    var jobs = completedJobs();
    var meta = (state.snapshot && state.snapshot.meta) || {};
    var nowUnix = Number(meta.generated_unix) || (Date.now() / 1000);
    var metric = graphMetricDef();
    var series = buildGraphSeries(jobs, nowUnix, metric.id);
    var rates = series.rates;
    var smoothRates = smoothSeries(rates, 5, 0.35);
    var maxRate = 0;
    for (var k2 = 0; k2 < COMPUTE_BINS; k2++) {
        if (rates[k2] > maxRate) maxRate = rates[k2];
        if (smoothRates[k2] > maxRate) maxRate = smoothRates[k2];
    }

    var W = 1000, H = 140, PAD_TOP = 12, PAD_BOTTOM = 14;
    var INNER_H = H - PAD_TOP - PAD_BOTTOM;
    var stepX = (COMPUTE_BINS > 1) ? W / (COMPUTE_BINS - 1) : 0;
    var baselineY = H - PAD_BOTTOM;

    function yFor(rate) {
        if (maxRate <= 0) return baselineY;
        return PAD_TOP + INNER_H - (rate / maxRate) * INNER_H;
    }

    var rawPts = [];
    var linePts = [];
    for (var p = 0; p < COMPUTE_BINS; p++) {
        var x = (p * stepX).toFixed(2);
        rawPts.push(x + "," + yFor(rates[p]).toFixed(2));
        var y = yFor(smoothRates[p]).toFixed(2);
        linePts.push(x + "," + y);
    }
    var rawLineStr = rawPts.join(" ");
    var lineStr = linePts.join(" ");
    var areaStr = "0," + baselineY + " " + lineStr + " " + W + "," + baselineY;

    var emptyEl = "";
    if (series.count === 0) {
        emptyEl = '<text class="compute-empty" x="500" y="' + (PAD_TOP + INNER_H / 2) +
            '" text-anchor="middle">' + esc(metric.empty) + '</text>';
    }

    svg.innerHTML =
        '<polygon class="compute-area" points="' + areaStr + '"/>' +
        '<line class="compute-baseline" x1="0" x2="' + W + '" y1="' + baselineY + '" y2="' + baselineY + '"/>' +
        '<polyline class="compute-line-raw" points="' + rawLineStr + '"/>' +
        '<polyline class="compute-line" points="' + lineStr + '"/>' +
        '<line id="compute-cursor-line" class="compute-cursor" x1="0" x2="0" y1="' + PAD_TOP +
        '" y2="' + baselineY + '" style="display:none"/>' +
        emptyEl;

    var rateBadge = document.getElementById("compute-rate");
    if (rateBadge) {
        var lastRate = smoothRates[smoothRates.length - 1] || 0;
        rateBadge.textContent = fmtMetricValue(metric.id, lastRate) + " \u00B7 " + fmtMetricTotal(metric.id, series.total, series.count);
    }

    var ymax = document.getElementById("compute-y-max");
    if (ymax) ymax.textContent = maxRate > 0 ? fmtMetricValue(metric.id, maxRate) : "";

    state.computeRates = smoothRates;
    state.graphRawRates = rates;
    state.graphMetricDef = metric;
    state.computeNowUnix = nowUnix;
}

function smoothSeries(values, windowSize, alpha) {
    var out = new Array(values.length);
    var ema = 0;
    for (var i = 0; i < values.length; i++) {
        var start = Math.max(0, i - Math.floor(windowSize / 2));
        var end = Math.min(values.length - 1, i + Math.floor(windowSize / 2));
        var total = 0;
        var n = 0;
        for (var j = start; j <= end; j++) {
            total += Number(values[j] || 0);
            n += 1;
        }
        var ma = n ? total / n : 0;
        ema = i === 0 ? ma : (alpha * ma + (1 - alpha) * ema);
        out[i] = ema;
    }
    return out;
}

function attachComputeHover() {
    var wrap = document.getElementById("compute-chart-wrap");
    var svg = document.getElementById("compute-chart");
    var tooltip = document.getElementById("compute-tooltip");
    if (!wrap || !svg || !tooltip) return;

    function hide() {
        var cursor = document.getElementById("compute-cursor-line");
        if (cursor) cursor.style.display = "none";
        tooltip.style.display = "none";
    }

    wrap.addEventListener("mousemove", function(e) {
        if (!state.computeRates) { hide(); return; }
        var rect = svg.getBoundingClientRect();
        var x = e.clientX - rect.left;
        if (x < 0 || x > rect.width || rect.width <= 0) { hide(); return; }
        var binIdx = Math.floor((x / rect.width) * COMPUTE_BINS);
        if (binIdx < 0) binIdx = 0;
        if (binIdx > COMPUTE_BINS - 1) binIdx = COMPUTE_BINS - 1;
        var rate = state.computeRates[binIdx] || 0;
        var nowUnix = state.computeNowUnix || (Date.now() / 1000);
        var binEndUnix = nowUnix - (COMPUTE_BINS - 1 - binIdx) * COMPUTE_BIN_SEC;
        var binStartUnix = binEndUnix - COMPUTE_BIN_SEC;

        var cursor = document.getElementById("compute-cursor-line");
        if (cursor) {
            var vbX = (COMPUTE_BINS > 1) ? (binIdx / (COMPUTE_BINS - 1)) * 1000 : 0;
            cursor.setAttribute("x1", vbX);
            cursor.setAttribute("x2", vbX);
            cursor.style.display = "";
        }

        var metric = state.graphMetricDef || graphMetricDef();
        var raw = state.graphRawRates ? (state.graphRawRates[binIdx] || 0) : rate;
        tooltip.textContent = fmtTime(binStartUnix) + " \u00B7 " + fmtMetricValue(metric.id, rate) +
            " smoothed \u00B7 " + fmtMetricValue(metric.id, raw) + " raw";
        tooltip.style.display = "block";
        var tRect = tooltip.getBoundingClientRect();
        var wrapRect = wrap.getBoundingClientRect();
        var relX = e.clientX - wrapRect.left;
        var tx = relX + 10;
        if (tx + tRect.width > wrapRect.width) tx = relX - tRect.width - 10;
        if (tx < 0) tx = 0;
        tooltip.style.left = tx + "px";
        tooltip.style.top = "2px";
    });
    wrap.addEventListener("mouseleave", hide);
}

function renderAll() {
    if (!state.snapshot) return;
    renderHero();
    renderQueue();
    renderCompute();
    renderMiners();
    renderOutstandingJobs();
    renderJobs();
}

function showError(message) {
    var el = document.getElementById("error");
    if (!message) { el.style.display = "none"; return; }
    el.style.display = "block";
    el.textContent = "ERROR \u2014 " + message;
}

async function poll() {
    try {
        var qs = window.location.search || "";
        var res = await fetch("/api/snapshot" + qs);
        if (!res.ok) throw new Error("HTTP " + res.status);
        state.snapshot = await res.json();
        showError("");
        renderAll();
    } catch (e) {
        showError(e.message || String(e));
    }
}

(function initThemeToggle() {
    var btn = document.getElementById("theme-toggle");
    function updateLabel() {
        btn.textContent = document.documentElement.dataset.theme === "dark" ? "LIGHT" : "DARK";
    }
    updateLabel();
    btn.addEventListener("click", function() {
        var isDark = document.documentElement.dataset.theme === "dark";
        document.documentElement.dataset.theme = isDark ? "light" : "dark";
        localStorage.setItem("theme", document.documentElement.dataset.theme);
        updateLabel();
        renderAll();
    });
})();

document.getElementById("jobs-filter").addEventListener("click", function(e) {
    var btn = e.target.closest && e.target.closest("[data-filter]");
    if (!btn) return;
    var next = btn.getAttribute("data-filter");
    if (!next || next === state.jobsFilter) return;
    state.jobsFilter = next;
    try { localStorage.setItem("jobsFilter", next); } catch (_) {}
    renderJobs();
});

document.getElementById("miners-filter").addEventListener("click", function(e) {
    var btn = e.target.closest && e.target.closest("[data-filter]");
    if (!btn) return;
    var next = btn.getAttribute("data-filter");
    if (!next || next === state.minersFilter) return;
    state.minersFilter = next;
    try { localStorage.setItem("minersFilter", next); } catch (_) {}
    renderMiners();
});

document.getElementById("graph-tabs").addEventListener("click", function(e) {
    var btn = e.target.closest && e.target.closest("[data-graph-metric]");
    if (!btn) return;
    var next = btn.getAttribute("data-graph-metric");
    if (!next || next === state.graphMetric) return;
    state.graphMetric = next;
    try { localStorage.setItem("graphMetric", next); } catch (_) {}
    renderCompute();
});

document.querySelector("thead").addEventListener("click", function(e) {
    var th = e.target.closest && e.target.closest("[data-miner-sort]");
    if (!th) return;
    var key = th.getAttribute("data-miner-sort");
    if (!key) return;
    if (state.minersSort.key === key) {
        state.minersSort.dir = state.minersSort.dir === "asc" ? "desc" : "asc";
    } else {
        state.minersSort.key = key;
        state.minersSort.dir = key === "ping" || key === "identity" || key === "status" ? "asc" : "desc";
    }
    try {
        localStorage.setItem("minersSortKey", state.minersSort.key);
        localStorage.setItem("minersSortDir", state.minersSort.dir);
    } catch (_) {}
    renderMiners();
});

renderJobsFilter({});
renderMinersFilter({});
renderCompute();
attachComputeHover();
poll();
setInterval(poll, POLL_MS);
</script>
</body>
</html>
"""


def serve_discovery_ui(
    *,
    bucket: ObjectStore,
    netuid: int,
    run_id: str | None = None,
    heartbeat_ttl_sec: float | None = 30.0,
    refresh_sec: float = 3.0,
    snapshot_cache_sec: float = 1.5,
    max_jobs: int = 500,
    max_artifacts: int = 300,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
) -> None:
    handler = _handler(
        bucket=bucket,
        netuid=netuid,
        run_id=run_id,
        heartbeat_ttl_sec=heartbeat_ttl_sec,
        refresh_sec=refresh_sec,
        snapshot_cache_sec=snapshot_cache_sec,
        max_jobs=max_jobs,
        max_artifacts=max_artifacts,
    )
    server = ThreadingHTTPServer((host, int(port)), handler)
    url = f"http://{host}:{port}/"
    print(f"[discovery-ui] serving {url} netuid={netuid} bucket={bucket.bucket}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[discovery-ui] stopped")
    finally:
        server.server_close()


def _handler(
    *,
    bucket: ObjectStore,
    netuid: int,
    run_id: str | None,
    heartbeat_ttl_sec: float | None,
    refresh_sec: float,
    snapshot_cache_sec: float,
    max_jobs: int,
    max_artifacts: int,
) -> type[BaseHTTPRequestHandler]:
    cache: dict[tuple[Any, ...], tuple[float, dict[str, Any]]] = {}

    logo_bytes = _load_logo_bytes()

    class DiscoveryHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if parsed.path == "/":
                self._send(HTTPStatus.OK, _index_html(refresh_sec).encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path in {"/teutonic.png", "/favicon.png", "/favicon.ico"}:
                if logo_bytes:
                    self._send(HTTPStatus.OK, logo_bytes, "image/png")
                else:
                    self._send(HTTPStatus.NOT_FOUND, b"", "image/png")
                return
            if parsed.path == "/api/discovery":
                self._send_json(discovery_payload(bucket, netuid=netuid, default_run_id=run_id, heartbeat_ttl_sec=heartbeat_ttl_sec, query=query))
                return
            if parsed.path == "/api/runs":
                self._send_json({"runs": discover_run_ids(bucket, netuid=netuid), "default_run_id": run_id})
                return
            if parsed.path == "/api/snapshot":
                selected_run = _first(query, "run_id") or run_id
                if not selected_run:
                    runs = discover_run_ids(bucket, netuid=netuid, limit=1)
                    selected_run = runs[0] if runs else ""
                if not selected_run:
                    self._send_json(_empty_snapshot(bucket=bucket, netuid=netuid, heartbeat_ttl_sec=heartbeat_ttl_sec, refresh_sec=refresh_sec))
                    return
                key = ("snapshot", selected_run, max_jobs, max_artifacts, heartbeat_ttl_sec)
                self._send_json(
                    _cached(
                        cache,
                        key,
                        snapshot_cache_sec,
                        lambda: visualizer_snapshot(
                            bucket,
                            netuid=netuid,
                            run_id=selected_run,
                            config=VisualizerConfig(
                                max_jobs=max_jobs,
                                max_artifacts=max_artifacts,
                                heartbeat_ttl_sec=heartbeat_ttl_sec,
                            ),
                        ),
                    )
                )
                return
            if parsed.path == "/api/job":
                selected_run = _first(query, "run_id") or run_id
                job_id = _first(query, "job_id")
                if not selected_run or not job_id:
                    self._send(HTTPStatus.BAD_REQUEST, b"run_id and job_id are required", "text/plain; charset=utf-8")
                    return
                self._send_json(
                    job_detail(
                        bucket,
                        netuid=netuid,
                        run_id=selected_run,
                        job_id=job_id,
                        config=VisualizerConfig(max_jobs=max_jobs, max_artifacts=max_artifacts, heartbeat_ttl_sec=heartbeat_ttl_sec),
                    )
                )
                return
            if parsed.path == "/api/artifact":
                uri = _first(query, "uri")
                if not uri:
                    self._send(HTTPStatus.BAD_REQUEST, b"uri is required", "text/plain; charset=utf-8")
                    return
                self._send_json(artifact_metadata(bucket, uri=uri))
                return
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain; charset=utf-8")

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, value: dict[str, Any]) -> None:
            self._send(HTTPStatus.OK, json.dumps(value, sort_keys=True).encode("utf-8"), "application/json")

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return DiscoveryHandler


def discovery_payload(
    bucket: ObjectStore,
    *,
    netuid: int,
    default_run_id: str | None = None,
    heartbeat_ttl_sec: float | None = 30.0,
    query: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    query = query or {}
    role = _first(query, "role") or "all"
    if role not in {"all", "train", "audit"}:
        role = "all"
    run_id = _first(query, "run_id") or default_run_id
    roles = ["train", "audit"] if role == "all" else [role]
    records: list[DiscoveryRecord] = []
    for item in roles:
        records.extend(
            scan_bucket_discovery_records(
                bucket,
                netuid=netuid,
                role=item,
                run_id=run_id,
                heartbeat_ttl_sec=heartbeat_ttl_sec,
            )
        )
    now = time.time()
    rows = [_record_to_dict(record, now=now) for record in records]
    rows.sort(key=lambda r: (r["role"], r["run_id"], r["worker"]["host_id"], r["worker"]["worker_id"]))
    return {
        "meta": {
            "bucket": bucket.bucket,
            "netuid": int(netuid),
            "run_id": run_id,
            "role": role,
            "heartbeat_ttl_sec": heartbeat_ttl_sec,
            "generated_unix": int(now),
        },
        "records": rows,
    }


def _index_html(refresh_sec: float) -> str:
    return INDEX_HTML.replace("__REFRESH_MS__", str(max(500, int(refresh_sec * 1000))))


def _cached(
    cache: dict[tuple[Any, ...], tuple[float, dict[str, Any]]],
    key: tuple[Any, ...],
    ttl_sec: float,
    build,
) -> dict[str, Any]:
    now = time.time()
    cached = cache.get(key)
    if cached is not None and now - cached[0] < ttl_sec:
        return cached[1]
    value = build()
    cache[key] = (now, value)
    return value


def _empty_snapshot(*, bucket: ObjectStore, netuid: int, heartbeat_ttl_sec: float | None, refresh_sec: float) -> dict[str, Any]:
    now = int(time.time())
    return {
        "meta": {
            "bucket": bucket.bucket,
            "netuid": int(netuid),
            "run_id": "",
            "generated_unix": now,
            "heartbeat_ttl_sec": heartbeat_ttl_sec,
            "refresh_sec": refresh_sec,
        },
        "run": {"run_id": ""},
        "machines": [],
        "jobs": [],
        "artifacts": [],
        "edges": [],
        "summary": {
            "machines": 0,
            "workers": 0,
            "jobs": 0,
            "in_flight_jobs": 0,
            "completed_jobs": 0,
            "failed_or_stale_jobs": 0,
            "bytes_read": 0,
            "bytes_written": 0,
            "artifacts": 0,
            "present_artifacts": 0,
            "missing_artifacts": 0,
            "audits": {"pending": 0, "pass": 0, "fail": 0},
            "by_status": {},
            "by_kind": {},
            "by_worker": {},
        },
    }


def _record_to_dict(record: DiscoveryRecord, *, now: float) -> dict[str, Any]:
    return {
        "miner": record.miner.to_dict(),
        "worker": record.worker.to_dict(),
        "run_id": record.run_id,
        "role": record.role,
        "last_seen_unix": record.last_seen_unix,
        "age_sec": max(0.0, now - record.last_seen_unix),
    }


def _first(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key) or []
    return values[0] if values and values[0] else None
