import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createReadStream } from "node:fs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");
loadEnv(path.join(root, ".env.local"));

const HOST = process.env.HOST || "127.0.0.1";
const PORT = Number(process.env.PORT || 8787);
const PI_URL = (process.env.PI_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const PI_TOKEN = process.env.PI_TOKEN || "";
const POLL_MS = Number(process.env.POLL_MS || 500);
const LOCAL_MEASURE_DIR = process.env.LOCAL_MEASURE_DIR || "";
const DIST = path.join(root, "dist");
const UPSTREAM_TIMEOUT_MS = Number(process.env.UPSTREAM_TIMEOUT_MS || 2000);
const UPSTREAM_MUTATE_TIMEOUT_MS = Number(process.env.UPSTREAM_MUTATE_TIMEOUT_MS || 60000);

/** @typedef {"ok"|"pi_unreachable"|"auth_failed"|"controller_down"} LinkState */

const state = {
  /** @type {LinkState} */
  link: "pi_unreachable",
  telemetry: null,
  health: null,
  lastError: null,
  updatedAt: 0,
  trialsSource: "pi",
};

/** @type {Set<http.ServerResponse>} */
const sseClients = new Set();

const GET_ALLOW = new Set([
  "/api/health",
  "/api/telemetry",
  "/api/logs",
  "/api/devices",
  "/api/trials",
  "/api/stream",
]);
const POST_ALLOW = new Set([
  "/api/start",
  "/api/stop",
  "/api/restart",
  "/api/taps",
]);

function loadEnv(file) {
  try {
    const text = fs.readFileSync(file, "utf8");
    for (const line of text.split(/\r?\n/)) {
      const t = line.trim();
      if (!t || t.startsWith("#")) continue;
      const i = t.indexOf("=");
      if (i < 0) continue;
      const k = t.slice(0, i).trim();
      let v = t.slice(i + 1).trim();
      if (
        (v.startsWith('"') && v.endsWith('"')) ||
        (v.startsWith("'") && v.endsWith("'"))
      ) {
        v = v.slice(1, -1);
      }
      if (!(k in process.env)) process.env[k] = v;
    }
  } catch {
    /* optional */
  }
}

function json(res, status, body) {
  const payload = JSON.stringify(body);
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
    "Content-Length": Buffer.byteLength(payload),
  });
  res.end(payload);
}

async function piFetch(pathname, opts = {}) {
  const ctrl = new AbortController();
  const timeoutMs =
    typeof opts.timeoutMs === "number" ? opts.timeoutMs : UPSTREAM_TIMEOUT_MS;
  const { timeoutMs: _ignored, ...fetchOpts } = opts;
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const headers = { ...(fetchOpts.headers || {}) };
    if (PI_TOKEN) headers.Authorization = `Bearer ${PI_TOKEN}`;
    if (fetchOpts.body && !headers["Content-Type"]) {
      headers["Content-Type"] = "application/json";
    }
    const r = await fetch(`${PI_URL}${pathname}`, {
      ...fetchOpts,
      headers,
      signal: ctrl.signal,
    });
    const text = await r.text();
    let data = null;
    try {
      data = text ? JSON.parse(text) : null;
    } catch {
      data = { raw: text };
    }
    return { ok: r.ok, status: r.status, data };
  } finally {
    clearTimeout(timer);
  }
}

function classifyUpstream(err, status) {
  if (status === 401 || status === 403) return "auth_failed";
  if (err) return "pi_unreachable";
  if (status >= 500) return "controller_down";
  return "ok";
}

async function pollOnce() {
  try {
    const health = await piFetch("/health");
    if (!health.ok && health.status === 0) {
      throw new Error("unreachable");
    }
    state.health = health.data;
    if (health.status === 401 || health.status === 403) {
      state.link = "auth_failed";
      state.lastError = "auth";
    } else if (!health.ok) {
      state.link = "controller_down";
      state.lastError = `health ${health.status}`;
    } else {
      try {
        const tel = await piFetch("/telemetry");
        if (tel.status === 401 || tel.status === 403) {
          state.link = "auth_failed";
        } else if (tel.status === 404) {
          // Planned endpoint missing — fall back to /status shape
          const st = await piFetch("/status");
          state.link = classifyUpstream(null, st.status);
          state.telemetry = synthesizeFromStatus(st.data);
        } else if (!tel.ok) {
          state.link = classifyUpstream(null, tel.status);
          state.lastError = `telemetry ${tel.status}`;
        } else {
          state.link = "ok";
          state.telemetry = tel.data;
          state.lastError = null;
        }
      } catch (e) {
        state.link = "pi_unreachable";
        state.lastError = String(e);
      }
    }
  } catch (e) {
    state.link = "pi_unreachable";
    state.lastError = String(e && e.message ? e.message : e);
  }
  state.updatedAt = Date.now();
  broadcast({
    type: "telemetry",
    link: state.link,
    updatedAt: state.updatedAt,
    telemetry: state.telemetry,
    health: state.health,
    trialsSource: state.trialsSource,
    lastError: state.lastError,
  });
}

function synthesizeFromStatus(status) {
  if (!status || typeof status !== "object") return null;
  return {
    source: status.running ? "owned" : "idle",
    runtime: status.runtime || "unknown",
    host: status.host || "pi",
    seq: status.seq || 0,
    last_profile: status.last_profile || null,
    last_monitor: status.last_monitor || null,
    taps: status.taps || { armed: false, select: "", truncated: false },
    trials_tail: status.trials_tail || [],
    config: status.config && typeof status.config === "object" ? status.config : undefined,
    _note: "synthesized from /status; GET /telemetry not available yet",
  };
}

function broadcast(obj) {
  const line = `data: ${JSON.stringify(obj)}\n\n`;
  for (const res of sseClients) {
    try {
      res.write(line);
    } catch {
      sseClients.delete(res);
    }
  }
}

async function readLocalTrials(runId, since) {
  if (!LOCAL_MEASURE_DIR) return { ok: false, status: 404, data: { error: "LOCAL_MEASURE_DIR unset" } };
  const dir = runId
    ? path.join(LOCAL_MEASURE_DIR, runId)
    : LOCAL_MEASURE_DIR;
  const file = path.join(dir, "telemetry.jsonl");
  if (!fs.existsSync(file)) {
    // Also try measure/<run>/telemetry.jsonl listing
    if (!runId && fs.existsSync(LOCAL_MEASURE_DIR)) {
      const runs = fs
        .readdirSync(LOCAL_MEASURE_DIR, { withFileTypes: true })
        .filter((d) => d.isDirectory())
        .map((d) => d.name)
        .sort();
      const latest = runs[runs.length - 1];
      if (latest) return readLocalTrials(latest, since);
    }
    return { ok: false, status: 404, data: { error: "telemetry.jsonl not found", path: file } };
  }
  const text = fs.readFileSync(file, "utf8");
  const sinceN = Number(since || 0);
  const records = [];
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim()) continue;
    try {
      const row = JSON.parse(line);
      if (sinceN && Number(row.seq || row.t || 0) <= sinceN) continue;
      records.push(row);
    } catch {
      /* skip bad line */
    }
  }
  state.trialsSource = "local";
  return { ok: true, status: 200, data: { run_id: runId || null, source: "local", records } };
}

function contentType(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  return (
    {
      ".html": "text/html; charset=utf-8",
      ".js": "text/javascript; charset=utf-8",
      ".css": "text/css; charset=utf-8",
      ".json": "application/json",
      ".svg": "image/svg+xml",
      ".png": "image/png",
      ".ico": "image/x-icon",
      ".woff2": "font/woff2",
    }[ext] || "application/octet-stream"
  );
}

function serveStatic(req, res, urlPath) {
  let rel = decodeURIComponent(urlPath.split("?")[0]);
  if (rel === "/" || rel === "") rel = "/index.html";
  const filePath = path.normalize(path.join(DIST, rel));
  if (!filePath.startsWith(DIST)) {
    json(res, 403, { error: "forbidden" });
    return;
  }
  if (!fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
    const fallback = path.join(DIST, "index.html");
    if (fs.existsSync(fallback)) {
      res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
      createReadStream(fallback).pipe(res);
      return;
    }
    json(res, 404, { error: "ui not built; run npm run build" });
    return;
  }
  res.writeHead(200, { "Content-Type": contentType(filePath) });
  createReadStream(filePath).pipe(res);
}

async function readBody(req) {
  const chunks = [];
  for await (const c of req) chunks.push(c);
  const buf = Buffer.concat(chunks);
  if (!buf.length) return null;
  try {
    return JSON.parse(buf.toString("utf8"));
  } catch {
    return buf.toString("utf8");
  }
}

const server = http.createServer(async (req, res) => {
  const host = req.headers.host || `${HOST}:${PORT}`;
  const url = new URL(req.url || "/", `http://${host}`);
  const pathname = url.pathname;
  const method = (req.method || "GET").toUpperCase();

  if (pathname.startsWith("/api/")) {
    if (method === "GET" && !GET_ALLOW.has(pathname)) {
      return json(res, 404, { error: "not allowlisted" });
    }
    if (method === "POST" && !POST_ALLOW.has(pathname)) {
      return json(res, 404, { error: "not allowlisted" });
    }
    if (method !== "GET" && method !== "POST") {
      return json(res, 405, { error: "method not allowed" });
    }

    if (pathname === "/api/stream" && method === "GET") {
      res.writeHead(200, {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
      });
      res.write(`data: ${JSON.stringify({
        type: "telemetry",
        link: state.link,
        updatedAt: state.updatedAt,
        telemetry: state.telemetry,
        health: state.health,
        trialsSource: state.trialsSource,
        lastError: state.lastError,
      })}\n\n`);
      sseClients.add(res);
      req.on("close", () => sseClients.delete(res));
      return;
    }

    if (pathname === "/api/health" && method === "GET") {
      return json(res, 200, {
        proxy: "ok",
        link: state.link,
        pi_url: PI_URL,
        updatedAt: state.updatedAt,
        health: state.health,
        lastError: state.lastError,
      });
    }

    if (pathname === "/api/telemetry" && method === "GET") {
      return json(res, 200, {
        link: state.link,
        updatedAt: state.updatedAt,
        telemetry: state.telemetry,
        trialsSource: state.trialsSource,
        lastError: state.lastError,
      });
    }

    if (pathname === "/api/trials" && method === "GET") {
      const runId = url.searchParams.get("run_id") || "";
      const since = url.searchParams.get("since") || "";
      // Prefer Pi when link ok; else local measure dir (Windows testbench)
      if (state.link === "ok") {
        try {
          const q = new URLSearchParams();
          if (runId) q.set("run_id", runId);
          if (since) q.set("since", since);
          const qs = q.toString() ? `?${q}` : "";
          const upstream = await piFetch(`/trials${qs}`);
          if (upstream.ok) {
            state.trialsSource = "pi";
            return json(res, 200, { source: "pi", ...normalizeTrials(upstream.data) });
          }
        } catch {
          /* fall through to local */
        }
      }
      const local = await readLocalTrials(runId, since);
      return json(res, local.status, local.data);
    }

    // Passthrough allowlisted Pi routes
    const map = {
      "/api/logs": "/logs",
      "/api/devices": "/devices",
      "/api/start": "/start",
      "/api/stop": "/stop",
      "/api/restart": "/restart",
      "/api/taps": "/taps",
    };
    const piPath = map[pathname];
    if (!piPath) return json(res, 404, { error: "unknown" });

    try {
      const body = method === "POST" ? await readBody(req) : null;
      const qs = url.search || "";
      const mutate =
        method === "POST" &&
        (piPath === "/start" || piPath === "/stop" || piPath === "/restart" || piPath === "/taps");
      const upstream = await piFetch(`${piPath}${qs}`, {
        method,
        body: body == null ? undefined : JSON.stringify(body),
        timeoutMs: mutate ? UPSTREAM_MUTATE_TIMEOUT_MS : UPSTREAM_TIMEOUT_MS,
      });
      return json(res, upstream.status || 502, upstream.data ?? { error: "empty" });
    } catch (e) {
      return json(res, 502, { error: String(e && e.message ? e.message : e), link: "pi_unreachable" });
    }
  }

  serveStatic(req, res, pathname);
});

function normalizeTrials(data) {
  if (Array.isArray(data)) return { records: data };
  if (data && Array.isArray(data.records)) return data;
  return { records: data ? [data] : [] };
}

server.listen(PORT, HOST, () => {
  console.log(`Sonitude dashboard proxy http://${HOST}:${PORT}`);
  console.log(`Upstream PI_URL=${PI_URL} poll=${POLL_MS}ms`);
  if (!PI_TOKEN) console.warn("PI_TOKEN empty — authenticated routes will fail");
  if (LOCAL_MEASURE_DIR) console.log(`LOCAL_MEASURE_DIR=${LOCAL_MEASURE_DIR}`);
});

async function pollLoop() {
  try {
    await pollOnce();
  } finally {
    setTimeout(pollLoop, POLL_MS);
  }
}
pollLoop();
