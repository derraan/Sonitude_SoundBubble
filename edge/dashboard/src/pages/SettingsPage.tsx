import { useEffect, useState } from "react";
import { PageHeader } from "../components/PageHeader";
import { apiGet, apiPost } from "../hooks/useTelemetryStream";
import type { Telemetry } from "../types";

type AudioDevice = {
  index: number;
  name: string;
  hostapi?: string;
  max_input_channels: number;
  max_output_channels: number;
  default_samplerate?: number;
};

type DevicesResp = {
  devices?: AudioDevice[];
};

type ConfigMap = Record<string, unknown>;

const DEFAULT_TAP_DIR = "measure/dashboard"; // Pi: measure -> /mnt/sonitude-drive/sonitude-recordings (Samsung T5)
// raw_8ch is model-independent (device order, pre-gain): the stream to keep for datasets.
const DEFAULT_TAP_SELECT = "raw,mapped,dry,wet";
// edge/MIC_INDEX_CANON.md §4 (owner): left ear = USB ch5.
const CANON_CHANNEL_MAP = "5,4,3,2,1,0";
// Wideband-only bench calibration 2026-11-04 (dev ch2 −12.6 dB, ch5 −12.9 dB), linear, device order.
const DEFAULT_MIC_GAINS = "1.514,1.514,6.457,1.514,1.514,6.683";
const DEFAULT_INPUT_GAIN_DB = 15;

type ModelEntry = { key: string; label: string; hint: string; model: string; contract: string };
type RadiusKey = "1.0" | "1.5" | "2.0";
const ZOO = "edge/zoo/";
const zooModel = (key: string, label: string, hint: string, file: string): ModelEntry => ({
  key,
  label,
  hint,
  model: `${ZOO}${file}.onnx`,
  contract: `${ZOO}${file}.runtime.json`,
});
/** Models on the Pi per bubble radius. First entry = default for that radius. */
const MODELS: Record<RadiusKey, ModelEntry[]> = {
  "1.5": [
    zooModel("v1", "V1 · EP35", "Production default. Deepest far rejection; darker speech.", "optim_sonitude_dist_count_1_5m_ep35"),
    zooModel("v2", "V2 · EP25", "v2 epoch 25: keeps more speech (close, ±60° elevation, behind); brighter.", "optim_v2_ep25_eval_20261104"),
    zooModel("v3", "V3 · EP37", "v2 epoch 37: best on ESS/synthetic sweeps, sharper 1.5 m edge; duller speech.", "optim_v2_ep37_eval_20261105"),
  ],
  "1.0": [
    zooModel("pre1m", "Pretrain 1 m", "Synthetic-only pretrain (epoch 80). No real-data finetune; no distance/count telemetry.", "optim_pretrain_1m_20261105"),
  ],
  "2.0": [
    zooModel("pre2m", "Pretrain 2 m", "Synthetic-only pretrain (epoch 48). No real-data finetune; no distance/count telemetry.", "optim_pretrain_2m_20261105"),
  ],
};
const RADII: { key: RadiusKey; label: string }[] = [
  { key: "1.0", label: "1 m" },
  { key: "1.5", label: "1.5 m (recommended)" },
  { key: "2.0", label: "2 m" },
];
function findModel(path: unknown): { radius: RadiusKey; entry: ModelEntry } | null {
  if (typeof path !== "string") return null;
  const file = path.split("/").pop();
  for (const r of Object.keys(MODELS) as RadiusKey[]) {
    const entry = MODELS[r].find((e) => e.model.split("/").pop() === file);
    if (entry) return { radius: r, entry };
  }
  return null;
}
const TAP_STREAMS = ["raw", "mapped", "dry", "wet"] as const;
type TapStream = (typeof TAP_STREAMS)[number];

const PRESETS = {
  Stable: {
    label: "Stable",
    hint: "hops_per_io 2, max_capture_blocks 2, target_output_ahead_sec 0.016 (16 ms), max_output_ahead_sec 0.10.",
  },
  "Low latency": {
    label: "Low latency",
    hint: "hops_per_io 1, max_capture_blocks 1, target_output_ahead_sec 0.009 (9 ms), max_output_ahead_sec 0.05.",
  },
} as const;

function presetOverlay(name: keyof typeof PRESETS, _base: ConfigMap): ConfigMap {
  if (name === "Stable") {
    return {
      hops_per_io: 2,
      max_capture_blocks: 2,
      target_output_ahead_sec: 0.016,
      max_output_ahead_sec: 0.1,
    };
  }
  return {
    hops_per_io: 1,
    max_capture_blocks: 1,
    target_output_ahead_sec: 0.009,
    max_output_ahead_sec: 0.05,
  };
}

function deviceIndex(value: string): number | undefined {
  if (value === "") return undefined;
  const n = Number(value);
  return Number.isInteger(n) ? n : undefined;
}

function parseTapSelect(text: string | undefined | null): Set<TapStream> {
  const selected = new Set<TapStream>();
  for (const token of String(text || DEFAULT_TAP_SELECT).split(",")) {
    const key = token.trim() as TapStream;
    if ((TAP_STREAMS as readonly string[]).includes(key)) selected.add(key);
  }
  if (selected.size === 0) {
    selected.add("raw");
    selected.add("mapped");
    selected.add("dry");
    selected.add("wet");
  }
  return selected;
}

function formatTapSelect(selected: Set<TapStream>): string {
  return TAP_STREAMS.filter((k) => selected.has(k)).join(",");
}

/** Class B duplex defaults when telemetry has no owned config (observe/idle). */
const CLASS_B_DEFAULTS: ConfigMap = {
  model: "edge/zoo/optim_sonitude_dist_count_1_5m_ep35.onnx",
  contract_path: "edge/zoo/optim_sonitude_dist_count_1_5m_ep35.runtime.json",
  input_device: 0,
  output_device: 0,
  input_channels: 8,
  output_channels: 2,
  input_sr: 24000,
  model_sr: 24000,
  output_sr: 24000,
  silence_dbfs: -80,
  profile: true,
  monitor: true,
  mix_uart_port: "/dev/ttyACM0",
  mix_uart_baud: 115200,
};

/** Only keys the Settings restart form owns — keeps POST bodies small/stable. */
const RESTART_KEYS = [
  "model",
  "contract_path",
  "bubble_radius",
  "input_device",
  "output_device",
  "input_channels",
  "output_channels",
  "input_sr",
  "model_sr",
  "output_sr",
  "hops_per_io",
  "max_capture_blocks",
  "target_output_ahead_sec",
  "max_output_ahead_sec",
  "input_gain_db",
  "mic_gains",
  "channel_map",
  "output_gain_db",
  "silence_dbfs",
  "profile",
  "monitor",
  "mix_uart_port",
  "mix_uart_baud",
  "runtime",
  "duplex_mode",
  "tap_dir",
  "tap_select",
  "tap_max_seconds",
  "tap_max_mb",
] as const;

export type RestartTapOptions = {
  enabled: boolean;
  tapDir: string;
  tapSelect: string;
  tapMaxSeconds: number;
  tapMaxMb: number;
};

export function restartConfig(
  telemetry: Telemetry | null,
  preset: keyof typeof PRESETS,
  capture: string,
  playback: string,
  radius: number,
  gainIn: number,
  gainOut: number,
  runtime: "cpp" | "python" = "cpp",
  mode: "normal" | "rapid" = "normal",
  taps: RestartTapOptions = {
    enabled: false,
    tapDir: DEFAULT_TAP_DIR,
    tapSelect: DEFAULT_TAP_SELECT,
    tapMaxSeconds: 120,
    tapMaxMb: 1024,
  },
  extra: ConfigMap = {},
): { config: ConfigMap } {
  const owned = telemetry?.config ?? {};
  const merged: ConfigMap = { ...CLASS_B_DEFAULTS, ...owned };
  const overlay = presetOverlay(preset, merged);
  const draft: ConfigMap = {
    ...merged,
    ...overlay,
    bubble_radius: radius,
    input_gain_db: gainIn,
    output_gain_db: gainOut,
    runtime,
    duplex_mode: mode === "rapid" ? "worker" : "off",
    // Class B lock — never inherit a stale 48 kHz owned/observe default.
    input_sr: 24000,
    model_sr: 24000,
    output_sr: 24000,
    monitor: true,
    profile: true,
    ...extra,
  };
  const input = deviceIndex(capture);
  const output = deviceIndex(playback);
  if (input !== undefined) draft.input_device = input;
  if (output !== undefined) draft.output_device = output;

  if (taps.enabled) {
    draft.tap_dir = (taps.tapDir || DEFAULT_TAP_DIR).trim() || DEFAULT_TAP_DIR;
    draft.tap_select = taps.tapSelect || DEFAULT_TAP_SELECT;
    draft.tap_max_seconds = taps.tapMaxSeconds;
    draft.tap_max_mb = taps.tapMaxMb;
  } else {
    draft.tap_dir = "";
    draft.tap_select = DEFAULT_TAP_SELECT;
    draft.tap_max_seconds = 120;
    draft.tap_max_mb = 1024;
  }

  const config: ConfigMap = {};
  for (const key of RESTART_KEYS) {
    if (draft[key] !== undefined) config[key] = draft[key];
  }
  // Prefer zoo-relative paths; absolute owned paths bloat the POST and confuse restarts.
  if (typeof config.model === "string" && config.model.includes("/edge/zoo/")) {
    config.model = "edge/zoo/" + String(config.model).split("/edge/zoo/").pop();
  }
  if (typeof config.contract_path === "string" && config.contract_path.includes("/edge/zoo/")) {
    config.contract_path = "edge/zoo/" + String(config.contract_path).split("/edge/zoo/").pop();
  }
  // Prefer repo-relative measure/… so the POST stays stable across hosts.
  if (typeof config.tap_dir === "string" && config.tap_dir.includes("/measure/")) {
    config.tap_dir = "measure/" + String(config.tap_dir).split("/measure/").pop();
  }
  return { config };
}

export function SettingsPage({ telemetry }: { telemetry: Telemetry | null }) {
  const owned = telemetry?.source === "owned";
  const [devices, setDevices] = useState<AudioDevice[]>([]);
  const [capture, setCapture] = useState("");
  const [playback, setPlayback] = useState("");
  const [radius, setRadius] = useState<RadiusKey>("1.5");
  const [modelKey, setModelKey] = useState("v1");
  const [micGains, setMicGains] = useState(DEFAULT_MIC_GAINS);
  const [channelMap, setChannelMap] = useState(CANON_CHANNEL_MAP);
  const [gainIn, setGainIn] = useState(DEFAULT_INPUT_GAIN_DB);
  const [gainOut, setGainOut] = useState(15);
  const [runtime, setRuntime] = useState<"cpp" | "python">("cpp");
  const [mode, setMode] = useState<"normal" | "rapid">("normal");
  const [preset, setPreset] = useState<keyof typeof PRESETS>("Stable");
  const [recordTaps, setRecordTaps] = useState(false);
  const [tapDir, setTapDir] = useState(DEFAULT_TAP_DIR);
  const [tapSelect, setTapSelect] = useState<Set<TapStream>>(() => parseTapSelect(DEFAULT_TAP_SELECT));
  const [tapMaxSeconds, setTapMaxSeconds] = useState(120);
  const [tapMaxMb, setTapMaxMb] = useState(1024);
  const [msg, setMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [seeded, setSeeded] = useState(false);

  useEffect(() => {
    apiGet<DevicesResp>("/api/devices")
      .then((d) => {
        const list = Array.isArray(d.devices) ? d.devices : [];
        setDevices(list);
        const caps = list.filter((dev) => dev.max_input_channels >= 8);
        const plays = list.filter((dev) => dev.max_output_channels >= 2);
        setCapture((prev) => prev || (caps[0] ? String(caps[0].index) : "0"));
        setPlayback((prev) => prev || (plays[0] ? String(plays[0].index) : "0"));
      })
      .catch(() => {
        setDevices([]);
        setCapture((prev) => prev || "0");
        setPlayback((prev) => prev || "0");
      });
  }, []);

  useEffect(() => {
    if (seeded) return;
    const cfg = telemetry?.config;
    const rt = telemetry?.runtime;
    if (rt === "cpp" || rt === "python") setRuntime(rt);
    if (!cfg) {
      // Observe/idle: keep Class B form defaults so Restart is not empty/(none)/0 dB.
      setCapture((prev) => prev || "0");
      setPlayback((prev) => prev || "0");
      return;
    }
    setSeeded(true);
    if (cfg.input_device != null && cfg.input_device !== "") setCapture(String(cfg.input_device));
    if (cfg.output_device != null && cfg.output_device !== "") setPlayback(String(cfg.output_device));
    const found = findModel(cfg.model);
    if (found) {
      setRadius(found.radius);
      setModelKey(found.entry.key);
    }
    if (typeof cfg.mic_gains === "string" && cfg.mic_gains.trim()) setMicGains(cfg.mic_gains);
    if (typeof cfg.channel_map === "string" && cfg.channel_map.trim()) setChannelMap(cfg.channel_map);
    if (typeof cfg.input_gain_db === "number") setGainIn(cfg.input_gain_db);
    if (typeof cfg.output_gain_db === "number") setGainOut(cfg.output_gain_db);
    if (cfg.runtime === "cpp" || cfg.runtime === "python") setRuntime(cfg.runtime);
    if (cfg.duplex_mode === "worker" || cfg.duplex_mode === "callback") setMode("rapid");
    else if (cfg.duplex_mode === "off" || cfg.duplex_mode === "") setMode("normal");
    const ownedTapDir = typeof cfg.tap_dir === "string" ? cfg.tap_dir.trim() : "";
    if (ownedTapDir) {
      setRecordTaps(true);
      if (ownedTapDir.includes("/measure/")) {
        setTapDir("measure/" + ownedTapDir.split("/measure/").pop());
      } else {
        setTapDir(ownedTapDir);
      }
    }
    if (typeof cfg.tap_select === "string" && cfg.tap_select.trim()) {
      setTapSelect(parseTapSelect(cfg.tap_select));
    } else if (typeof telemetry?.taps?.select === "string" && telemetry.taps.select.trim()) {
      setTapSelect(parseTapSelect(telemetry.taps.select));
    }
    if (typeof cfg.tap_max_seconds === "number") setTapMaxSeconds(cfg.tap_max_seconds);
    if (typeof cfg.tap_max_mb === "number") setTapMaxMb(cfg.tap_max_mb);
  }, [telemetry, seeded]);

  const captureOpts = withCurrent(
    devices.filter((d) => d.max_input_channels >= 8),
    capture,
  );
  const playbackOpts = withCurrent(
    devices.filter((d) => d.max_output_channels >= 2),
    playback,
  );

  const radiusModels = MODELS[radius];
  const selectedModel = radiusModels.find((e) => e.key === modelKey) ?? radiusModels[0];

  function pickRadius(r: RadiusKey) {
    setRadius(r);
    if (!MODELS[r].some((e) => e.key === modelKey)) setModelKey(MODELS[r][0].key);
  }

  function tapOptions(): RestartTapOptions {
    return {
      enabled: recordTaps,
      tapDir,
      tapSelect: formatTapSelect(tapSelect),
      tapMaxSeconds,
      tapMaxMb,
    };
  }

  async function restart() {
    setBusy(true);
    setMsg(null);
    try {
      await apiPost(
        "/api/restart",
        restartConfig(
          telemetry,
          preset,
          capture,
          playback,
          Number(radius),
          gainIn,
          gainOut,
          runtime,
          mode,
          tapOptions(),
          {
            model: selectedModel.model,
            contract_path: selectedModel.contract,
            mic_gains: micGains.trim(),
            channel_map: channelMap.trim() || CANON_CHANNEL_MAP,
          },
        ),
      );
      const tapNote = recordTaps
        ? ` · taps=${formatTapSelect(tapSelect)} → ${tapDir.trim() || DEFAULT_TAP_DIR}`
        : " · taps off";
      setMsg(
        `Restart requested · ${selectedModel.label} @ ${radius} m · runtime=${runtime} · mode=${mode}${tapNote}.`,
      );
    } catch (e) {
      setMsg(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function setTaps(action: "arm" | "disarm" | "flush") {
    setBusy(true);
    try {
      await apiPost("/api/taps", { action });
      if (action === "flush") setMsg("Tap flush requested (SIGUSR2 → WAVs on Pi).");
      else setMsg(action === "arm" ? "Taps armed." : "Taps disarmed.");
    } catch (e) {
      setMsg(String(e));
    } finally {
      setBusy(false);
    }
  }

  function toggleStream(stream: TapStream) {
    setTapSelect((prev) => {
      const next = new Set(prev);
      if (next.has(stream)) {
        if (next.size === 1) return prev;
        next.delete(stream);
      } else {
        next.add(stream);
      }
      return next;
    });
  }

  const tapsArmed = telemetry?.taps?.armed === true;
  const liveTapSelect = telemetry?.taps?.select || formatTapSelect(tapSelect);

  return (
    <div>
      <PageHeader title="Settings">
        Restart stops the production unit, then starts an owned cpp/python child with this config.
      </PageHeader>

      {!owned ? (
        <div className="card-peach mb-6 px-6 py-5">
          <p className="text-[20px] font-medium tracking-[-0.16px]">Observe mode</p>
          <p className="mt-2 max-w-3xl text-[16px] leading-snug opacity-95">
            Production <code>sonitude-soundbubble</code> is running. Press{" "}
            <strong>Restart with config</strong> to stop it and take ownership (requires one-time
            passwordless sudo for that unit).
          </p>
        </div>
      ) : null}

      <fieldset disabled={busy} className="space-y-5 disabled:opacity-45">
        <section className="card-float p-6">
          <h3 className="font-display text-[22px] tracking-[-0.2px]">Runtime</h3>
          <div className="mt-4 flex flex-wrap gap-3">
            {(["cpp", "python"] as const).map((key) => (
              <button
                key={key}
                type="button"
                onClick={() => setRuntime(key)}
                className={`pill ${runtime === key ? "pill-filled" : "pill-ghost"}`}
              >
                {key}
              </button>
            ))}
          </div>
          <p className="mt-3 text-[14px] text-slate-gray">
            Restart with config launches the selected runtime as an owned child. Flip here to A/B
            compare without touching the production unit.
          </p>
        </section>

        <section className="card-float p-6">
          <h3 className="font-display text-[22px] tracking-[-0.2px]">Mode</h3>
          <div className="mt-4 flex flex-wrap gap-3">
            {(["normal", "rapid"] as const).map((key) => (
              <button
                key={key}
                type="button"
                onClick={() => setMode(key)}
                className={`pill ${mode === key ? "pill-filled" : "pill-ghost"}`}
              >
                {key === "normal" ? "Normal" : "Rapid"}
              </button>
            ))}
          </div>
          <p className="mt-3 text-[14px] text-slate-gray">
            {mode === "rapid"
              ? "Rapid = single full-duplex PortAudio stream (same clock for capture and playback)."
              : "Normal = two independent streams (legacy path)."}
          </p>
        </section>

        <section className="card-float p-6">
          <h3 className="font-display text-[22px] tracking-[-0.2px]">Presets</h3>
          <div className="mt-4 flex flex-wrap gap-3">
            {(Object.keys(PRESETS) as (keyof typeof PRESETS)[]).map((key) => (
              <button
                key={key}
                type="button"
                onClick={() => setPreset(key)}
                className={`pill ${preset === key ? "pill-filled" : "pill-ghost"}`}
              >
                {PRESETS[key].label}
              </button>
            ))}
          </div>
          <p className="mt-3 text-[14px] text-slate-gray">{PRESETS[preset].hint}</p>
        </section>

        <section className="grid gap-5 md:grid-cols-2">
          <DeviceField
            label="capture · input_device"
            value={capture}
            onChange={setCapture}
            options={captureOpts}
          />
          <DeviceField
            label="playback · output_device"
            value={playback}
            onChange={setPlayback}
            options={playbackOpts}
          />
        </section>

        <section className="card-float p-6">
          <h3 className="font-display text-[22px] tracking-[-0.2px]">Bubble size</h3>
          <div className="mt-4 flex flex-wrap gap-3">
            {RADII.map((r) => (
              <button
                key={r.key}
                type="button"
                onClick={() => pickRadius(r.key)}
                className={`pill ${radius === r.key ? "pill-filled" : "pill-ghost"}`}
              >
                {r.label}
              </button>
            ))}
          </div>
          <h3 className="mt-6 font-display text-[22px] tracking-[-0.2px]">Model</h3>
          <div className="mt-4 flex flex-wrap gap-3">
            {radiusModels.map((e) => (
              <button
                key={e.key}
                type="button"
                onClick={() => setModelKey(e.key)}
                className={`pill ${selectedModel.key === e.key ? "pill-filled" : "pill-ghost"}`}
              >
                {e.label}
              </button>
            ))}
          </div>
          <p className="mt-3 text-[14px] text-slate-gray">
            {selectedModel.hint} <code>{selectedModel.model}</code>
          </p>
        </section>

        <section className="grid gap-5 md:grid-cols-2">
          <NumberField label="input_gain_db (on top of mic gains)" value={gainIn} onChange={setGainIn} step={0.5} />
          <NumberField label="output_gain_db" value={gainOut} onChange={setGainOut} step={0.5} />
        </section>

        <section className="grid gap-5 md:grid-cols-2">
          <TextField
            label="mic_gains · linear, device ch 0–5 (empty = off)"
            value={micGains}
            onChange={setMicGains}
            onReset={() => setMicGains(DEFAULT_MIC_GAINS)}
          />
          <TextField
            label="channel_map · model i ← device ch (canon 5,4,3,2,1,0)"
            value={channelMap}
            onChange={setChannelMap}
            onReset={() => setChannelMap(CANON_CHANNEL_MAP)}
          />
        </section>

        <section className="card-float p-6">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h3 className="font-display text-[22px] tracking-[-0.2px]">Dataset recording</h3>
            <button
              type="button"
              onClick={() => setRecordTaps((v) => !v)}
              className={`pill ${recordTaps ? "pill-filled" : "pill-ghost"}`}
            >
              {recordTaps ? "On" : "Off"}
            </button>
          </div>
          <p className="mt-3 text-[14px] text-slate-gray">
            When on, Restart records audio <em>and</em> logs into a new timestamped folder on the Pi&apos;s
            external SSD (<code>measure/</code> = <code>/mnt/sonitude-drive/sonitude-recordings</code>):
            WAVs (raw_8ch / mapped_6ch / dry_ref / wet_out), <code>session.json</code> (config, model,
            gains, argv), <code>pipeline.log</code> (all runtime output), <code>telemetry.jsonl</code>{" "}
            (levels, latency, underruns, distance) and <code>session_end.json</code>. Starts armed; use
            Arm/Disarm/Flush while owned.
          </p>
          {recordTaps ? (
            <div className="mt-5 space-y-4">
              <label className="block">
                <span className="text-[13px] text-ash-gray">tap_dir (under measure/)</span>
                <input
                  type="text"
                  className="mt-2 w-full rounded-2xl border border-line-gray bg-paper-white px-3 py-3 outline-none focus:border-ink-black"
                  value={tapDir}
                  onChange={(e) => setTapDir(e.target.value)}
                  placeholder={DEFAULT_TAP_DIR}
                />
              </label>
              <div>
                <span className="text-[13px] text-ash-gray">tap_select</span>
                <div className="mt-2 flex flex-wrap gap-2">
                  {TAP_STREAMS.map((stream) => (
                    <button
                      key={stream}
                      type="button"
                      onClick={() => toggleStream(stream)}
                      className={`pill ${tapSelect.has(stream) ? "pill-filled" : "pill-ghost"}`}
                    >
                      {stream}
                    </button>
                  ))}
                </div>
              </div>
              <div className="grid gap-4 md:grid-cols-2">
                <NumberField
                  label="tap_max_seconds (0 = unlimited)"
                  value={tapMaxSeconds}
                  onChange={setTapMaxSeconds}
                  step={10}
                />
                <NumberField
                  label="tap_max_mb (0 = unlimited)"
                  value={tapMaxMb}
                  onChange={setTapMaxMb}
                  step={64}
                />
              </div>
            </div>
          ) : null}
          <p className="mt-4 text-[13px] text-ash-gray">
            live · {tapsArmed ? "armed" : "disarmed"}
            {liveTapSelect ? ` · ${liveTapSelect}` : ""}
            {telemetry?.taps?.truncated ? " · truncated" : ""}
          </p>
        </section>

        <div className="flex flex-wrap gap-3">
          <button type="button" className="pill pill-filled" onClick={restart}>
            Restart with config
          </button>
          <button type="button" className="pill pill-ghost" onClick={() => setTaps("arm")}>
            Arm taps
          </button>
          <button type="button" className="pill pill-ghost" onClick={() => setTaps("disarm")}>
            Disarm taps
          </button>
          <button type="button" className="pill pill-ghost" onClick={() => setTaps("flush")}>
            Flush taps
          </button>
        </div>
      </fieldset>

      {msg ? <p className="mt-5 text-[15px] text-slate-gray">{msg}</p> : null}

      <section className="card-mist mt-8 p-5 text-[14px] text-slate-gray">
        <p className="text-[15px] text-ink-black">Proxy / Tailscale</p>
        <ul className="mt-3 space-y-2">
          <li className="flex gap-2">
            <span className="text-ash-gray">01</span>
            UI + proxy only on 127.0.0.1:8787 — bearer injected server-side
          </li>
          <li className="flex gap-2">
            <span className="text-ash-gray">02</span>
            Controller on Pi MagicDNS / tailnet IP :8000 (ACL)
          </li>
          <li className="flex gap-2">
            <span className="text-ash-gray">03</span>
            Single poller fans out on GET /api/stream (SSE)
          </li>
          <li className="flex gap-2">
            <span className="text-ash-gray">04</span>
            Token never shown in the browser
          </li>
        </ul>
      </section>
    </div>
  );
}

function withCurrent(options: AudioDevice[], current: string): AudioDevice[] {
  if (current === "") return options;
  const idx = Number(current);
  if (!Number.isInteger(idx) || options.some((d) => d.index === idx)) return options;
  return [
    ...options,
    {
      index: idx,
      name: `device ${idx}`,
      max_input_channels: 0,
      max_output_channels: 0,
    },
  ];
}

function DeviceField({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: AudioDevice[];
}) {
  return (
    <label className="card-mist block p-5">
      <span className="text-[13px] text-ash-gray">{label}</span>
      <select
        className="mt-2 w-full rounded-2xl border border-line-gray bg-paper-white px-3 py-3 outline-none focus:border-ink-black"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {options.length === 0 ? <option value="">(none)</option> : null}
        {options.map((d) => (
          <option key={d.index} value={String(d.index)}>
            {d.name}
          </option>
        ))}
      </select>
    </label>
  );
}

function TextField({
  label,
  value,
  onChange,
  onReset,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  onReset: () => void;
}) {
  return (
    <label className="card-mist block p-5">
      <span className="text-[13px] text-ash-gray">{label}</span>
      <div className="mt-2 flex gap-2">
        <input
          type="text"
          className="w-full rounded-2xl border border-line-gray bg-paper-white px-3 py-3 tabular-nums outline-none focus:border-ink-black"
          value={value}
          onChange={(e) => onChange(e.target.value)}
        />
        <button type="button" className="pill pill-ghost" onClick={onReset}>
          Default
        </button>
      </div>
    </label>
  );
}

function NumberField({
  label,
  value,
  onChange,
  step,
}: {
  label: string;
  value: number;
  onChange: (n: number) => void;
  step: number;
}) {
  return (
    <label className="card-mist block p-5">
      <span className="text-[13px] text-ash-gray">{label}</span>
      <input
        type="number"
        step={step}
        className="mt-2 w-full rounded-2xl border border-line-gray bg-paper-white px-3 py-3 tabular-nums outline-none focus:border-ink-black"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}
