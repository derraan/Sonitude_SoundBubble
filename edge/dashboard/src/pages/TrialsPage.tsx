import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { EmptyState } from "../components/EmptyState";
import { PageHeader } from "../components/PageHeader";
import { apiGet } from "../hooks/useTelemetryStream";
import type { Telemetry, TrialRecord } from "../types";

type TrialsResp = { records?: TrialRecord[]; source?: string; run_id?: string | null };

const PERSIST_MS = 30_000;

type PersistPoint = { x: number; y: number; born: number };

export function TrialsPage({
  telemetry,
  trialsSource,
}: {
  telemetry: Telemetry | null;
  trialsSource?: string;
}) {
  const [records, setRecords] = useState<TrialRecord[]>([]);
  const [source, setSource] = useState(trialsSource || "pi");
  const [hostFilter, setHostFilter] = useState("all");
  const [runtimeFilter, setRuntimeFilter] = useState("all");
  const [err, setErr] = useState<string | null>(null);
  const [copied, setCopied] = useState<string | null>(null);
  const [points, setPoints] = useState<PersistPoint[]>([]);
  const [now, setNow] = useState(() => performance.now());
  const lastSampleKey = useRef<string>("");

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const data = await apiGet<TrialsResp>("/api/trials");
        if (cancelled) return;
        const rows = data.records?.length ? data.records : telemetry?.trials_tail || [];
        setRecords(rows);
        setSource(data.source || trialsSource || "pi");
        setErr(null);
      } catch (e) {
        if (!cancelled) {
          setRecords(telemetry?.trials_tail || []);
          setErr(String(e));
        }
      }
    };
    load();
    const id = setInterval(load, 2000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [telemetry?.trials_tail, trialsSource]);

  // Live monitor → persistence dots (level vs distance).
  useEffect(() => {
    const m = telemetry?.last_monitor;
    if (!m) return;
    const dist = Number(m.distance_m);
    const level =
      Number.isFinite(Number(m.output_db))
        ? Number(m.output_db)
        : Number(m.model_out_db);
    const valid =
      m.distance_status === "valid" ||
      (m.distance_status == null && Number.isFinite(dist));
    if (!valid || !Number.isFinite(dist) || !Number.isFinite(level)) return;

    const key = `${telemetry?.seq ?? ""}:${dist.toFixed(3)}:${level.toFixed(2)}`;
    if (key === lastSampleKey.current) return;
    lastSampleKey.current = key;

    const born = performance.now();
    setPoints((prev) => {
      const kept = prev.filter((p) => born - p.born < PERSIST_MS);
      return [...kept, { x: dist, y: level, born }];
    });
  }, [telemetry?.last_monitor, telemetry?.seq]);

  // Drive fade / despawn independently of SSE rate.
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const t = performance.now();
      setNow(t);
      setPoints((prev) => {
        const next = prev.filter((p) => t - p.born < PERSIST_MS);
        return next.length === prev.length ? prev : next;
      });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  const hosts = useMemo(
    () => ["all", ...Array.from(new Set(records.map((r) => String(r.host || "unknown"))))],
    [records],
  );
  const runtimes = useMemo(
    () => ["all", ...Array.from(new Set(records.map((r) => String(r.runtime || "unknown"))))],
    [records],
  );

  const filtered = records.filter((r) => {
    if (hostFilter !== "all" && String(r.host || "unknown") !== hostFilter) return false;
    if (runtimeFilter !== "all" && String(r.runtime || "unknown") !== runtimeFilter)
      return false;
    return true;
  });

  const radiusM = Number(telemetry?.last_monitor?.bubble_radius_m ?? 1.5);

  async function copyScp(trial: TrialRecord) {
    const path = String(trial.trial_path || "");
    const cmd = path
      ? `scp -r sonitude@172.20.10.8:${path} .`
      : "scp -r sonitude@172.20.10.8:<trial_path> .";
    await navigator.clipboard.writeText(cmd);
    setCopied(cmd);
    setTimeout(() => setCopied(null), 2000);
  }

  return (
    <div>
      <PageHeader
        title="Trials"
        actions={
          <>
            <FilterSelect label="host" value={hostFilter} onChange={setHostFilter} options={hosts} />
            <FilterSelect
              label="runtime"
              value={runtimeFilter}
              onChange={setRuntimeFilter}
              options={runtimes}
            />
          </>
        }
      >
        Live <code className="text-ink-black">output_db</code> vs distance (30&nbsp;s persistence).
        Table: shared <code className="text-ink-black">telemetry.jsonl</code> · source{" "}
        <span className="text-ink-black">{source}</span>
        {source === "local" ? " · Windows testbench / LOCAL_MEASURE_DIR" : ""}.
      </PageHeader>

      {err ? <p className="mb-4 text-[14px] text-alert-red">{err}</p> : null}
      {copied ? (
        <p className="mb-4 rounded-3xl bg-blush-peach px-4 py-3 text-[14px] text-sienna-brown">
          Copied · {copied}
        </p>
      ) : null}

      <ChartCard title="level vs distance">
        {points.length === 0 ? (
          <EmptyState
            title="Waiting for samples"
            body="Need valid distance_m and output_db (or model_out_db) from live monitor."
          />
        ) : (
          <PersistenceScatter points={points} now={now} radiusM={radiusM} />
        )}
      </ChartCard>

      <section className="card-float mt-5 overflow-hidden">
        <table className="w-full text-left text-[15px]">
          <thead className="bg-mist-gray text-[13px] text-slate-gray">
            <tr>
              <th className="px-5 py-3 font-normal">host</th>
              <th className="px-5 py-3 font-normal">runtime</th>
              <th className="px-5 py-3 font-normal">dist</th>
              <th className="px-5 py-3 font-normal">angle</th>
              <th className="px-5 py-3 font-normal">supp</th>
              <th className="px-5 py-3 font-normal">scp</th>
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 ? (
              <tr>
                <td className="px-5 py-8 text-smoke-gray" colSpan={6}>
                  No trial records yet — measure wrapper or LOCAL_MEASURE_DIR sample JSONL.
                </td>
              </tr>
            ) : (
              filtered
                .slice(-50)
                .reverse()
                .map((r, i) => (
                  <tr key={i} className="border-t border-line-gray">
                    <td className="px-5 py-3.5">{String(r.host || "—")}</td>
                    <td className="px-5 py-3.5">{String(r.runtime || "—")}</td>
                    <td className="px-5 py-3.5 tabular-nums">{fmt(r.distance_m)}</td>
                    <td className="px-5 py-3.5 tabular-nums">{fmt(r.angle_deg)}</td>
                    <td className="px-5 py-3.5 tabular-nums">{fmt(r.suppression_db)}</td>
                    <td className="px-5 py-3.5">
                      <button
                        type="button"
                        className="text-ink-black transition-opacity hover:opacity-70"
                        onClick={() => copyScp(r)}
                      >
                        copy scp →
                      </button>
                    </td>
                  </tr>
                ))
            )}
          </tbody>
        </table>
      </section>
    </div>
  );
}

function fmt(v: unknown) {
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(2) : "—";
}

function FilterSelect({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: string[];
}) {
  return (
    <label className="flex items-center gap-2 text-[13px] text-ash-gray">
      {label}
      <select
        className="rounded-2xl border border-line-gray bg-paper-white px-3 py-2 text-[15px] text-ink-black outline-none focus:border-ink-black"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </label>
  );
}

function ChartCard({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="card-float p-5">
      <h3 className="mb-4 font-display text-[22px] tracking-[-0.2px]">{title}</h3>
      {children}
    </section>
  );
}

function PersistenceScatter({
  points,
  now,
  radiusM,
}: {
  points: PersistPoint[];
  now: number;
  radiusM: number;
}) {
  const w = 720;
  const h = 320;
  const pad = 44;
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const minX = 0;
  const maxX = Math.max(2, Number.isFinite(radiusM) ? radiusM * 1.25 : 2, ...xs);
  const minY = Math.min(-60, ...ys);
  const maxY = Math.max(0, ...ys);
  const sx = (x: number) => pad + ((x - minX) / (maxX - minX || 1)) * (w - pad * 2);
  const sy = (y: number) => h - pad - ((y - minY) / (maxY - minY || 1)) * (h - pad * 2);

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full">
      <rect x={pad} y={pad} width={w - pad * 2} height={h - pad * 2} fill="#fafafb" rx={12} />
      <line x1={pad} y1={h - pad} x2={w - pad} y2={h - pad} stroke="#ececec" />
      <line x1={pad} y1={pad} x2={pad} y2={h - pad} stroke="#ececec" />
      {Number.isFinite(radiusM) && radiusM > 0 ? (
        <line
          x1={sx(radiusM)}
          y1={pad}
          x2={sx(radiusM)}
          y2={h - pad}
          stroke="#f5d76e"
          strokeDasharray="4 4"
          opacity={0.85}
        />
      ) : null}
      <text x={w / 2} y={h - 12} textAnchor="middle" fontSize={12} fill="#979799">
        distance_m
      </text>
      <text
        x={16}
        y={h / 2}
        fontSize={12}
        fill="#979799"
        transform={`rotate(-90 16 ${h / 2})`}
        textAnchor="middle"
      >
        output_db
      </text>
      <text x={pad} y={pad - 10} fontSize={11} fill="#979799">
        {maxY.toFixed(0)}
      </text>
      <text x={pad} y={h - pad + 14} fontSize={11} fill="#979799">
        {minY.toFixed(0)}
      </text>
      <text x={w - pad} y={h - pad + 14} fontSize={11} fill="#979799" textAnchor="end">
        {maxX.toFixed(1)}m
      </text>
      {points.map((p, i) => {
        const age = now - p.born;
        const opacity = Math.max(0, 1 - age / PERSIST_MS);
        return (
          <circle
            key={`${p.born}-${i}`}
            cx={sx(p.x)}
            cy={sy(p.y)}
            r={4.5}
            fill="#5d2a1a"
            opacity={opacity}
          />
        );
      })}
    </svg>
  );
}
