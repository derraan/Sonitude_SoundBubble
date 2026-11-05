import { useEffect, useState } from "react";
import { apiGet } from "../hooks/useTelemetryStream";
import { EmptyState } from "./EmptyState";

type LogsResponse = { lines?: string[]; log?: string; raw?: string };

function classify(line: string) {
  if (line.includes("[profile]")) return "text-sienna-brown";
  if (/error|stderr|fatal|underrun/i.test(line)) return "text-alert-red";
  return "text-ink-black/85";
}

export function LogsDrawer({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [lines, setLines] = useState<string[]>([]);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    const load = async () => {
      try {
        const data = await apiGet<LogsResponse>("/api/logs?tail=200");
        if (cancelled) return;
        if (Array.isArray(data.lines)) setLines(data.lines);
        else if (typeof data.log === "string") setLines(data.log.split(/\r?\n/));
        else if (typeof data.raw === "string") setLines(data.raw.split(/\r?\n/));
        else setLines([JSON.stringify(data)]);
        setErr(null);
      } catch (e) {
        if (!cancelled) setErr(String(e));
      }
    };
    load();
    const id = setInterval(load, 2000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-40 flex justify-end">
      <button
        className="absolute inset-0 bg-ink-black/25 backdrop-blur-[2px]"
        aria-label="Close logs"
        onClick={onClose}
      />
      <aside className="relative flex h-full w-full max-w-xl flex-col bg-paper-white shadow-[var(--shadow-drawer)]">
        <header className="flex items-start justify-between gap-4 border-b border-line-gray px-6 py-5">
          <div>
            <h2 className="font-display text-[26px] tracking-[-0.23px]">Logs</h2>
            <p className="mt-1 text-[14px] text-slate-gray">
              <span className="text-sienna-brown">[profile]</span> and{" "}
              <span className="text-alert-red">stderr</span> highlighted · Esc to close
            </p>
          </div>
          <button type="button" className="pill pill-ghost shrink-0" onClick={onClose}>
            Close
          </button>
        </header>
        <div className="flex-1 overflow-auto bg-fog-white p-4 font-mono text-[12.5px] leading-5">
          {err ? (
            <EmptyState title="Logs unavailable" body={err} />
          ) : lines.length === 0 ? (
            <EmptyState
              title="No lines yet"
              body="Waiting for /api/logs. In observe mode the controller tails journalctl."
            />
          ) : (
            lines.map((line, i) => (
              <div key={i} className={`whitespace-pre-wrap break-all ${classify(line)}`}>
                {line || " "}
              </div>
            ))
          )}
        </div>
      </aside>
    </div>
  );
}
