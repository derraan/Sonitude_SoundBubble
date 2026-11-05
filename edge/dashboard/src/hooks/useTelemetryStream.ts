import { useEffect, useState } from "react";
import type { LinkState, StreamPayload, Telemetry } from "../types";

export function useTelemetryStream() {
  const [link, setLink] = useState<LinkState>("pi_unreachable");
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null);
  const [updatedAt, setUpdatedAt] = useState(0);
  const [trialsSource, setTrialsSource] = useState<string>("pi");
  const [lastError, setLastError] = useState<string | null>(null);

  useEffect(() => {
    const es = new EventSource("/api/stream");
    es.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data) as StreamPayload;
        setLink(msg.link);
        setTelemetry(msg.telemetry);
        setUpdatedAt(msg.updatedAt);
        setTrialsSource(msg.trialsSource || "pi");
        setLastError(msg.lastError ?? null);
      } catch {
        /* ignore malformed */
      }
    };
    es.onerror = () => {
      setLink("proxy_down");
      setLastError("sse disconnected");
    };
    return () => es.close();
  }, []);

  return {
    link,
    telemetry,
    updatedAt,
    trialsSource,
    lastError,
  };
}

export async function apiGet<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path} ${r.status}`);
  return r.json() as Promise<T>;
}

export async function apiPost<T>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((data as { error?: string }).error || `${path} ${r.status}`);
  return data as T;
}
