import type { LinkState } from "../types";

const copy: Record<Exclude<LinkState, "ok">, { title: string; body: string }> = {
  pi_unreachable: {
    title: "Pi unreachable",
    body: "Proxy cannot reach PI_URL over Tailscale. Check MagicDNS, ACL to :8000, and .env.local.",
  },
  proxy_down: {
    title: "Local proxy down",
    body: "The browser lost its EventSource to 127.0.0.1:8787. The local proxy stopped — this is not the Pi.",
  },
  auth_failed: {
    title: "Auth failed",
    body: "Bearer token rejected. Update PI_TOKEN in edge/dashboard/.env.local — it is injected by the proxy only.",
  },
  controller_down: {
    title: "Controller down",
    body: "Pi answered but /health or /telemetry failed. Is soundbubble-controller running on the tailnet IP?",
  },
};

export function ConnectionBanner({
  link,
  lastError,
}: {
  link: LinkState;
  lastError?: string | null;
}) {
  if (link === "ok") return null;
  const c = copy[link];
  return (
    <div className="mb-8 rounded-3xl bg-mist-gray px-5 py-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-[16px] font-medium text-ink-black">{c.title}</p>
          <p className="mt-1 max-w-3xl text-[15px] leading-snug text-slate-gray">{c.body}</p>
          {lastError ? (
            <p className="mt-2 font-mono text-[12px] text-ash-gray">{lastError}</p>
          ) : null}
        </div>
        <span className="rounded-full bg-paper-white px-3 py-1 text-[13px] text-slate-gray">
          UI still live · mock/empty until link recovers
        </span>
      </div>
    </div>
  );
}
