import { NavLink, Outlet } from "react-router-dom";
import { ConnectionBanner } from "./ConnectionBanner";
import { StatusBadges } from "./StatusBadges";
import type { LinkState, Telemetry } from "../types";

const nav = [
  { to: "/bubble", label: "Bubble" },
  { to: "/trials", label: "Trials" },
  { to: "/settings", label: "Settings" },
];

export function AppShell({
  link,
  telemetry,
  trialsSource,
  lastError,
  onOpenLogs,
}: {
  link: LinkState;
  telemetry: Telemetry | null;
  trialsSource?: string;
  lastError?: string | null;
  onOpenLogs: () => void;
}) {
  return (
    <div className="flex min-h-full bg-paper-white">
      <aside className="sticky top-0 flex h-screen w-[232px] shrink-0 flex-col bg-fog-white px-5 py-8">
        <div className="mb-12 px-2">
          <p className="text-[13px] uppercase tracking-[0.08em] text-ash-gray">
            Sonitude
          </p>
          <h1 className="mt-2 font-display text-[28px] leading-[1.1] tracking-[-0.28px] text-ink-black">
            Telemetry
          </h1>
          <p className="mt-3 text-[14px] leading-snug text-slate-gray">
            127.0.0.1:8787
            <br />
            Tailscale JSON only
          </p>
        </div>

        <nav className="flex flex-col gap-1">
          {nav.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `rounded-[16px] px-4 py-3 text-[16px] transition-colors ${
                  isActive
                    ? "bg-ink-black text-paper-white"
                    : "text-ink-black hover:bg-mist-gray"
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
          <button
            type="button"
            onClick={onOpenLogs}
            className="rounded-[16px] px-4 py-3 text-left text-[16px] text-ink-black transition-colors hover:bg-mist-gray"
          >
            Logs →
          </button>
        </nav>

        <div className="mt-auto card-peach p-4">
          <p className="text-[14px] font-medium">Ops note</p>
          <p className="mt-1.5 text-[14px] leading-snug opacity-90">
            WAV stays on disk. Copy trials with scp — never stream PCM over
            Tailscale.
          </p>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-10 border-b border-line-gray bg-paper-white/90 px-8 py-4 backdrop-blur-md">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <StatusBadges
              link={link}
              source={telemetry?.source}
              runtime={telemetry?.runtime}
              taps={telemetry?.taps}
              trialsSource={trialsSource}
            />
            <div className="text-[13px] tabular-nums text-slate-gray">
              <span className="text-ash-gray">host</span>{" "}
              {telemetry?.host || "—"}
              <span className="mx-2 text-line-gray">·</span>
              <span className="text-ash-gray">seq</span>{" "}
              {telemetry?.seq ?? "—"}
            </div>
          </div>
        </header>

        <main className="flex-1 overflow-auto px-8 py-8">
          <div className="mx-auto max-w-[1200px]">
            <ConnectionBanner link={link} lastError={lastError} />
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
}
