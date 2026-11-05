import type { ReactNode } from "react";
import type { LinkState, RuntimeKind, SourceMode, TapsState } from "../types";

const linkLabel: Record<LinkState, string> = {
  ok: "ok",
  pi_unreachable: "pi unreachable",
  proxy_down: "proxy down",
  auth_failed: "auth failed",
  controller_down: "controller down",
};

function linkClass(link: LinkState) {
  if (link === "ok") return "bg-ok-ink/10 text-ok-ink";
  if (link === "auth_failed" || link === "proxy_down") return "bg-warn-amber/12 text-warn-amber";
  return "bg-alert-red/10 text-alert-red";
}

function Badge({
  children,
  className = "bg-mist-gray text-ink-black",
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <span
      className={`inline-flex items-center rounded-full px-3 py-1.5 text-[13px] tracking-[0.01em] ${className}`}
    >
      {children}
    </span>
  );
}

export function StatusBadges({
  link,
  source,
  runtime,
  taps,
  trialsSource,
}: {
  link: LinkState;
  source?: SourceMode | null;
  runtime?: RuntimeKind | null;
  taps?: TapsState | null;
  trialsSource?: string;
}) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Badge className={linkClass(link)}>
        <span className="mr-1.5 inline-block h-1.5 w-1.5 rounded-full bg-current opacity-80" />
        link · {linkLabel[link]}
      </Badge>
      <Badge>mode · {source || "—"}</Badge>
      <Badge>runtime · {runtime || "—"}</Badge>
      <Badge
        className={
          taps?.armed
            ? "bg-blush-peach text-sienna-brown"
            : "bg-mist-gray text-slate-gray"
        }
      >
        tap · {taps?.armed ? "armed" : "disarmed"}
      </Badge>
      {trialsSource === "local" ? (
        <Badge className="bg-blush-peach text-sienna-brown">
          trials · LOCAL_MEASURE_DIR
        </Badge>
      ) : null}
    </div>
  );
}
