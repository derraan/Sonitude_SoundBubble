import { PageHeader } from "../components/PageHeader";
import type { Telemetry } from "../types";

export function BubblePage({ telemetry }: { telemetry: Telemetry | null }) {
  const m = telemetry?.last_monitor;
  const radiusM = Number(m?.bubble_radius_m ?? 1.5);
  const distM = Number(m?.distance_m ?? NaN);
  const valid = m?.distance_status === "valid";
  const levels = m?.levels_dbfs ?? Array(6).fill(-60);
  const statuses = m?.statuses ?? Array(6).fill("unknown");
  const speakers = m?.speaker_count ?? "—";
  const speakerDetail = typeof m?.speakers === "string" ? m.speakers : "";
  const modelOut = Number(m?.model_out_db ?? NaN);
  const outputDb = Number(m?.output_db ?? NaN);
  const hasLevels = Array.isArray(m?.levels_dbfs) && m!.levels_dbfs!.length > 0;

  const size = 440;
  const cx = size / 2;
  // Ground plane sits lower; dome rises above it
  const groundY = size * 0.62;
  const baseRx = 148;
  const baseRy = 42;
  const domeH = 158;
  const hasDist = Number.isFinite(distM);
  const distFrac = hasDist
    ? Math.min(1.28, Math.max(0, distM / Math.max(radiusM, 0.05)))
    : 0;
  const distPx = distFrac * baseRx;
  const inside = hasDist && distM <= radiusM;
  // Marker on elliptical ground plane (perspective foreshortening)
  const markerX = cx + distPx * 0.95;
  const markerY = groundY + (distPx / baseRx) * baseRy * 0.15;

  // Dome silhouette: semicircle-ish path from left base → apex → right base
  const left = cx - baseRx;
  const right = cx + baseRx;
  const apexY = groundY - domeH;
  const domePath = [
    `M ${left} ${groundY}`,
    `C ${left} ${groundY - domeH * 0.72}, ${cx - baseRx * 0.55} ${apexY}, ${cx} ${apexY}`,
    `C ${cx + baseRx * 0.55} ${apexY}, ${right} ${groundY - domeH * 0.72}, ${right} ${groundY}`,
    `Z`,
  ].join(" ");

  return (
    <div>
      <PageHeader title="Bubble">
        Hemispherical privacy dome with you at the center. Estimated distance places a
        marker on the ground disk — inside or outside the rim. Channel levels stay in the
        side panel.
      </PageHeader>

      <div className="grid gap-6 lg:grid-cols-[1.25fr_0.75fr]">
        <section className="card-float relative flex items-center justify-center overflow-hidden bg-transparent p-2 shadow-none ring-1 ring-line-gray/80">
          <svg
            width={size}
            height={size}
            viewBox={`0 0 ${size} ${size}`}
            className="mx-auto max-w-full"
            aria-label="Hemispherical sound bubble with user at center"
          >
            <defs>
              <linearGradient id="domeSkin" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#fff6c8" stopOpacity="0.55" />
                <stop offset="45%" stopColor="#f5d76e" stopOpacity="0.28" />
                <stop offset="100%" stopColor="#e0b83a" stopOpacity="0.12" />
              </linearGradient>
              <radialGradient id="domeHighlight" cx="38%" cy="18%" r="45%">
                <stop offset="0%" stopColor="#ffffff" stopOpacity="0.7" />
                <stop offset="55%" stopColor="#ffffff" stopOpacity="0.15" />
                <stop offset="100%" stopColor="#ffffff" stopOpacity="0" />
              </radialGradient>
              <radialGradient id="baseDisk" cx="50%" cy="40%" r="55%">
                <stop offset="0%" stopColor="#f5d76e" stopOpacity="0.22" />
                <stop offset="70%" stopColor="#f5d76e" stopOpacity="0.08" />
                <stop offset="100%" stopColor="#f5d76e" stopOpacity="0" />
              </radialGradient>
              <filter id="domeGlow" x="-30%" y="-30%" width="160%" height="160%">
                <feGaussianBlur stdDeviation="5" result="b" />
                <feMerge>
                  <feMergeNode in="b" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>

            {/* Ground disk (hemisphere base) */}
            <ellipse
              cx={cx}
              cy={groundY}
              rx={baseRx}
              ry={baseRy}
              fill="url(#baseDisk)"
            />
            <ellipse
              cx={cx}
              cy={groundY}
              rx={baseRx}
              ry={baseRy}
              fill="none"
              stroke="#e8c547"
              strokeWidth={2}
              strokeOpacity={0.75}
            />
            <ellipse
              cx={cx}
              cy={groundY}
              rx={baseRx * 0.72}
              ry={baseRy * 0.72}
              fill="none"
              stroke="#ececec"
              strokeWidth={1}
              strokeDasharray="4 6"
            />

            {/* Soft glow behind dome */}
            <path d={domePath} fill="#f5d76e" opacity={0.1} filter="url(#domeGlow)" />

            {/* Hemisphere shell */}
            <path d={domePath} fill="url(#domeSkin)" />
            <path
              d={domePath}
              fill="url(#domeHighlight)"
            />
            <path
              d={[
                `M ${left} ${groundY}`,
                `C ${left} ${groundY - domeH * 0.72}, ${cx - baseRx * 0.55} ${apexY}, ${cx} ${apexY}`,
                `C ${cx + baseRx * 0.55} ${apexY}, ${right} ${groundY - domeH * 0.72}, ${right} ${groundY}`,
              ].join(" ")}
              fill="none"
              stroke="#e8c547"
              strokeWidth={2.75}
              strokeLinecap="round"
            />

            {/* Meridian depth cues */}
            <path
              d={`M ${cx} ${groundY} Q ${cx - 36} ${groundY - domeH * 0.55}, ${cx} ${apexY}`}
              fill="none"
              stroke="#c9a227"
              strokeOpacity={0.28}
              strokeWidth={1.25}
            />
            <path
              d={`M ${cx} ${groundY} Q ${cx + 36} ${groundY - domeH * 0.55}, ${cx} ${apexY}`}
              fill="none"
              stroke="#c9a227"
              strokeOpacity={0.28}
              strokeWidth={1.25}
            />

            <text
              x={cx}
              y={apexY - 16}
              textAnchor="middle"
              fill="#5d2a1a"
              fontSize={13}
              fontFamily="Inter, ui-sans-serif"
            >
              bubble {radiusM.toFixed(2)} m
            </text>

            <UserHead cx={cx} cy={groundY - 28} />

            {hasDist ? (
              <g>
                <ellipse
                  cx={markerX}
                  cy={markerY + 8}
                  rx={11}
                  ry={4.5}
                  fill={valid ? "#17191c" : "#a3a6af"}
                  opacity={0.12}
                />
                <circle
                  cx={markerX}
                  cy={markerY}
                  r={8}
                  fill={valid ? (inside ? "#17191c" : "#ffffff") : "#a3a6af"}
                  stroke={valid ? (inside ? "#f5d76e" : "#17191c") : "#777b86"}
                  strokeWidth={2.5}
                />
                <circle
                  cx={markerX}
                  cy={markerY}
                  r={14}
                  fill={valid ? "#f5d76e" : "#a3a6af"}
                  opacity={0.18}
                >
                  <animate
                    attributeName="r"
                    values="12;16;12"
                    dur="2.4s"
                    repeatCount="indefinite"
                  />
                </circle>
                <text
                  x={markerX}
                  y={markerY - 14}
                  textAnchor="middle"
                  fill="#17191c"
                  fontSize={12}
                  fontFamily="Inter, ui-sans-serif"
                >
                  {distM.toFixed(2)} m · {inside ? "inside" : "outside"}
                </text>
              </g>
            ) : (
              <text
                x={cx}
                y={groundY + baseRy + 28}
                textAnchor="middle"
                fill="#a3a6af"
                fontSize={13}
                fontFamily="Inter, ui-sans-serif"
              >
                waiting for distance estimate
              </text>
            )}
          </svg>
        </section>

        <section className="flex flex-col gap-4">
          <div className="card-mist p-5">
            <p className="text-[13px] text-ash-gray">distance</p>
            <p className="mt-1 font-display text-[32px] tracking-[-0.4px]">
              {hasDist ? distM.toFixed(2) : "—"}
              {hasDist ? <span className="ml-1 text-[16px] text-slate-gray">m</span> : null}
            </p>
            <p className={`mt-2 text-[14px] ${valid ? "text-ok-ink" : "text-smoke-gray"}`}>
              {m?.distance_status || "unknown"}
              {hasDist ? (inside ? " · inside bubble" : " · outside bubble") : ""}
              {!valid && hasDist ? " · marker muted" : ""}
            </p>
          </div>
          <div className="card-mist p-5">
            <p className="text-[13px] text-ash-gray">speakers</p>
            <p className="mt-1 font-display text-[32px] tracking-[-0.4px]">{speakers}</p>
            {speakerDetail ? (
              <p className="mt-2 break-all font-mono text-[12px] text-slate-gray">{speakerDetail}</p>
            ) : null}
          </div>
          <div className="card-mist p-5">
            <p className="text-[13px] text-ash-gray">model_out / out</p>
            <p className="mt-1 tabular-nums text-[18px]">
              {Number.isFinite(modelOut) ? `${modelOut.toFixed(1)} dBFS` : "—"}
              <span className="mx-2 text-ash-gray">·</span>
              {Number.isFinite(outputDb) ? `${outputDb.toFixed(1)} dBFS` : "—"}
            </p>
          </div>
          <div className="card-mist flex-1 p-5">
            <p className="text-[13px] text-ash-gray">levels_dbfs</p>
            <p className="mt-1 text-[12px] text-smoke-gray">
              {hasLevels
                ? "Same ch* rows as runtime --monitor (post input gain)"
                : "Waiting for monitor…"}
            </p>
            <ul className="mt-4 space-y-2.5">
              {Array.from({ length: 6 }).map((_, i) => (
                <li key={i} className="flex items-center justify-between text-[15px]">
                  <span className="text-slate-gray">Ch{i}</span>
                  <span className="tabular-nums">
                    {hasLevels && Number.isFinite(Number(levels[i]))
                      ? Number(levels[i]).toFixed(1)
                      : "—"}
                    <span className="ml-2 text-[13px] text-ash-gray">
                      {hasLevels ? statuses[i] || "—" : "—"}
                    </span>
                  </span>
                </li>
              ))}
            </ul>
          </div>
        </section>
      </div>
    </div>
  );
}

function UserHead({ cx, cy }: { cx: number; cy: number }) {
  return (
    <g aria-label="User">
      <ellipse cx={cx} cy={cy + 22} rx={18} ry={6} fill="#17191c" opacity={0.08} />
      <circle cx={cx} cy={cy + 18} r={14} fill="#17191c" />
      <circle cx={cx} cy={cy - 2} r={11} fill="#17191c" />
      <circle cx={cx - 3} cy={cy - 5} r={3.5} fill="#ffffff" opacity={0.2} />
      <text
        x={cx}
        y={cy + 40}
        textAnchor="middle"
        fill="#777b86"
        fontSize={11}
        fontFamily="Inter, ui-sans-serif"
      >
        you
      </text>
    </g>
  );
}
