export type LinkState =
  | "ok"
  | "pi_unreachable"
  | "proxy_down"
  | "auth_failed"
  | "controller_down";

export type SourceMode = "observe" | "owned" | "idle" | string;
export type RuntimeKind = "cpp" | "python" | string;

export type ProfileSample = {
  e2e_ms?: number;
  infer_ms?: number;
  out_ahead_ms?: number;
  underrun_frames?: number;
  drops_oldest?: number;
  out_dropped_frames?: number;
  model_ring_drops?: number;
  model_ring_resets?: number;
  [k: string]: unknown;
};

export type MonitorSample = {
  levels_dbfs?: number[];
  statuses?: string[];
  distance_m?: number;
  distance_status?: string;
  speaker_count?: number | string;
  bubble_radius_m?: number;
  model_out_db?: number;
  output_db?: number;
  /** Raw `speakers=[S1: …]` string from the runtime monitor block. */
  speakers?: string;
  [k: string]: unknown;
};

export type TapsState = {
  armed?: boolean;
  select?: string;
  truncated?: boolean;
};

export type TrialRecord = {
  host?: string;
  runtime?: string;
  distance_m?: number;
  angle_deg?: number;
  suppression_db?: number;
  levels_dbfs?: number[];
  infer_mean_ms?: number;
  underrun_frames?: number;
  tap_truncated?: boolean;
  trial_path?: string;
  [k: string]: unknown;
};

export type Telemetry = {
  source?: SourceMode;
  runtime?: RuntimeKind;
  host?: string;
  seq?: number;
  last_profile?: ProfileSample | null;
  last_monitor?: MonitorSample | null;
  taps?: TapsState | null;
  trials_tail?: TrialRecord[];
  /** Running controller config when /status or /telemetry includes it. */
  config?: Record<string, unknown>;
  _note?: string;
};

export type StreamPayload = {
  type: "telemetry";
  link: LinkState;
  updatedAt: number;
  telemetry: Telemetry | null;
  health: unknown;
  trialsSource?: "pi" | "local" | string;
  lastError?: string | null;
};
