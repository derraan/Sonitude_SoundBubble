#!/usr/bin/env bash
# Record a live Sound Bubble session on the Pi: runs soundbubble-rt with taps armed and
# writes raw_8ch / mapped_6ch / dry_ref / wet_out WAVs + session.json into a NEW folder.
#
#   ./record_session.sh <label> [seconds]        e.g. ./record_session.sh cafe_walk 300
#
# Settings come from the same env file as the service (SOUNDBUBBLE_ENV, default
# /etc/sonitude-soundbubble.env), so a recording matches what the wearer hears.
# Stop the service first if it holds the Pico (no sudo here; the owner runs
# `sudo systemctl stop sonitude-soundbubble`). raw_8ch is pre-mic-gain device order.
set -euo pipefail

LABEL="${1:?usage: record_session.sh <label> [seconds]}"
SECONDS_MAX="${2:-600}"
ENV_FILE="${SOUNDBUBBLE_ENV:-/etc/sonitude-soundbubble.env}"
# shellcheck disable=SC1090
[[ -r "$ENV_FILE" ]] && set -a && . "$ENV_FILE" && set +a

ROOT="${SOUNDBUBBLE_ROOT:-/home/sonitude/sb_rt_dev}"
RT_BIN="${SOUNDBUBBLE_RT_BIN:-$ROOT/edge/cpp/build/pi/soundbubble-rt}"
MODEL="${SOUNDBUBBLE_MODEL:-edge/zoo/optim_sonitude_dist_count_1_5m_ep35.onnx}"
CONTRACT="${SOUNDBUBBLE_CONTRACT:-edge/zoo/optim_sonitude_dist_count_1_5m_ep35.runtime.json}"
REC_DRIVE="${SOUNDBUBBLE_REC_DRIVE:-/mnt/sonitude-drive}"
REC_ROOT="${SOUNDBUBBLE_REC_DIR:-$REC_DRIVE/sonitude-recordings/sessions}"
CHANNEL_MAP="${CHANNEL_MAP:-5,4,3,2,1,0}"   # edge/MIC_INDEX_CANON.md §4
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$REC_ROOT/${STAMP}_${LABEL//[^A-Za-z0-9_-]/_}"

# Recordings live on the external SSD only (never the SD card).
if [[ "$REC_ROOT" == "$REC_DRIVE"/* ]] && ! mountpoint -q "$REC_DRIVE"; then
  echo "[rec] $REC_DRIVE is not mounted; plug in the Samsung T5 (fstab UUID=B424-9B8F) and retry" >&2
  exit 1
fi
if [[ -e "$OUT" ]]; then echo "[rec] refusing to reuse $OUT" >&2; exit 1; fi
if systemctl is-active --quiet sonitude-soundbubble 2>/dev/null; then
  echo "[rec] sonitude-soundbubble is running and owns the Pico; stop it first (owner: sudo systemctl stop sonitude-soundbubble)" >&2
  exit 1
fi
mkdir -p "$OUT"
cd "$ROOT"
[[ -x "$RT_BIN" ]] || { echo "[rec] missing $RT_BIN" >&2; exit 1; }

ARGS=(--model "$MODEL" --contract-path "$CONTRACT"
  --input-device "${INPUT_DEVICE:-0}" --output-device "${OUTPUT_DEVICE:-0}"
  --input-channels 8 --output-channels 2 --channel-map "$CHANNEL_MAP"
  --input-sr "${INPUT_SR:-24000}" --output-sr "${OUTPUT_SR:-24000}"
  --silence-dbfs "${SILENCE_DBFS:--80}" --input-gain-db "${INPUT_GAIN_DB:-0}"
  --output-gain-db "${OUTPUT_GAIN_DB:-0}" --mix-dry "${MIX_DRY:-0.0}"
  --intra-op-threads "${INTRA_OP_THREADS:-2}" --inter-op-threads "${INTER_OP_THREADS:-1}"
  --tap-dir "$OUT" --tap-select "${TAP_SELECT:-raw,mapped,dry,wet}"
  --tap-max-seconds "$SECONDS_MAX" --profile --monitor)
[[ -n "${MIC_GAINS:-}" ]] && ARGS+=(--mic-gains "$MIC_GAINS")
[[ -n "${WET_GATE_DBFS:-}" ]] && ARGS+=(--wet-gate-dbfs "$WET_GATE_DBFS")
[[ -e "${MIX_UART_PORT:-/dev/ttyACM0}" ]] && ARGS+=(--mix-uart-port "${MIX_UART_PORT:-/dev/ttyACM0}" --mix-uart-baud "${MIX_UART_BAUD:-115200}")

python3 - "$OUT/session.json" "$LABEL" "$ENV_FILE" "$RT_BIN" "$ROOT/$MODEL" "${ARGS[@]}" <<'PY'
import hashlib, json, socket, sys, datetime
out, label, env, rt, model, *argv = sys.argv[1:]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
json.dump({"label": label, "created_local": datetime.datetime.now().isoformat(timespec="seconds"),
           "host": socket.gethostname(), "env_file": env, "rt_bin": rt, "rt_sha256": sha(rt),
           "model": model, "model_sha256": sha(model), "argv": argv,
           "notes": "raw_8ch = device order, before input gain and mic gains; see edge/MIC_INDEX_CANON.md"},
          open(out, "w"), indent=1)
PY

echo "[rec] recording up to ${SECONDS_MAX}s into $OUT (Ctrl+C to stop; taps flush on exit)"
set +e
timeout --signal=INT "$((SECONDS_MAX + 5))" "$RT_BIN" "${ARGS[@]}" 2>&1 | tee "$OUT/pipeline.log"
set -e
ls -la "$OUT"
