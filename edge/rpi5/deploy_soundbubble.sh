#!/usr/bin/env bash
# Deploy entrypoint for sonitude-soundbubble.service.
# Launches the neural Sound Bubble realtime pipeline (ep35) with proven Pi settings.
set -euo pipefail

ROOT="${SOUNDBUBBLE_ROOT:-/home/sonitude/sb_rt_dev}"
PY="${SOUNDBUBBLE_PYTHON:-/home/sonitude/Sonitude/Sound_Bubble/.venv/bin/python}"
LOG_DIR="${SOUNDBUBBLE_LOG_DIR:-/home/sonitude/sb_rt_dev/logs}"
MIX_PORT="${MIX_UART_PORT:-/dev/ttyACM0}"

MODEL="${SOUNDBUBBLE_MODEL:-edge/zoo/optim_sonitude_dist_count_1_5m_ep35.onnx}"
CONTRACT="${SOUNDBUBBLE_CONTRACT:-edge/zoo/optim_sonitude_dist_count_1_5m_ep35.runtime.json}"
# Defaults = HARDWARE_CLASSES.md Class B (combined Pico cafe:4030).
# For Class A (Prototype 1 X1+INMP441) override via /etc/sonitude-soundbubble.env.
INPUT_DEVICE="${INPUT_DEVICE:-0}"
OUTPUT_DEVICE="${OUTPUT_DEVICE:-0}"
# Capture locked to model rate (24 kHz) — no input SRC. Class B output also 24 kHz.
INPUT_SR="${INPUT_SR:-24000}"
OUTPUT_SR="${OUTPUT_SR:-24000}"
INPUT_GAIN_DB="${INPUT_GAIN_DB:-0}"
SILENCE_DBFS="${SILENCE_DBFS:--80}"
OUTPUT_GAIN_DB="${OUTPUT_GAIN_DB:-0}"
if [[ -n "${OUTPUT_GAIN:-}" ]]; then
  echo "[deploy] warning: OUTPUT_GAIN (linear) is ignored; use OUTPUT_GAIN_DB (0 = unity)" >&2
fi
MIX_DRY="${MIX_DRY:-0.0}"
INTRA="${INTRA_OP_THREADS:-2}"
INTER="${INTER_OP_THREADS:-1}"
CHANNEL_MAP="${CHANNEL_MAP:-0,1,2,3,4,5}"

cd "$ROOT" || exit 1
mkdir -p "$LOG_DIR"
export PYTHONUNBUFFERED=1
# ROOT: edge.* package. ROOT/edge: top-level dummy_realtime / runtime_contract.
# SRC_ROOT: lab dummy_realtime imports src.utils (torch stack lives in the Sonitude tree).
SRC_ROOT="${SOUNDBUBBLE_SRC_ROOT:-/home/sonitude/Sonitude/Sound_Bubble}"
export PYTHONPATH="$ROOT:$ROOT/edge:$SRC_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -x "$PY" ]]; then
  echo "[deploy] missing python: $PY" >&2
  exit 1
fi
if [[ ! -f "$ROOT/$MODEL" ]]; then
  echo "[deploy] missing model: $ROOT/$MODEL" >&2
  exit 1
fi
if [[ ! -f "$ROOT/$CONTRACT" ]]; then
  echo "[deploy] missing contract: $ROOT/$CONTRACT" >&2
  exit 1
fi

# Device preflight (fails the unit so systemd can restart once USB audio is ready)
"$PY" - <<PY
import sounddevice as sd
sd.check_input_settings(device=int("${INPUT_DEVICE}"), channels=8, samplerate=int("${INPUT_SR}"), dtype="float32")
sd.check_output_settings(device=int("${OUTPUT_DEVICE}"), channels=2, samplerate=int("${OUTPUT_SR}"), dtype="float32")
print("[deploy] audio devices ok input=${INPUT_DEVICE} output=${OUTPUT_DEVICE}")
PY

# Shared flags only. Python gets -u -m … prepended; C++ rejects those tokens.
ARGS=(
  --model "$MODEL"
  --contract-path "$CONTRACT"
  --input-device "$INPUT_DEVICE"
  --output-device "$OUTPUT_DEVICE"
  --input-channels 8
  --output-channels 2
  --channel-map "$CHANNEL_MAP"
  --input-sr "$INPUT_SR"
  --output-sr "$OUTPUT_SR"
  --silence-dbfs "$SILENCE_DBFS"
  --input-gain-db "$INPUT_GAIN_DB"
  --output-gain-db "$OUTPUT_GAIN_DB"
  --mix-dry "$MIX_DRY"
  --intra-op-threads "$INTRA"
  --inter-op-threads "$INTER"
  --profile
  --monitor
)

RUNTIME="${SOUNDBUBBLE_RUNTIME:-python}"
RT_BIN="${SOUNDBUBBLE_RT_BIN:-/opt/soundbubble/bin/soundbubble-rt}"
# C++-only knobs (2026-11-04 runtime): per-mic linear trims by DEVICE channel,
# residual wet gate, optional always-on tap recording (TAP_DIR must be a new folder).
if [[ "$RUNTIME" == "cpp" ]]; then
  [[ -n "${MIC_GAINS:-}" ]] && ARGS+=(--mic-gains "$MIC_GAINS")
  [[ -n "${WET_GATE_DBFS:-}" ]] && ARGS+=(--wet-gate-dbfs "$WET_GATE_DBFS")
  if [[ -n "${TAP_DIR:-}" ]]; then
    TAP_RUN="$TAP_DIR/$(date +%Y%m%d_%H%M%S)"; mkdir -p "$TAP_RUN"
    ARGS+=(--tap-dir "$TAP_RUN" --tap-select "${TAP_SELECT:-raw,wet}" --tap-max-seconds "${TAP_MAX_SECONDS:-900}")
  fi
fi
# The knob receiver lives in soundbubble-rt. The Python pipeline does not
# implement --mix-uart-port, so those flags are passed only to the C++ runtime.
if [[ "$RUNTIME" == "cpp" && -e "$MIX_PORT" ]]; then
  echo "[deploy] MIX UART enabled on $MIX_PORT"
  ARGS+=(--mix-uart-port "$MIX_PORT" --mix-uart-baud "${MIX_UART_BAUD:-115200}")
elif [[ -e "$MIX_PORT" ]]; then
  echo "[deploy] MIX UART port $MIX_PORT present; knob control requires SOUNDBUBBLE_RUNTIME=cpp"
else
  echo "[deploy] MIX UART port $MIX_PORT missing; continuing without mix control"
fi

echo "[deploy] starting neural Sound Bubble pipeline runtime=$RUNTIME"
if [[ "$RUNTIME" == "cpp" ]]; then
  if [[ ! -x "$RT_BIN" ]]; then
    echo "[deploy] missing soundbubble-rt: $RT_BIN" >&2
    exit 1
  fi
  exec "$RT_BIN" "${ARGS[@]}"
fi
exec "$PY" -u -m edge.inference_pipeline "${ARGS[@]}"
