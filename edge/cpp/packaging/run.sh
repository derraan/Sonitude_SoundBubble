#!/usr/bin/env bash
# Installed to /opt/soundbubble/bin/run.sh. Dispatches the Python fallback or soundbubble-rt.
set -euo pipefail

ROOT="${SOUNDBUBBLE_ROOT:-/home/sonitude/sb_rt_dev}"
RUNTIME="${SOUNDBUBBLE_RUNTIME:-python}"
LOG_DIR="${SOUNDBUBBLE_LOG_DIR:-$ROOT/logs}"
PY="${SOUNDBUBBLE_PYTHON:-python3}"
RT_BIN="${SOUNDBUBBLE_RT_BIN:-/opt/soundbubble/bin/soundbubble-rt}"

MODEL="${SOUNDBUBBLE_MODEL:-edge/zoo/optim_pretrain_sonitude_dist_count_1_5m_ep36.onnx}"
CONTRACT="${SOUNDBUBBLE_CONTRACT:-edge/zoo/optim_pretrain_sonitude_dist_count_1_5m_ep36.runtime.json}"
INPUT_DEVICE="${INPUT_DEVICE:-0}"
OUTPUT_DEVICE="${OUTPUT_DEVICE:-0}"
INPUT_SR="${INPUT_SR:-24000}"
OUTPUT_SR="${OUTPUT_SR:-24000}"
INPUT_GAIN_DB="${INPUT_GAIN_DB:-0}"
SILENCE_DBFS="${SILENCE_DBFS:--80}"
OUTPUT_GAIN_DB="${OUTPUT_GAIN_DB:-0}"
MIX_DRY="${MIX_DRY:-0.0}"
INTRA="${INTRA_OP_THREADS:-2}"
INTER="${INTER_OP_THREADS:-1}"
CHANNEL_MAP="${CHANNEL_MAP:-0,1,2,3,4,5}"
MIX_PORT="${MIX_UART_PORT:-/dev/ttyACM0}"
MIX_BAUD="${MIX_UART_BAUD:-115200}"

cd "$ROOT"
mkdir -p "$LOG_DIR"

resolve() {
  local p="$1"
  if [[ "$p" = /* ]]; then
    printf '%s\n' "$p"
  else
    printf '%s\n' "$ROOT/$p"
  fi
}

MODEL_PATH="$(resolve "$MODEL")"
CONTRACT_PATH="$(resolve "$CONTRACT")"
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "[deploy] missing model: $MODEL_PATH" >&2
  exit 1
fi
if [[ ! -f "$CONTRACT_PATH" ]]; then
  echo "[deploy] missing contract: $CONTRACT_PATH" >&2
  exit 1
fi

# Stay inside one start attempt while USB audio enumerates, so RestartSec/StartLimitBurst
# are not burned by a fast preflight failure.
ready=0
for _ in $(seq 1 30); do
  if "$RT_BIN" --preflight --input-device "$INPUT_DEVICE" --output-device "$OUTPUT_DEVICE" \
      --input-channels 8 --output-channels 2 --input-sr "$INPUT_SR" --output-sr "$OUTPUT_SR" \
      --model "$MODEL_PATH" --contract-path "$CONTRACT_PATH" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done
if [[ "$ready" -ne 1 ]]; then
  echo "[deploy] audio devices not ready" >&2
  exit 1
fi

ARGS=(
  --model "$MODEL_PATH"
  --contract-path "$CONTRACT_PATH"
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

if [[ -e "$MIX_PORT" ]]; then
  echo "[deploy] MIX UART enabled on $MIX_PORT"
  ARGS+=(--mix-uart-port "$MIX_PORT" --mix-uart-baud "$MIX_BAUD")
fi

echo "[deploy] runtime=$RUNTIME"
if [[ "$RUNTIME" == "cpp" ]]; then
  exec "$RT_BIN" "${ARGS[@]}"
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ ! -x "$PY" && ! -x "$(command -v "$PY" || true)" ]]; then
  echo "[deploy] missing python: $PY" >&2
  exit 1
fi
exec "$PY" -u -m edge.inference_pipeline "${ARGS[@]}"
