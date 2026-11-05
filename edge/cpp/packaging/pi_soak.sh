#!/usr/bin/env bash
# Capture a timed profile log from the running sonitude-soundbubble unit.
# Usage: pi_soak.sh [--minutes N] [--unit sonitude-soundbubble]
set -euo pipefail
MINUTES=30
UNIT=sonitude-soundbubble
while [[ $# -gt 0 ]]; do
  case "$1" in
    --minutes) MINUTES="$2"; shift 2 ;;
    --unit) UNIT="$2"; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
if ! systemctl is-active --quiet "$UNIT"; then
  echo "$UNIT is not active" >&2
  exit 1
fi
LOG="${SOUNDBUBBLE_LOG_DIR:-/tmp}/soak_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG")"
echo "[soak] following $UNIT for ${MINUTES} min -> $LOG"
timeout "$((MINUTES * 60))" journalctl -u "$UNIT" -f -o cat | tee "$LOG" || true
if ! systemctl is-active --quiet "$UNIT"; then
  echo "[soak] unit died during the window" >&2
  exit 1
fi
echo "[soak] still active. profile lines: $(grep -c '\[profile\]' "$LOG" || true)"
grep '\[profile\]' "$LOG" | awk '
  /steps=/ {
    for (i = 1; i <= NF; i++) if ($i ~ /^steps=/) { split($i, a, "="); steps += a[2]; n++ } }
  END { if (n) printf "[soak] mean steps=%.2f over %d lines\n", steps / n, n }
'
