#!/usr/bin/env bash
# One-time: let soundbubble-ctl stop the production unit on Settings → Restart.
set -euo pipefail
if [[ "$EUID" -ne 0 ]]; then
  echo "run with sudo" >&2
  exit 1
fi
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$(cd "$(dirname "$0")" && pwd)/sudoers-sonitude-soundbubble-ctl"
DST=/etc/sudoers.d/sonitude-soundbubble-ctl
install -m 440 "$SRC" "$DST"
visudo -cf "$DST"
echo "[sudoers] installed $DST"
sudo -u sonitude -n systemctl is-active sonitude-soundbubble >/dev/null 2>&1 || true
echo "[sudoers] test: sudo -u sonitude -n systemctl stop sonitude-soundbubble"
sudo -u sonitude -n systemctl stop sonitude-soundbubble || true
echo "[sudoers] ok"
