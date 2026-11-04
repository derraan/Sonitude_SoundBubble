"""Replug series: the operator physically unplugs/replugs the Pico (USB power cycle);
this script detects each disconnect and reconnect, re-resolves device indices by name,
and runs debug/mic_state_check.py after every reconnect. N cycles (default 5).

Log: debug/mic_replug_series_<utc>.log (+ the per-run mic_state_check_*.json/.wav).
"""
import subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import sounddevice as sd

REPO = Path(__file__).resolve().parents[1]
PICO = "INMP441 8ch"
STIM = "QUAD-CAPTURE"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
LOG = REPO / "debug" / f"mic_replug_series_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"


def log(msg):
    line = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def scan():
    sd._terminate(); sd._initialize()  # refresh PortAudio's device list
    pico = stim = None
    for i, d in enumerate(sd.query_devices()):
        h = sd.query_hostapis(d["hostapi"])["name"]
        if "WDM-KS" not in h:
            continue
        if PICO in d["name"] and d["max_input_channels"] >= 8:
            pico = i
        if STIM in d["name"] and d["max_output_channels"] >= 2 and d["name"].startswith("1-2"):
            stim = i
    return pico, stim


def wait_for(present: bool, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        pico, stim = scan()
        if (pico is not None) == present:
            return pico, stim
        time.sleep(0.5)
    raise TimeoutError(f"Pico {'reconnect' if present else 'disconnect'} not seen within {timeout}s")


log(f"replug series start, {N} cycles; log={LOG.name}")
for k in range(1, N + 1):
    log(f"cycle {k}/{N}: >>> UNPLUG the Pico now <<<")
    wait_for(False)
    log(f"cycle {k}/{N}: disconnected. >>> REPLUG the Pico now <<<")
    pico, stim = wait_for(True)
    log(f"cycle {k}/{N}: reconnected (pico in={pico}, stimulus out={stim}); settling 3 s")
    time.sleep(3.0)
    pico, stim = scan()
    r = subprocess.run([sys.executable, str(REPO / "debug" / "mic_state_check.py"), "--in", str(pico), "--out", str(stim)],
                       capture_output=True, text=True, cwd=str(REPO))
    out = [l for l in r.stdout.splitlines() if l.strip() and "warn" not in l.lower()]
    for l in out:
        log(f"cycle {k}/{N}:   {l}")
    if r.returncode != 0:
        log(f"cycle {k}/{N}: check FAILED rc={r.returncode}: {r.stderr.strip()[-400:]}")
log("replug series done")
