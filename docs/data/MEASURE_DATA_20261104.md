# Recorded measurement data (2026-11-04): Google Drive

The raw recordings and per-take analysis outputs behind the 2026-11-04 docs are **not in git** (about 2.6 GB). They live on Google Drive:

**Drive folder:** https://drive.google.com/drive/folders/1gF2gqEybPuUn5eR00L4myzW5VXbS391c?usp=sharing

The Drive folder mirrors the local `measure/` directory. The upload was still in progress on 2026-11-04, so check it against the manifest before relying on it.

## Contents used by the docs

| Folder (under `measure/`) | Files | Size | WAVs | What it is |
|---|---:|---:|---:|---|
| `run_20261104_040645_1abaf4` | 670 | 0.95 GB | 218 | Valid 1 m azimuth sweep (13 takes) + elevation takes (pass/check) + offline analyses. **Protected** |
| `run_20261104_073046_3447eb` | 325 | 0.82 GB | 120 | First speech run (15 usable takes) + `analysis/speech_eval_20261104/` |
| `run_20261104_075009_b48fe6` | 316 | 0.73 GB | 112 | Speech × elevation × distance (14 takes incl. 1.5 m) + `analysis/speech_elev_eval_20261104/`, `speech_1p5m_eval_20261104/` |
| `mic_state_20261104` | 34 | 0.14 GB | 13 | Direct mic-state checks: single, 5 repeats, 5 USB power cycles, silence, stimulus |

Each take folder holds `raw_8ch.wav` (8 ch, 24 kHz, float32, device channel order, no mic gains), `mapped_6ch.wav`, `dry_ref.wav`, `wet_out.wav`, `stimulus_used.wav`, HATS WAVs (for polar takes), `trial.json`, `metrics.json` and `pipeline.log`.

## Verify a download

`docs/data/measure_20261104_manifest.json` lists every file with its size and **sha256**. From the repo root, after placing the Drive folders under `measure/`:

```python
import hashlib, json
from pathlib import Path
m = json.load(open("docs/data/measure_20261104_manifest.json"))
bad = []
for folder in m["folders"].values():
    for e in folder["entries"]:
        p = Path(e["path"])
        if not p.is_file() or p.stat().st_size != e["bytes"] or \
           hashlib.sha256(p.read_bytes()).hexdigest() != e["sha256"]:
            bad.append(e["path"])
print("all files OK" if not bad else f"{len(bad)} missing/mismatched, e.g. {bad[:5]}")
```

## Rules

- Do not modify `run_20261104_040645_1abaf4` (see `.cursor/rules/protected-polar-sweep-1m-20261104.mdc`). Write re-analysis outputs to new paths.
- Measurement data stays out of git. Use Drive and update this manifest when folders are added.
