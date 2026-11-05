# Microphone index canon (three layers)

**Owner:** human only (Darren).  
**Agents:** READ ONLY — do not edit, rewrite, “fix”, or invent values in this file.  
**Status:** B5 remains OPEN until the owner fills §4 from a measured tap probe.

This document is the single place that defines what “channel / mic index” means across training, Pico firmware, and Pi realtime. If code and this file disagree, update the code to match this file — never the other way around — and only the owner edits this file.

---

## 1. Model / training index (authoritative for the neural net)

Model input dimension `i` and dataset channel `mic0i` are the **vertical-arc headphone** roles below. Coordinates are centimetres relative to the array centre (same table as `generate_adaptive_dataset.py`).

| Model / train index | Dataset key | Role | x (cm) | y (cm) | z (cm) |
|---------------------|-------------|------|--------|--------|--------|
| 0 | mic00 | Left ear | −12.8 | −1.5 | 0.0 |
| 1 | mic01 | Headband L-outer | −10.2 | 0.0 | 11.3 |
| 2 | mic02 | Headband L-inner (top) | −3.8 | 0.0 | 16.9 |
| 3 | mic03 | Headband R-inner (top) | 3.8 | 0.0 | 16.9 |
| 4 | mic04 | Headband R-outer | 10.6 | 0.0 | 11.7 |
| 5 | mic05 | Right ear | 13.1 | −1.5 | 0.7 |

- Ear span ≈ 25.9 cm; vertical span ≈ 16.9 cm.
- `mixture.wav` channel `k` ≡ `mic0k` ≡ model input `k` during training.
- Train configs use `mic_config: ["mic00","mic01","mic02","mic03","mic04","mic05"]`.

**Do not renumber this table lightly** — weights are tied to these roles.

---

## 2. Pico USB / I2S device index (firmware packing order)

Firmware packs UAC2 frames from I2S lanes (not from ear/arc roles). After `decode_sample`, USB channels are:

| USB device channel | Firmware source | Electrical meaning |
|--------------------|-----------------|--------------------|
| 0 | `sample_l[0]` | SD0, WS left slot |
| 1 | `sample_r[0]` | SD0, WS right slot |
| 2 | `sample_l[1]` | SD1, left |
| 3 | `sample_r[1]` | SD1, right |
| 4 | `sample_l[2]` | SD2, left |
| 5 | `sample_r[2]` | SD2, right |
| 6 | (zero) | padding |
| 7 | (zero) | padding |

Pins (Pico 2W path): SCK GPIO0, WS GPIO1, SD0 GPIO2, SD1 GPIO3, SD2 GPIO4, SD3 unused/low GPIO5.

**There is no firmware claim that USB ch0 = left ear.** Lane↔physical mic is board wiring.

---

## 3. Pi realtime map (bridge)

Runtime rule:

```text
model_input[i] = device_channel[ channel_map[i] ]
```

- `--channel-map` / contract `expected_channel_map` is a list of six device indices.
- Default in code today: `0,1,2,3,4,5` (identity). That default is **not** proven correct for hardware.
- Launcher: `run_pi_live_pipeline.ps1 -ChannelMap <this file §4>`.

---

## 4. Authoritative mapping (owner-filled — required for B5 closed)

Fill after a successful tap probe (`edge/tools/channel_probe.py tap` / `run_channel_probe.ps1 -Mode tap`).  
Agents must leave this section untouched.

| Model index (role) | USB device channel | Notes / date |
|--------------------|--------------------|--------------|
| 0 left ear | _TBD_ | |
| 1 headband L-outer | _TBD_ | |
| 2 headband L-inner | _TBD_ | |
| 3 headband R-inner | _TBD_ | |
| 4 headband R-outer | _TBD_ | |
| 5 right ear | _TBD_ | |

**Canonical `--channel-map` string (owner):**

```text
TBD,TBD,TBD,TBD,TBD,TBD
```

Example once filled: `2,0,1,3,5,4` means model left-ear reads USB ch2, etc.

**Probe / firmware notes (owner):**

- Date:
- Pico UF2 / commit:
- Headset description:
- Clap sync (B6) result:

---

## 5. Consistency rules

1. Training and model indices always mean §1 roles.
2. Pico indices always mean §2 lane packing.
3. Only §4 equates §1 ↔ §2; until §4 is filled, B5 stays OPEN.
4. Agents may **read** this file and may suggest a measured map in chat, but must **not** edit this file or silently bake an unverified map into launcher defaults without the owner updating §4 first.
5. Related audit: `edge/LECTURE_HALL_LEAKAGE_FAULT_AUDIT.md` (B4/B5). This canon file wins on index definitions.
