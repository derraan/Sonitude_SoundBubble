# Firmware (vendored)

## `mic-array-pico2w-usb6ch`

INMP441 6+2 channel Pico 2W USB Audio Class 2 (UAC2) mic-array firmware, vendored into this repo for Pi/edge bring-up and audit.

- Sources and build files live under [`mic-array-pico2w-usb6ch/`](mic-array-pico2w-usb6ch/).
- Design documentation (`README.md`, `OVERRUN.md`, `skills_planner.md`, `docs/**`) is **owner-only** — agents must not edit those files (see `.cursor/rules/pico-firmware-design-docs-readonly.mdc`).
- Build artifacts (`build*/`, `.cache/`) and capture WAVs are gitignored; there is no nested `.git`.
