# Dashboard models (2026-11-05)

ONNX exports selectable in the dashboard (Settings → Bubble size / Model). Same files and names as `~/sb_rt_dev/edge/zoo/` on the Pi (sha256 verified against the Pi copies). Opset 9, onnx.checker OK, Torch↔ORT equivalence OK. Checkpoints (`*.pt`) are **not** in git.

| Dashboard | Radius | File | Source checkpoint | sha256 (onnx) |
|---|---|---|---|---|
| V1 · EP35 | 1.5 m | `optim_sonitude_dist_count_1_5m_ep35.onnx` + `.runtime.json` | existing promoted zoo export (production default) | `249e0490cb56265b…` |
| V2 · EP25 | 1.5 m | `optim_v2_ep25_eval_20261104.onnx` + `.runtime.json` | v2 run `best.pt`, epoch 25 | `707bd865f44348de…` |
| V3 · EP37 | 1.5 m | `optim_v2_ep37_eval_20261105.onnx` + `.runtime.json` | v2 run `last.pt`, epoch 37 | `9a51cdc740a1d17e…` |
| Pretrain 1 m | 1.0 m | `optim_pretrain_1m_20261105.onnx` + `.runtime.json` | `runs/optim_pretrain_1m` `best.pt`, epoch 80 (synthetic only) | `d03de36f86961a5b…` |
| Pretrain 2 m | 2.0 m | `optim_pretrain_2m_20261105.onnx` + `.runtime.json` | `runs/optim_pretrain_2m` `best.pt`, epoch 48 (synthetic only) | `ab65b2b8aa6ae488…` |

- **Evaluations:** `docs/eval_v2_ep37_20261105.md`, `docs/training_status_v2_and_v3_improvements_20261105.md` §4.
- **Eval-only exports:** `*_eval_*` and `*_pretrain_*_20261105` are not promoted. Never overwrite them in place; new iterations get new names (`.cursor/rules/local-checkpoint-files.mdc`).
