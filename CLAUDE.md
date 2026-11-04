# Sound_Bubble — agent guardrails

This repo is co-developed with Cursor. The guardrails in `.cursor/rules/*.mdc` apply to **all** agents, Claude Code included. Read them before deleting, moving or overwriting anything under `runs/`, `edge/zoo/`, `measure/` or `firmware/`.

Protected (never delete, move or overwrite; ask the owner first):

- **Valid polar sweep @ 1 m (EP35):** `measure/run_20261104_040645_1abaf4/` (entire folder, including `analysis/` and `analysis/offline_ep35_20261104/`), plus `docs/polar_sweep_1m_ep35_20261104.md`, `docs/img/polar_offline_ep35_20261104/` , `docs/img/offline_elev_norm_20261104/` and `docs/img/offline_elev_wideband_20261104/`. See `.cursor/rules/protected-polar-sweep-1m-20261104.mdc`.
- **EP24 mic0m25 sacred set:** see `.cursor/rules/promoted-ep24-sacred.mdc`.
- **Checkpoints:** see `.cursor/rules/local-checkpoint-files.mdc`.

Read-only for agents: `edge/MIC_INDEX_CANON.md` and the Pico firmware design docs (see their rules).

Latest consolidated findings (training, runtime, testbench, mic hardware, EP35 on the real array): `docs/FINDINGS_20261104.md`.
