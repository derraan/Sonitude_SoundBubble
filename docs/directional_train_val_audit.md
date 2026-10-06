# Directional train / val audit

**Date:** 2026-10-06  
**Scope:** DSENet β steered head on Sound Bubble (`optim_pretrain_sonitude_dist_count_directional_1_5m`)  
**Promote path:** untouched (`runs/optim_pretrain_sonitude_dist_count_1_5m`)

## Spec alignment

| Check | Result | Notes |
|---|---|---|
| Hard bubble GT unchanged | **PASS** | Dataset `target` = sum of inside ref stems; β only builds `target_steered` |
| 6 dB look-delta cap | **PASS** | `look_delta_cap_db=6`; metrics clamp; loss = `relu(6 − Δ)` |
| 360° / 30° sweep | **PASS** | `make_theta0_grid_deg(30)` → 12 looks `[0…330]`; configs updated |
| Promote untouched | **PASS** | `init_ckpt` still promote `best.pt`; no writes to promote checkpoints |

## Dataset / GT pipeline

| Check | Result | Notes |
|---|---|---|
| Angle metadata key | **PASS** (with fix) | Libri packaged metadata uses `angle` (sample verified). Loader now accepts `angle` **or** `angle_deg` fallback for generator-side keys |
| Angle coverage | **INFO** | Libri scenes include bearings beyond ±90° (e.g. 153°, −144°) — full-circle look grid is appropriate |
| All inside vs K=2 slots | **DOCUMENTED** | Steered GT sums **all** inside speakers; count/distance slots keep closest 2 only |
| β modulation baseline | **DOCUMENTED** | Oracle no-look = stem-sum bubble; train look-delta no-look = **model** \(\hat b\) — intentional (oracle for target construction, model-centric for loss) |
| σ search / 6 dB cap | **PASS** | Unit test asserts `capped_delta_db ≤ 6` |
| Perturbations | **PASS** | Directional config uses empty `perturbations`; when enabled, `target_steered` is stacked with bubble/slots |

## Train / val evaluation semantics

| Check | Result | Notes |
|---|---|---|
| Primary loss θ₀ | **PASS** | Batch `inputs.theta0_deg` from look-toward-speaker sampling |
| Look-delta loss | **PASS** | Single `looked_spk_idx` vs \(\hat b\); matches “looked speaker” intent |
| θ₀ sweep metric | **PASS** / **DOCUMENTED** | For each grid θ₀, speaker = nearest angle to that θ₀. Not “both speakers at batch look” — measures directionality vs look angle |
| Sweep baseline | **PASS** | Uses training-step \(\hat b\) |
| Metric keys | **PASS** | Normalized `look_delta_db_th000` … `th330` (0–360, zero-padded) |
| Train sweep cost | **PASS** | `theta0_sweep_every_n: 4` (12 forwards × every 4th train step); val every batch |
| Negative early deltas | **EXPECTED** | Untrained steered head yields negative Δ; track over epochs |

## Model / steered head

| Check | Result |
|---|---|
| θ₀ cyclic embed + FiLM | **PASS** (`SteeredHead`) |
| `theta0_rad` plumbing | **PASS** (`net.forward`) |
| Frozen backbone, steered-only train | **PASS** (`train_count_heads: false`) |

## Scripts / deploy

| Check | Result |
|---|---|
| Stage G default grid | **PASS** (imports `DEFAULT_THETA0_GRID_DEG` → 12-point) |
| Stage I β gains σ/ρ | **PASS** (same 0.2 / 8.0 defaults) |
| Edge `--beta-theta0-deg` | **PASS** (accepts any float; circular β) |

## Follow-ups (optional)

1. Log per-slot `look_delta_db_spk0/spk1` at **batch** θ₀.
2. Analysis-only metric with oracle-bubble no-look for apples-to-apples vs β modulation.
3. Subsampled train sweep if 12× forward remains too slow even with `every_n=4`.

## How to re-run audit smoke

```powershell
python -m pytest tests/test_dsenet_beta_grid.py -q
python -c "from src.datasets.dsenet_beta import DEFAULT_THETA0_GRID_DEG, nearest_theta0_deg; assert len(DEFAULT_THETA0_GRID_DEG)==12; assert nearest_theta0_deg(350)==0.0; print('ok', DEFAULT_THETA0_GRID_DEG)"
```
