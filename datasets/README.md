# Datasets (Google Drive — not in GitHub)

Raw audio is too large for GitHub (~100GB+). Keep datasets on **Google Drive**, then sync/copy them into this folder locally as:

```
datasets/
  syn_1m/          (or syn_1m.tar)
  syn_1_5m/        (or syn_1_5m.tar)
  syn_2m/          (or syn_2m.tar)
  syn_test/        (or syn_test.tar)
  LibriSpeech/
    train-clean-100/
    dev-clean/
  syn_libri_edge_1_35m_ls/   # optional stopband set (generated)
  vctk_split.json
  WHAM_split.json
```

## Suggested Drive layout

Create a private Drive folder, e.g. `Sonitude_SoundBubble_datasets`, and upload:

1. Prefer the `.tar` archives (smaller to manage as single files):
   - `syn_1m.tar`, `syn_1_5m.tar`, `syn_2m.tar`, `syn_test.tar`
2. LibriSpeech:
   - `train-clean-100.tar.gz`, `dev-clean.tar.gz`
   - or the extracted `LibriSpeech/` tree
3. Split metadata:
   - `vctk_split.json`, `WHAM_split.json`

After download, extract archives under `datasets/` so training configs keep working.

## Local source on this machine

Current canonical local copy:

`C:\Users\darre\Sound_Bubble\datasets`

Upload from there to Drive (web UI, Drive for desktop, or `rclone`).

## Generate stopband edge set (after LibriSpeech is present)

```powershell
python scripts\generate_libri_edge_dataset.py `
  --speech-dir datasets\LibriSpeech\train-clean-100 `
  --val-speech-dir datasets\LibriSpeech\dev-clean `
  --output-path datasets\syn_libri_edge_1_35m_ls `
  --n-train 3000 --n-val 400 --duration 5.0 `
  --dis-threshold 1.35 --outside-gap 0.08 `
  --p-n-in0 0.5 --p-n-in1 0.25 --seed 12
```
