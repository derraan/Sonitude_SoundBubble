"""Per-sample SNRLP balance diagnostic (read-only; no checkpoint writes).

For each checkpoint x dataset: on val clips (no perturbations), report
negative-clip output level / LogPower term and positive-clip SNR term.
"""
import argparse, copy, json, sys
import numpy as np, torch
sys.path.insert(0, ".")
from src import utils
from src.losses.LogPowerLoss import LogPowerLoss
from src.losses.SNRLosses import SNRLosses

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="real_experiments/optim_finetune_ep24_ep35style_synreal_lean.json")
ap.add_argument("--ckpt", nargs="+", required=True)
ap.add_argument("--n", type=int, default=48)
args = ap.parse_args()
cfg = json.load(open(args.config))
lp, snr = LogPowerLoss(), SNRLosses("snr")
dev = torch.device("cuda")

def dbfs(x):
    return 10 * torch.log10(torch.mean(x.double() ** 2) + 1e-20).item()

for ck in args.ckpt:
    pl_args = copy.deepcopy(cfg["pl_module_args"]); pl_args["init_ckpt"] = ck
    pl_args.get("distance_sweep_val", {})["enabled"] = False
    hl = utils.import_attr(cfg["pl_module"])(**pl_args)
    model = hl.model.to(dev).eval()
    print(f"\n=== {ck}")
    for dd in cfg["val_data_args"]["dataset_dirs"]:
        va = copy.deepcopy(cfg["val_data_args"]); va["dataset_dirs"] = [dict(dd, max_samples=args.n)]
        ds = utils.import_attr(cfg["train_dataset"])(**va, split="val")
        neg_out, neg_lp, pos_snr, pos_out_rel = [], [], [], []
        with torch.no_grad():
            for i in range(len(ds)):
                inp, tgt = ds[i]
                mix = inp["mixture"].unsqueeze(0).to(dev)
                est = model({"mixture": mix})["output"].float()
                gt = tgt["target"].unsqueeze(0).to(dev)
                if gt.abs().max() == 0:
                    neg_out.append(dbfs(est)); neg_lp.append(lp(est, gt).item())
                else:
                    pos_snr.append(-snr(est, gt).item()); pos_out_rel.append(dbfs(est) - dbfs(gt))
        f = lambda a: f"{np.median(a):7.1f} [p10 {np.percentile(a,10):6.1f} p90 {np.percentile(a,90):6.1f}]" if a else "   n/a"
        print(f" {dd['path'].split('datasets/')[-1]:45s} neg n={len(neg_out):2d} out_dBFS {f(neg_out)} LP*60 {f([x*60 for x in neg_lp])}")
        print(f" {'':45s} pos n={len(pos_snr):2d} SNR_dB  {f(pos_snr)} est-gt dB {f(pos_out_rel)}")
    del hl, model; torch.cuda.empty_cache()
