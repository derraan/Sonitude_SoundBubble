"""Look for block-rate harmonic combs (buzz) in a recording. Read-only."""
import sys, numpy as np, soundfile as sf
from scipy.signal import welch
for p in sys.argv[1:]:
    x, sr = sf.read(p, always_2d=True); x = x[:, 0]
    f, P = welch(x, sr, nperseg=sr // 2)   # 2 Hz bins
    Pdb = 10*np.log10(P + 1e-20)
    def line(fr):  # peak vs local median (+-30 Hz)
        i = int(round(fr / (f[1]-f[0]))); lo, hi = max(1,i-15), i+16
        return Pdb[i-1:i+2].max() - np.median(Pdb[lo:hi])
    out = []
    for base in (125.0, 250.0):
        prom = [line(base*k) for k in range(1, 13) if base*k < sr/2-50]
        out.append(f"{int(base)}Hz comb: mean prominence {np.mean(prom):5.1f} dB  (h1..h6 {[round(v,1) for v in prom[:6]]})")
    ctrl = np.mean([line(fr) for fr in (173, 311, 457, 619, 883, 1171)])
    print(p.split('measure')[-1], "| control non-harmonic", round(ctrl,1), "dB"); [print("   ", o) for o in out]
