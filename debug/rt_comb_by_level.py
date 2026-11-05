"""Comb (hop-rate harmonic) energy share vs wet level, in 0.5 s windows."""
import sys, numpy as np, soundfile as sf
for p in sys.argv[1:]:
    y, sr = sf.read(p); y = y if y.ndim == 1 else y[:, 0]; w = sr // 2
    rows = []
    for i in range(0, len(y) - w, w // 2):
        s = y[i:i+w]; Y = np.abs(np.fft.rfft(s * np.hanning(w)))**2; f = np.fft.rfftfreq(w, 1/sr)
        h = np.zeros_like(f, bool)
        for k in range(1, 80): h |= np.abs(f - 125*k) < 2.5
        rows.append((20*np.log10(np.std(s)+1e-12), Y[h].sum()/Y.sum(), h.mean()))
    r = np.array(rows); print(p.split('/')[-1] if 'measure' not in p else p.split('measure')[-1], f"(random share would be {r[0,2]:.0%})")
    for lo, hi in [(-200,-80),(-80,-65),(-65,-50),(-50,-35),(-35,0)]:
        m = (r[:,0]>=lo)&(r[:,0]<hi)
        if m.any(): print(f"   level {lo:4d}..{hi:4d} dBFS  n={m.sum():3d}  comb share {r[m,1].mean():.0%}")
