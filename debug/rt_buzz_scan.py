"""Scan tap WAVs for RT discontinuities: exact-zero holes inside active audio,
large sample jumps, and block-periodic energy (buzz). Read-only."""
import sys, numpy as np, soundfile as sf
for p in sys.argv[1:]:
    x, sr = sf.read(p, always_2d=True); x = x[:, 0]
    n = len(x); blk = 192
    # activity envelope (20 ms)
    w = int(0.02 * sr); env = np.sqrt(np.convolve(x**2, np.ones(w)/w, 'same'))
    active = env > 10**(-50/20)
    # zero runs >= 24 samples inside active
    z = (x == 0)
    runs = []; i = 0
    while i < n:
        if z[i]:
            j = i
            while j < n and z[j]: j += 1
            if j - i >= 24 and active[max(0,i-w)] and active[min(n-1,j+w)]: runs.append((i, j-i))
            i = j
        else: i += 1
    d = np.abs(np.diff(x)); thr = 8 * (np.median(d[active[1:]]) + 1e-9) if active.any() else 1
    jumps = np.where((d > max(thr, 0.05)) & active[1:])[0]
    jp = np.bincount(jumps % blk, minlength=blk) if len(jumps) else np.zeros(blk)
    print(f"{p.split('measure')[-1]}  sr={sr} dur={n/sr:.1f}s active={active.mean():.0%} peak={np.abs(x).max():.3f}")
    print(f"   zero-holes(>=1ms) in active: {len(runs)}  lens(ms) {sorted(set(round(l/sr*1000,1) for _,l in runs))[:8]}  pos%192 {sorted(set(s%blk for s,_ in runs))[:8]}")
    print(f"   big jumps: {len(jumps)}  top phases mod 192: {np.argsort(jp)[::-1][:5].tolist()} counts {np.sort(jp)[::-1][:5].astype(int).tolist()}")
