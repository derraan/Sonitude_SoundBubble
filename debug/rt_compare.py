import sys, numpy as np, soundfile as sf
ref, _ = sf.read(sys.argv[1]); ref = ref if ref.ndim == 1 else ref[:, 0]
for p in sys.argv[2:]:
    y, _ = sf.read(p); y = y if y.ndim == 1 else y[:, 0]; n = min(len(ref), len(y))
    a, b = ref[:n], y[:n]; c = np.correlate(b, a, 'full'); lag = int(np.argmax(np.abs(c))) - (n - 1)
    a2 = np.roll(a, lag)
    print(f"{p:60s} rms {20*np.log10(np.std(b)+1e-12):6.1f} dB (ref {20*np.log10(np.std(a)+1e-12):6.1f})  lag {lag:5d}  corr {np.abs(c).max()/np.sqrt((a**2).sum()*(b**2).sum()):.3f}")
