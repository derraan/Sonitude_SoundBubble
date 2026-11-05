"""Band-averaged magnitude-squared coherence between model output and mic0 input
(lag-aligned). Low coherence = output not linearly derived from input (artifacts)."""
import sys, numpy as np, soundfile as sf
from scipy.signal import coherence, correlate
bands = [(80, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 6000), (6000, 8000), (8000, 11500)]
print(f"{'':28s}" + "".join(f"{lo}-{hi}".rjust(11) for lo, hi in bands) + "   out dBFS")
for inp, out, ch in [a.split(":") for a in sys.argv[1:]]:
    x, sr = sf.read(inp, always_2d=True); x = x[:, int(ch)]
    y, _ = sf.read(out); y = y[:, 0] if y.ndim > 1 else y
    n = min(len(x), len(y)); x, y = x[:n], y[:n]
    lag = int(np.argmax(np.abs(correlate(y, x, 'full', method='fft')))) - (n - 1)
    y = np.roll(y, -lag)[2400:n - 2400]; x = x[2400:n - 2400]
    f, C = coherence(x, y, sr, nperseg=1024)
    row = [C[(f >= lo) & (f < hi)].mean() for lo, hi in bands]
    print(f"{out.split('/')[-1]:28s}" + "".join(f"{v:11.2f}" for v in row) + f"   {20*np.log10(np.std(y)+1e-12):6.1f}")
