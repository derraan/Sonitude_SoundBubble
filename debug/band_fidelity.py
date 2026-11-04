"""Per-band fidelity of model output vs ground-truth target (lag-aligned).
Band SNR = target band energy / error band energy after optimal scaling."""
import sys, numpy as np, soundfile as sf
from scipy.signal import butter, sosfiltfilt
gt, sr = sf.read(sys.argv[1]); gt = gt if gt.ndim == 1 else gt[:, 0]
mix, _ = sf.read(sys.argv[2]); mix = mix[:, 0] if mix.ndim > 1 else mix
bands = [(80, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 6000), (6000, 8000), (8000, 11500)]
def align(y):
    n = min(len(gt), len(y)); from scipy.signal import correlate; c = correlate(y[:n], gt[:n], 'full', method='fft'); lag = int(np.argmax(c)) - (n - 1)
    return np.roll(y[:n], -lag), gt[:n], lag
def bp(x, lo, hi): return sosfiltfilt(butter(4, [lo, hi], 'band', fs=sr, output='sos'), x)
print("band(Hz)        " + "".join(f"{lo}-{hi}".rjust(11) for lo, hi in bands))
for p in [sys.argv[2]] + sys.argv[3:]:
    y = mix if p == sys.argv[2] else sf.read(p)[0]
    y = y[:, 0] if y.ndim > 1 else y
    y, g, lag = align(y); y, g = y[2400:], g[2400:]
    row = []; lvl = []
    for lo, hi in bands:
        yb, gb = bp(y, lo, hi), bp(g, lo, hi)
        a = np.dot(yb, gb) / (np.dot(gb, gb) + 1e-20)
        row.append(10*np.log10(np.sum((a*gb)**2) / (np.sum((yb - a*gb)**2) + 1e-20) + 1e-20))
        lvl.append(10*np.log10(np.sum(yb**2)/(np.sum(gb**2)+1e-20)+1e-20))
    name = "INPUT mic0" if p == sys.argv[2] else p.split('/')[-1]
    print(f"{name:16s}" + "".join(f"{v:11.1f}" for v in row) + f"   (lag {lag})")
    print(f"{'  out/gt dB':16s}" + "".join(f"{v:11.1f}" for v in lvl))
