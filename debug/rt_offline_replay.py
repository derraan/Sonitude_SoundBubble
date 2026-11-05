"""Offline ONNX replay of a mapped_6ch tap through the streaming model (no RT
rings/threads) to see whether the hop-rate comb is model-inherent. Read-only."""
import sys, json, numpy as np, soundfile as sf
sys.path.insert(0, "edge")
from pipeline.onnx_streamer import ONNXStreamer as OnnxStreamer
mapped, onnx, contract, out, gain_db = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5])
c = json.load(open(contract))
chunk = int(c.get("model_chunk", c.get("stft_chunk_size", 192))); pad = int(c.get("model_pad", c.get("stft_pad_size", 96)))
x, sr = sf.read(mapped, always_2d=True, dtype="float32"); x = x.T * np.float32(10 ** (gain_db / 20))
x = x - x.mean(axis=1, keepdims=True)
s = OnnxStreamer(onnx, 1, 1); s.set_radius(1.5) if hasattr(s, "set_radius") else None
frame = np.zeros((1, x.shape[0], chunk + pad), np.float32); ys = []
for i in range(0, x.shape[1] - chunk + 1, chunk):
    frame[:, :, :-chunk] = frame[:, :, chunk:]; frame[0, :, -chunk:] = x[:, i:i + chunk]
    ys.append(np.asarray(s.run(frame).output).reshape(-1)[:chunk])
y = np.concatenate(ys); sf.write(out, y, sr); print("chunk", chunk, "pad", pad, "wrote", out, "peak", float(np.abs(y).max()))
