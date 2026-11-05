"""What fraction of the first-N sorted dirs (what max_samples actually selects)
are empty-GT negatives vs the whole pool? Read-only."""
import json, os, sys
thr = 1.5
def is_neg(p):
    m = json.load(open(os.path.join(p, "metadata.json"))); real = m.get("real", False)
    vs = [k for k in m if "voice" in k and str(m[k].get("speaker_id", "")).lower() != "noise"]
    return not any((int(m[v]["dis"]) / 100 if real else m[v]["dis"]) <= thr for v in vs)
for arg in sys.argv[1:]:
    root, n = arg.rsplit(":", 1); n = int(n)
    dirs = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    neg = [is_neg(os.path.join(root, d)) for d in dirs]
    print(f"{root}  pool={len(dirs)} neg={sum(neg)/len(neg):.0%}  first {n}: neg={sum(neg[:n])/n:.0%}  "
          f"first 5 names={dirs[:5]} ")
