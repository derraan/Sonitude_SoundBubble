"""Count in-bubble voices whose reference stem file is missing (loader would
silently turn them into empty-GT negatives). Read-only."""
import json, os, sys
thr = 1.5
for root in sys.argv[1:]:
    tot = inb_clips = poisoned = partial = 0
    for d in os.listdir(root):
        p = os.path.join(root, d); mp = os.path.join(p, "metadata.json")
        if not os.path.exists(mp): continue
        m = json.load(open(mp)); real = m.get("real", False); tot += 1
        vs = [k for k in m if "voice" in k and str(m[k].get("speaker_id", "")).lower() != "noise"]
        inb = [v for v in vs if (int(m[v]["dis"]) / 100 if real else m[v]["dis"]) <= thr]
        if not inb: continue
        inb_clips += 1
        have = [v for v in inb if os.path.exists(os.path.join(p, f"mic00_{v}.wav"))]
        if not have: poisoned += 1
        elif len(have) < len(inb): partial += 1
    print(f"{root}: clips={tot} in-bubble clips={inb_clips} "
          f"ALL stems missing (-> fake negative)={poisoned} ({poisoned/max(1,inb_clips):.0%}) "
          f"some missing (-> under-target)={partial}")
