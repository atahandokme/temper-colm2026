#!/usr/bin/env python3
"""Convert DeepSeek-direct generation files into reasoning-harness emotionbench
format, one dir per baseline variant. Usage:
  python rebuttal_convert_baseline.py <label> <gsm8k_file> [multiarith_file] [arc_file]
Writes results_v3/baseline/<label>/emotionbench_<ds>.json
"""
import json, os, sys

R = os.path.join(os.path.dirname(__file__), "results_v3")
EMOS = ["anger", "joy", "sadness", "fear", "disgust", "surprise"]

def extract(path, ds):
    d = json.load(open(path))
    if isinstance(d, dict) and "datasets" in d: recs = d["datasets"][ds]
    elif isinstance(d, list) and d and "datasets" in d[0]: recs = d[0]["datasets"][ds]
    else: recs = d
    out = []
    for r in recs:
        if not all(e in r["emotions"] and r["emotions"][e].get("emotional_text") for e in EMOS):
            continue
        emo = {e: {"emotional_text": r["emotions"][e]["emotional_text"],
                   "neutral_text": r["emotions"][e].get("neutral_text", "")} for e in EMOS}
        out.append({"id": r["id"], "original": r["original"], "emotions": emo})
    return out

label = sys.argv[1]
files = {"gsm8k": sys.argv[2]}
if len(sys.argv) > 3 and sys.argv[3] != "-": files["multiarith"] = sys.argv[3]
if len(sys.argv) > 4 and sys.argv[4] != "-": files["arc_challenge"] = sys.argv[4]

outdir = f"{R}/baseline/{label}"
os.makedirs(outdir, exist_ok=True)
for ds, path in files.items():
    recs = extract(path, ds)
    json.dump(recs, open(f"{outdir}/emotionbench_{ds}.json", "w"))
    print(f"{label}/{ds}: {len(recs)} problems -> {outdir}/emotionbench_{ds}.json")
