#!/usr/bin/env python3
"""Classify gross-but-neutral rewrites with the emotion classifier (the paper's teacher,
j-hartmann). Build reasoning-harness emotionbench files:
  baseline/gross_all/      : all gross rewrites (pseudo-emotion 'gross')
  baseline/gross_neutralcls/: only rewrites the classifier rates NEUTRAL (clean control)
Report what fraction stay neutral (vs the emotional disgust set for reference).
"""
import json, os, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

R = "results_v3"
DATASETS = ["gsm8k", "multiarith", "arc_challenge"]
name = "j-hartmann/emotion-english-distilroberta-base"
tok = AutoTokenizer.from_pretrained(name)
m = AutoModelForSequenceClassification.from_pretrained(name, use_safetensors=True).eval().cuda()
labels = [m.config.id2label[i] for i in range(m.config.num_labels)]

def classify(text):
    enc = tok(text, truncation=True, max_length=512, return_tensors="pt").to("cuda")
    with torch.no_grad():
        p = F.softmax(m(**enc).logits, dim=-1)[0]
    sc = {labels[i]: float(p[i]) for i in range(len(labels))}
    return max(sc, key=sc.get), sc

os.makedirs(f"{R}/baseline/gross_all", exist_ok=True)
os.makedirs(f"{R}/baseline/gross_neutralcls", exist_ok=True)
summary = {}
for ds in DATASETS:
    f = f"{R}/gross_neutral_{ds}.json"
    if not os.path.exists(f):
        print(f"SKIP {ds}: not generated yet"); continue
    d = json.load(open(f))
    recs = d["datasets"][ds] if isinstance(d, dict) and "datasets" in d else (d[0]["datasets"][ds] if "datasets" in d[0] else d)
    all_out, neu_out = [], []
    n_neutral = 0; tops = {}
    for r in recs:
        t = r["emotions"]["gross"]["emotional_text"]
        top, sc = classify(t)
        tops[top] = tops.get(top, 0) + 1
        rec = {"id": r["id"], "original": r["original"],
               "emotions": {"gross": {"emotional_text": t, "neutral_text": r["original"],
                                       "top_emotion": top, "neutral_score": round(sc["neutral"], 3)}}}
        all_out.append(rec)
        if top == "neutral":
            n_neutral += 1; neu_out.append(rec)
    json.dump(all_out, open(f"{R}/baseline/gross_all/emotionbench_{ds}.json", "w"))
    json.dump(neu_out, open(f"{R}/baseline/gross_neutralcls/emotionbench_{ds}.json", "w"))
    summary[ds] = {"n": len(recs), "neutral_classified": n_neutral,
                   "pct_neutral": round(100*n_neutral/len(recs), 1), "top_dist": tops}
    print(f"{ds}: {n_neutral}/{len(recs)} = {100*n_neutral/len(recs):.1f}% classified NEUTRAL | dist={tops}")
json.dump(summary, open(f"{R}/baseline/gross_classification_summary.json", "w"), indent=2)
print("\nWrote baseline/gross_all/ and baseline/gross_neutralcls/ emotionbench files.")
