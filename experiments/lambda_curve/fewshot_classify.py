#!/usr/bin/env python3
"""Classify few-shot emotionalization outputs through the same emotion classifier
(j-hartmann distilroberta) and compute intensity monotonicity per (model, emotion).
Writes per-(problem,emotion,intensity) classifier scores back into the file, then
prints the controllability table.
"""
import json, os, sys, statistics as st
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

EM = ['anger','joy','sadness','fear','disgust','surprise']
INTENSITIES = [1, 3, 5]
HF = "j-hartmann/emotion-english-distilroberta-base"

def main(infile, outfile=None):
    tok = AutoTokenizer.from_pretrained(HF)
    m = AutoModelForSequenceClassification.from_pretrained(HF, use_safetensors=True).eval().cuda()
    labels = [m.config.id2label[i] for i in range(m.config.num_labels)]
    data = json.load(open(infile))
    # bulk classify in batches
    items = []  # (i, emo, intensity, text)
    for i, p in enumerate(data):
        fs = p.get("fewshot", {})
        for em in EM:
            for inten in INTENSITIES:
                t = fs.get(em, {}).get(str(inten), {}).get("emotional_text", "")
                if t:
                    items.append((i, em, inten, t))
    print(f"{infile}: {len(items)} texts to classify", flush=True)
    BS = 64
    for i in range(0, len(items), BS):
        batch = items[i:i+BS]
        texts = [x[3] for x in batch]
        enc = tok(texts, truncation=True, max_length=512, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            p = F.softmax(m(**enc).logits, dim=-1).cpu().numpy()
        for (pi, em, inten, _), row in zip(batch, p):
            scores = {labels[k]: float(row[k]) for k in range(len(labels))}
            data[pi]["fewshot"][em][str(inten)]["clf_top"] = max(scores, key=scores.get)
            data[pi]["fewshot"][em][str(inten)]["clf_target_score"] = scores.get(em, 0.0)
            data[pi]["fewshot"][em][str(inten)]["clf_scores"] = {k:round(v,3) for k,v in scores.items()}
        if (i // BS) % 50 == 0:
            print(f"  {i+len(batch)}/{len(items)}", flush=True)
    json.dump(data, open(outfile or infile, "w"))
    # === Analysis ===
    print()
    print(f"=== {Path(infile).stem} ===")
    print(f'{"emotion":<10}{"i=1":>10}{"i=3":>10}{"i=5":>10}{"monotonic?":>14}')
    overall_mono = 0; overall_total = 0
    for em in EM:
        means = {}
        for inten in INTENSITIES:
            vals = [p["fewshot"][em][str(inten)]["clf_target_score"]
                    for p in data
                    if str(inten) in p.get("fewshot", {}).get(em, {})
                    and "clf_target_score" in p["fewshot"][em][str(inten)]]
            means[inten] = st.mean(vals) if vals else 0
        mono = means[1] <= means[3] <= means[5]
        overall_mono += int(mono); overall_total += 1
        flag = "YES" if mono else "NO"
        print(f'{em:<10}{means[1]:>10.3f}{means[3]:>10.3f}{means[5]:>10.3f}{flag:>14}')
    print(f"\nmonotonic emotions: {overall_mono}/{overall_total}")
    # Also: % top-emotion = target
    print("\n% texts where classifier top-prediction == target emotion:")
    print(f'{"emotion":<10}{"i=1":>10}{"i=3":>10}{"i=5":>10}')
    for em in EM:
        row = []
        for inten in INTENSITIES:
            xs = [p["fewshot"][em][str(inten)] for p in data
                  if str(inten) in p.get("fewshot", {}).get(em, {})
                  and "clf_top" in p["fewshot"][em][str(inten)]]
            pct = 100 * sum(1 for x in xs if x["clf_top"] == em) / len(xs) if xs else 0
            row.append(pct)
        print(f'{em:<10}{row[0]:>9.1f}%{row[1]:>9.1f}%{row[2]:>9.1f}%')

if __name__ == "__main__":
    main(sys.argv[1])
