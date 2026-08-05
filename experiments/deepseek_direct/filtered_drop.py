#!/usr/bin/env python3
"""Recompute DeepSeek baseline O->E drop on GSM8K with broken-math pairs EXCLUDED.
Broken = GPT-4o adjudicated REAL_ERROR (false positives kept in). Isolates the
PURE emotional degradation from math-corruption-driven failures.
"""
import json, glob, os
from collections import defaultdict

R = "results_v3"
EMOS = ["anger", "joy", "sadness", "fear", "disgust", "surprise"]
ADJ = json.load(open(f"{R}/rebuttal_adjudication_gsm8k.json"))

def real_errors(name):
    return {k for k, v in ADJ.get(name, {}).items() if v["verdict"] == "REAL_ERROR"}

def drop(prefix, exclude=None, ds_only="gsm8k"):
    exclude = exclude or set()
    n=o=e=kept_excl=0
    for f in glob.glob(f"{R}/baseline_reasoning/{prefix}__*.json"):
        b = json.load(open(f)); samples = b["samples"] if isinstance(b, dict) else b[0]["samples"]
        for s in samples:
            if s.get("dataset") != ds_only: continue
            c = s["conditions"]; oc = c.get("original", {}).get("correct")
            if oc is None: continue
            for em in EMOS:
                ek = f"emo_{em}"
                if ek not in c: continue
                if f"{s['id']}|{em}" in exclude:
                    kept_excl += 1; continue
                n += 1; o += int(bool(oc)); e += int(bool(c[ek].get("correct")))
    return n, 100*o/n, 100*e/n, 100*(o-e)/n, kept_excl

def num_broken_keys(gen_file):
    d = json.load(open(f"{R}/{gen_file}"))
    recs = d["datasets"]["gsm8k"] if isinstance(d, dict) and "datasets" in d else (d[0]["datasets"]["gsm8k"] if "datasets" in d[0] else d)
    out = set()
    for r in recs:
        for em in EMOS:
            if not r["emotions"][em].get("numbers_preserved"):
                out.add(f"{r['id']}|{em}")
    return out

print("GSM8K O->E drop: UNFILTERED vs broken-math-EXCLUDED (per model-pair pooled)\n")
print(f"{'Source':<22} {'unfilt':>8} {'sem-filt':>9} {'numpres-filt':>13}")
print("-"*54)
for label, prefix, name, genf in [
    ("DeepSeek-direct", "deepseek_direct", "DeepSeek-direct", "deepseek_direct_gsm8k.json"),
    ("DeepSeek-terse", "deepseek_terse", "DeepSeek-terse", "deepseek_terse_gsm8k.json"),
]:
    nu, ou, eu, du, _ = drop(prefix)
    _, _, _, df_sem, _ = drop(prefix, exclude=real_errors(name))           # semantic real-error filter
    _, _, _, df_np, _ = drop(prefix, exclude=num_broken_keys(genf))        # strict number-preservation filter
    print(f"{label:<22} {du:>8.2f} {df_sem:>9.2f} {df_np:>13.2f}")
print("\nsem-filt = exclude GPT-4o REAL_ERROR pairs; numpres-filt = exclude any digit-dropped pair (over-filters word forms)")
print("trained-translator clean per-variant GSM8K drop ~3.0-3.5 (broken ~0.08%, already clean)")
