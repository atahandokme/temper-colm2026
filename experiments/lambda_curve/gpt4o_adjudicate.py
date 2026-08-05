#!/usr/bin/env python3
"""Adjudicate Haiku-DIFFERENT cases to separate REAL math errors from false alarms,
for both DeepSeek-direct and the trained translator, on the SAME 400 GSM8K problems.

Independent adjudicator = GPT-4o-mini (neither Haiku nor DeepSeek). Strict test:
does the rewrite change the correct NUMERICAL ANSWER? -> REAL_ERROR / FALSE_ALARM.
This is the manual-review step the paper does, automated + reproducible.
"""
import json, os, time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
R = Path("/home/heck2/adokme3/emotion-translator-training/translator_evaluation/results_v3")
EMOS = ["anger", "joy", "sadness", "fear", "disgust", "surprise"]
HAIKU = json.load(open(R / "rebuttal_haiku_fidelity_gsm8k.json"))
OUT = R / "rebuttal_adjudication_gsm8k.json"

PROMPT = """You are verifying whether a rewritten math word problem still has the SAME correct numerical answer as the original. Emotional/stylistic language is fine and must be IGNORED. Spelled-out numbers ("half"=0.5, "triple"=3x, "twenty percent"=20%) are EQUIVALENT to digits.

Only call it an ERROR if a NUMBER, QUANTIFIER, RELATIONSHIP, REFERENT, or the QUESTION changed in a way that changes the correct final numeric answer.

ORIGINAL:
{original}

REWRITTEN:
{rewritten}

Would a correct solver get the SAME final numeric answer for both? Answer ONLY one word: FALSE_ALARM (same answer) or REAL_ERROR (answer changes). If REAL_ERROR, add one short clause why."""

def load_emo(path, ds="gsm8k"):
    d = json.load(open(path))
    if isinstance(d, dict) and "datasets" in d: recs = d["datasets"][ds]
    elif isinstance(d, list) and d and "datasets" in d[0]: recs = d[0]["datasets"][ds]
    else: recs = d
    return {(str(r["id"]), e): (r["original"], r["emotions"][e]["emotional_text"]) for r in recs for e in EMOS}

SRC = {
    "DeepSeek-direct": load_emo(R / "deepseek_direct_gsm8k.json"),
    "Trained-translator (benchmark)": load_emo(R / "emotionbench_gsm8k.json"),
    "DeepSeek-terse": load_emo(R / "deepseek_terse_gsm8k.json"),
}

def adj(original, rewritten):
    for _ in range(4):
        try:
            r = client.chat.completions.create(model="gpt-4o", max_tokens=60, temperature=0,
                messages=[{"role": "user", "content": PROMPT.format(original=original, rewritten=rewritten)}])
            t = r.choices[0].message.content.strip()
            return ("REAL_ERROR" if "REAL_ERROR" in t.upper() else "FALSE_ALARM"), t
        except Exception:
            time.sleep(3)
    return "ERROR", "max_retries"

results = json.load(open(OUT)) if OUT.exists() else {}
lock = threading.Lock()
for name, mapping in SRC.items():
    results.setdefault(name, {})
    flagged = [k for k, v in HAIKU.get(name, {}).items() if v["verdict"] == "DIFFERENT"]
    todo = [k for k in flagged if k not in results[name]]
    print(f"{name}: {len(flagged)} Haiku-DIFFERENT, adjudicating {len(todo)}", flush=True)
    def work(k):
        pid, e = k.split("|")
        if (pid, e) not in mapping: return k, ("MISSING", "")
        return k, adj(*mapping[(pid, e)])
    with ThreadPoolExecutor(max_workers=12) as ex:
        for f in as_completed([ex.submit(work, k) for k in todo]):
            k, (verdict, raw) = f.result()
            with lock:
                results[name][k] = {"verdict": verdict, "raw": raw}
    json.dump(results, open(OUT, "w"))

print("\n=== TRUE math-error rate (Haiku-DIFFERENT adjudicated by GPT-4o-mini), GSM8K 400x6 ===")
for name in SRC:
    n_total = sum(1 for v in HAIKU.get(name, {}).values() if v["verdict"] != "ERROR")
    real = sum(1 for v in results[name].values() if v["verdict"] == "REAL_ERROR")
    fa = sum(1 for v in results[name].values() if v["verdict"] == "FALSE_ALARM")
    flagged = real + fa
    print(f"{name:34s}: Haiku-DIFFERENT={flagged} -> REAL={real} ({100*real/n_total:.2f}% of {n_total})  FALSE_ALARM={fa} ({100*fa/max(flagged,1):.0f}% of flagged)")
