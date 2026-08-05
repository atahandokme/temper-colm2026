#!/usr/bin/env python3
"""Head-to-head Haiku math-fidelity judge: DeepSeek-direct vs trained-translator
emotional translations on the SAME 400 GSM8K problems x 6 emotions.

Judge = Claude Haiku 4.5 (non-DeepSeek -> no circularity). Forward direction
(emotionalization) is what the 'does it corrupt math' claim is about.
"""
import json, os, time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
R = Path("/home/heck2/adokme3/emotion-translator-training/translator_evaluation/results_v3")
EMOS = ["anger", "joy", "sadness", "fear", "disgust", "surprise"]
OUT = R / "rebuttal_haiku_fidelity_gsm8k.json"

PROMPT = """You are a math problem verifier. Compare the ORIGINAL math problem with the REWRITTEN version.

Determine if the rewritten version preserves the EXACT same mathematical problem. Check:
1. All numbers are the same
2. All mathematical relationships are preserved (e.g., "twice as many", "3 more than")
3. The question being asked is the same
4. No information is added or removed that changes the answer

ORIGINAL:
{original}

REWRITTEN:
{rewritten}

Does the rewritten version preserve the exact same mathematical problem? Answer ONLY "SAME" or "DIFFERENT". If DIFFERENT, briefly explain why."""

def judge(original, rewritten):
    for _ in range(4):
        try:
            r = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=80,
                messages=[{"role": "user", "content": PROMPT.format(original=original, rewritten=rewritten)}])
            t = r.content[0].text.strip()
            return "SAME" if "SAME" in t.upper().split("\n")[0] else "DIFFERENT", t
        except Exception as e:
            time.sleep(3)
    return "ERROR", "max_retries"

def load_emo(path, label):
    """Return {(id, emotion): (original, emotional_text)} for GSM8K 400-set."""
    d = json.load(open(path))
    if isinstance(d, dict) and "datasets" in d: recs = d["datasets"]["gsm8k"]
    elif isinstance(d, list) and d and "datasets" in d[0]: recs = d[0]["datasets"]["gsm8k"]
    else: recs = d  # plain list (emotionbench format)
    out = {}
    for r in recs:
        for e in EMOS:
            out[(str(r["id"]), e)] = (r["original"], r["emotions"][e]["emotional_text"])
    return out

bench_ids = [str(r["id"]) for r in json.load(open(R / "emotionbench_gsm8k.json"))]
SOURCES = {
    "DeepSeek-direct": load_emo(R / "deepseek_direct_gsm8k.json", "ds"),
    "Trained-translator (benchmark)": load_emo(R / "emotionbench_gsm8k.json", "tr"),
    "DeepSeek-terse": load_emo(R / "deepseek_terse_gsm8k.json", "te"),
}

# restrict both to the canonical 400 ids x 6 emo
results = json.load(open(OUT)) if OUT.exists() else {}
lock = threading.Lock()

def run_source(name, mapping):
    results.setdefault(name, {})
    tasks = [(pid, e) for pid in bench_ids for e in EMOS
             if (pid, e) in mapping and f"{pid}|{e}" not in results[name]]
    print(f"{name}: {len(tasks)} to judge", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=12) as ex:
        fut = {ex.submit(judge, *mapping[(pid, e)]): (pid, e) for pid, e in tasks}
        for f in as_completed(fut):
            pid, e = fut[f]
            verdict, raw = f.result()
            with lock:
                results[name][f"{pid}|{e}"] = {"verdict": verdict, "raw": raw}
                done += 1
                if done % 200 == 0:
                    json.dump(results, open(OUT, "w"))
                    print(f"  {name}: {done}/{len(tasks)}", flush=True)
    json.dump(results, open(OUT, "w"))

for name, mp in SOURCES.items():
    run_source(name, mp)

print("\n=== HAIKU MATH FIDELITY (forward emotionalization, GSM8K 400x6) ===")
for name in SOURCES:
    v = list(results[name].values())
    same = sum(1 for x in v if x["verdict"] == "SAME")
    err = sum(1 for x in v if x["verdict"] == "ERROR")
    n = len(v) - err
    print(f"{name:34s}: SAME={100*same/n:.1f}%  (n={n}, errors={err})")
json.dump(results, open(OUT, "w"))
