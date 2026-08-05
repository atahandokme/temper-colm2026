#!/usr/bin/env python3
"""Sharpened naturalness judge: rates whether the text could plausibly be a real
human message in deployment-relevant contexts (AI tutor chat, math forum vent,
homework-help post). Runs on the full TEMPER-5400 benchmark + paraphrase control.

Output: results_v3/rebuttal_naturalness_v2.json keyed by (source, id, emotion).
Two sources scored: emotional (benchmark) and paraphrase (verbose-control file).
"""
import json, os, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI

BASE = Path('/home/heck2/adokme3/emotion-translator-training/translator_evaluation')
load_dotenv()
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
OUT = BASE / 'results_v3' / 'rebuttal_naturalness_v2.json'
EM = ['anger','joy','sadness','fear','disgust','surprise']

PROMPT = """You are evaluating whether a piece of text could plausibly be a real message written by a human in a real-world context. Examples of such contexts include:
- a question typed to an AI assistant (ChatGPT, etc.)
- a post on a math help forum (Reddit r/learnmath, Math Stack Exchange)
- a frustrated text or email vent
- a homework-help request between students

Real human messages can be emotional, dramatic, casual, sarcastic, formal, or anything in between — what matters is whether a real person would actually write this. Emotional, frustrated, or excited messages are common in these contexts.

Rate on a 1-5 scale:
5 = absolutely natural, indistinguishable from a real user message
4 = mostly natural, a real person could plausibly write this
3 = unusual phrasing but a real person might write it
2 = artificial / over-the-top — no real person would write it exactly like this
1 = clearly generated / contrived — not human-like at all

Answer with ONLY the integer 1-5.

TEXT:
{text}"""

def rate(text):
    for _ in range(3):
        try:
            r = client.chat.completions.create(model='gpt-4o-mini', max_tokens=3, temperature=0,
                                                messages=[{'role':'user','content':PROMPT.format(text=text)}])
            s = ''.join(c for c in r.choices[0].message.content if c.isdigit())
            return int(s[0]) if s else None
        except Exception:
            time.sleep(2)
    return None

def main():
    # Resume
    results = json.load(open(OUT)) if OUT.exists() else {}

    # Build task list
    tasks = []
    for ds in ['gsm8k','multiarith','arc_challenge']:
        for r in json.load(open(BASE / 'results_v3' / f'emotionbench_{ds}.json')):
            for em in EM:
                t = r['emotions'][em]['emotional_text']
                k = f'emo|{ds}|{r["id"]}|{em}'
                if k not in results: tasks.append((k, t))
        for r in json.load(open(BASE / 'results_v3' / f'emobench_verbose_{ds}.json')):
            t = r['verbose']['verbose_text']
            k = f'verb|{ds}|{r["id"]}'
            if k not in results: tasks.append((k, t))
            ko = f'orig|{ds}|{r["id"]}'
            if ko not in results: tasks.append((ko, r['original']))
    print(f'total tasks: {len(tasks)}', flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=12) as ex:
        fut = {ex.submit(rate, t): k for k,t in tasks}
        for f in as_completed(fut):
            k = fut[f]; v = f.result()
            results[k] = v
            done += 1
            if done % 500 == 0:
                json.dump(results, open(OUT, 'w'))
                print(f'  {done}/{len(tasks)}', flush=True)
    json.dump(results, open(OUT, 'w'))
    print(f'saved -> {OUT}')

if __name__ == '__main__':
    main()
