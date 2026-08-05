#!/usr/bin/env python3
"""Build the TEMPER-5400 HuggingFace dataset from the canonical benchmark files.

Core config only: one row per (problem, emotion) pair, 5,400 rows across three
splits (gsm8k 2400, multiarith 1800, arc_challenge 1200).
"""
import json, re
from pathlib import Path

BASE = Path('/home/heck2/adokme3/emotion-translator-training/translator_evaluation')
OUT = Path('/home/heck2/adokme3/emotion-translator-training/release/temper-5400/data')
OUT.mkdir(parents=True, exist_ok=True)
EMOTIONS = ["anger", "joy", "sadness", "fear", "disgust", "surprise"]

# ground truth answers from the canonical reasoning run
gt = {}
d = json.load(open(BASE / 'results_v3' / 'sysprompt_reasoning' / 'nosys__Llama-3.1-8B.json'))
for s in (d[0] if isinstance(d, list) else d)['samples']:
    gt[str(s['id'])] = str(s['ground_truth'])

def parse_choices(text):
    """Extract answer choices from an ARC original. Handles A-D letters
    (3 or 4 options) and numeric 1-4 labels."""
    for pattern, order in [(r'\s*\n?\s*([A-D])[\.\)]\s+', 'ABCD'),
                           (r'\s*\n?\s*([1-4])[\.\)]\s+', '1234')]:
        m = re.split(pattern, text)
        if len(m) < 7:
            continue
        opts = {}
        for i in range(1, len(m) - 1, 2):
            opts[m[i]] = m[i + 1].strip()
        labels = [k for k in order if k in opts]
        if len(labels) >= 3 and labels == list(order[:len(labels)]):
            return [opts[k] for k in labels]
    return None

counts = {}
missing_gt, bad_choices = 0, 0
for ds in ['gsm8k', 'multiarith', 'arc_challenge']:
    bench = json.load(open(BASE / 'results_v3' / f'emotionbench_{ds}.json'))
    rows = []
    for r in bench:
        pid = str(r['id'])
        answer = gt.get(pid)
        if answer is None:
            missing_gt += 1
        choices = []
        if ds == 'arc_challenge':
            choices = parse_choices(r['original'])
            if choices is None:
                bad_choices += 1
                choices = []
        for em in EMOTIONS:
            e = r['emotions'][em]
            rows.append({
                'id': pid,
                'source': ds,
                'original': r['original'],
                'choices': choices,
                'answer': answer,
                'emotion': em,
                'emotional_text': e['emotional_text'],
                'neutralized_text': e['neutral_text'],
                'translator_variant': e.get('translator_name') or e.get('translator'),
                'forward_bertscore': e.get('forward_bertscore'),
                'roundtrip_bertscore': e.get('roundtrip_bertscore'),
            })
    with open(OUT / f'{ds}.jsonl', 'w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    counts[ds] = len(rows)

print('rows per split:', counts, '| total:', sum(counts.values()))
print('missing ground truth:', missing_gt, '| ARC choice parse failures:', bad_choices)
