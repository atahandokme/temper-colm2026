#!/usr/bin/env python3
"""
Build math-faithful subset for one λ value.

Reads:
  - Haiku-verified output (with claude_verdict per record)
  - Optional hand-verified rescues file (sample_id|emotion -> 'NOT BROKEN')
  - Ground truth answers from data/{gsm8k_test, multiarith, arc_challenge}.jsonl

Writes:
  - Flat JSON list of math-faithful records for eval_lambda_reasoning.py

Usage:
  python build_math_faithful_subset.py \
      --haiku   results_v3/lambda_curve/haiku_inputs/emo100_10__haiku_out.json \
      --output  results_v3/lambda_curve/subsets/emo100_10__math_faithful.json \
      [--rescues results_v3/lambda_curve/handverify/emo100_10__rescues.json]
"""
import argparse
import json
from pathlib import Path


def load_ground_truth():
    base = Path('/home/heck2/adokme3/emotion-translator-training/translator_evaluation/data')
    gt = {}
    for fname in ['gsm8k_test.jsonl', 'multiarith.jsonl', 'arc_challenge.jsonl']:
        p = base / fname
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                rec = json.loads(line)
                gt[rec['id']] = rec['answer']
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--haiku', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--rescues', default=None,
                    help='Optional hand-verify rescues: JSON list of {sample_id, emotion} that should be treated as NOT BROKEN despite Haiku verdict')
    ap.add_argument('--include_pending', action='store_true',
                    help='If Haiku run partial, include records with no claude_verdict (treats them as NOT BROKEN — dangerous)')
    args = ap.parse_args()

    haiku = json.load(open(args.haiku))
    print(f"Loaded {len(haiku)} Haiku records from {args.haiku}")

    rescue_set = set()
    if args.rescues and Path(args.rescues).exists():
        rs = json.load(open(args.rescues))
        for r in rs:
            rescue_set.add((r['sample_id'], r['emotion']))
        print(f"Loaded {len(rescue_set)} hand-verify rescues")

    gt = load_ground_truth()
    print(f"Loaded {len(gt)} ground-truth answers")

    kept = []
    n_pending = n_broken = n_rescued = n_missing_gt = 0
    for c in haiku:
        v = c.get('claude_verdict')
        if v is None:
            n_pending += 1
            if not args.include_pending:
                continue
        elif v == 'BROKEN' and (c['sample_id'], c['emotion']) not in rescue_set:
            n_broken += 1
            continue
        elif v == 'BROKEN' and (c['sample_id'], c['emotion']) in rescue_set:
            n_rescued += 1
        if v == 'ERROR':
            continue

        ans = gt.get(c['sample_id'])
        if ans is None:
            n_missing_gt += 1
            continue

        kept.append({
            'id': c['sample_id'],
            'dataset': c['dataset'],
            'emotion': c['emotion'],
            'original': c['original'],
            'emotional_text': c['emotional'],
            'answer': ans,
        })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(kept, open(args.output, 'w'), indent=2)

    print(f"\nMath-faithful subset built:")
    print(f"  Kept:               {len(kept)}")
    print(f"  Broken (filtered):  {n_broken}")
    print(f"  Rescued (hand-vfy): {n_rescued}")
    print(f"  Pending (no Haiku): {n_pending}")
    print(f"  Missing GT:         {n_missing_gt}")
    print(f"  Total input:        {len(haiku)}")
    print(f"  Wrote: {args.output}")

    # Per-emotion breakdown
    by_emo = {}
    for r in kept:
        by_emo.setdefault(r['emotion'], 0)
        by_emo[r['emotion']] += 1
    print("\nKept by emotion:")
    for e in ['anger', 'joy', 'sadness', 'fear', 'disgust', 'surprise']:
        if e in by_emo:
            print(f"  {e:10s}: {by_emo[e]}")


if __name__ == '__main__':
    main()
