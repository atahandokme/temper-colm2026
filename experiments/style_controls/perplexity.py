#!/usr/bin/env python3
"""Compute Llama-3.1-8B perplexity for benchmark texts: original / emotional / verbose /
gross / (archaic/bureaucratic when ready). Used for perplexity-binned drop analysis
(QESb's perplexity-matched-perturbation request).

Saves: results_v3/rebuttal_perplexity.json with {source|ds|id[|emotion]: ppl}.
"""
import json, os, sys, math, argparse
from pathlib import Path
BASE = Path('/home/heck2/adokme3/emotion-translator-training/translator_evaluation')
OUT = BASE / 'results_v3' / 'rebuttal_perplexity.json'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=7)
    ap.add_argument('--max_per_source', type=int, default=2000,
                    help="cap to keep cost manageable; emotional has 5400 by default")
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    mn = 'meta-llama/Llama-3.1-8B-Instruct'
    tok = AutoTokenizer.from_pretrained(mn)
    m = AutoModelForCausalLM.from_pretrained(mn, torch_dtype=torch.bfloat16, device_map='cuda').eval()

    def ppl(text):
        enc = tok(text, return_tensors='pt', truncation=True, max_length=512).to('cuda')
        with torch.no_grad():
            out = m(**enc, labels=enc['input_ids'])
        return math.exp(min(float(out.loss), 20))

    EM = ['anger','joy','sadness','fear','disgust','surprise']
    results = json.load(open(OUT)) if OUT.exists() else {}

    # Build task list
    tasks = []
    # originals (1 per problem) + emotional (6 per problem) + neutral (6 per problem)
    for ds in ['gsm8k','multiarith','arc_challenge']:
        for r in json.load(open(BASE / 'results_v3' / f'emotionbench_{ds}.json')):
            pid = str(r['id'])
            k = f'orig|{ds}|{pid}'
            if k not in results: tasks.append((k, r['original']))
            for em in EM:
                k = f'emo|{ds}|{pid}|{em}'
                if k not in results: tasks.append((k, r['emotions'][em]['emotional_text']))
                k = f'neu|{ds}|{pid}|{em}'
                if k not in results: tasks.append((k, r['emotions'][em]['neutral_text']))
        # verbose
        for r in json.load(open(BASE / 'results_v3' / f'emobench_verbose_{ds}.json')):
            k = f'verb|{ds}|{r["id"]}'
            if k not in results: tasks.append((k, r['verbose']['verbose_text']))
        # gross
        f = BASE / 'results_v3' / f'gross_neutral_{ds}.json'
        if f.exists():
            d = json.load(open(f))
            recs = d['datasets'][ds] if isinstance(d,dict) and 'datasets' in d else (d[0]['datasets'][ds] if 'datasets' in d[0] else d)
            for r in recs:
                k = f'gross|{ds}|{r["id"]}'
                if k not in results: tasks.append((k, r['emotions']['gross']['emotional_text']))
        # archaic + bureaucratic (when ready)
        for tag, fn in [('archaic','archaic_neutral'),('bureaucratic','bureaucratic_neutral')]:
            f = BASE / 'results_v3' / f'{fn}_{ds}.json'
            if not f.exists(): continue
            d = json.load(open(f))
            recs = d['datasets'][ds] if isinstance(d,dict) and 'datasets' in d else (d[0]['datasets'][ds] if 'datasets' in d[0] else d)
            for r in recs:
                t = r['emotions'].get(tag,{}).get('emotional_text','')
                if not t: continue
                k = f'{tag}|{ds}|{r["id"]}'
                if k not in results: tasks.append((k, t))

    print(f'tasks: {len(tasks)}', flush=True)
    done = 0
    for k, text in tasks:
        results[k] = ppl(text)
        done += 1
        if done % 500 == 0:
            json.dump(results, open(OUT, 'w'))
            print(f'  {done}/{len(tasks)}', flush=True)
    json.dump(results, open(OUT, 'w'))
    print(f'done. saved -> {OUT}')

if __name__ == '__main__':
    main()
