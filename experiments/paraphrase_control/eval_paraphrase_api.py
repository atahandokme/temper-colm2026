#!/usr/bin/env python3
"""Evaluate OpenAI models on verbose/paraphrase control (900 samples × 3 datasets)."""
import os
import json, sys, time, argparse
from pathlib import Path
from openai import OpenAI

BASE = Path('/home/heck2/adokme3/emotion-translator-training/translator_evaluation')
sys.path.insert(0, str(BASE / 'scripts'))
from reasoning_utils import extract_answer, extract_ground_truth, check_answer, SOLVE_PROMPT, SOLVE_PROMPT_BASE, SOLVE_PROMPT_MCQ, SOLVE_PROMPT_MCQ_COT

VERBOSE_DATASETS = {
    'gsm8k': {'file': 'emobench_verbose_gsm8k.json', 'answers': 'gsm8k_test.jsonl', 'type': 'math'},
    'multiarith': {'file': 'emobench_verbose_multiarith.json', 'answers': 'multiarith.jsonl', 'type': 'math'},
    'arc_challenge': {'file': 'emobench_verbose_arc_challenge.json', 'answers': 'arc_challenge.jsonl', 'type': 'mc'},
}

PROMPT_STYLE = "base"  # will be set by args

def evaluate_single(client, question, model, ds_type):
    if ds_type == 'mc':
        prompt = (SOLVE_PROMPT_MCQ if PROMPT_STYLE == "base" else SOLVE_PROMPT_MCQ_COT).format(question=question)
    else:
        prompt = (SOLVE_PROMPT_BASE if PROMPT_STYLE == "base" else SOLVE_PROMPT).format(question=question)
    try:
        is_reasoning = any(model.startswith(p) for p in ['o1', 'o3', 'o4']) or model in ['gpt-5', 'gpt-5-2025-08-07']
        kwargs = dict(model=model, messages=[{"role": "user", "content": prompt}], max_completion_tokens=2048)
        if not is_reasoning:
            kwargs['temperature'] = 0
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  Error: {e}", flush=True)
        time.sleep(2)
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--provider', default='openai', choices=['openai', 'deepseek'])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--prompt_style', default='base', choices=['base', 'zero_shot_cot'])
    parser.add_argument('--datasets', default='all', help='Comma-separated: gsm8k,multiarith,arc_challenge or all')
    args = parser.parse_args()

    global PROMPT_STYLE
    PROMPT_STYLE = args.prompt_style

    if args.provider == 'deepseek':
        client = OpenAI(api_key=os.environ['DEEPSEEK_API_KEY'], base_url='https://api.deepseek.com')
    else:
        client = OpenAI()
    model_tag = args.model.replace('.', '').replace('-', '_')
    suffix = '_cot' if args.prompt_style == 'zero_shot_cot' else ''
    output_file = BASE / 'results_v3' / f'emobench_eval_verbose_{model_tag}{suffix}.json'

    # Load answer keys
    answer_keys = {}
    for ds_name, ds in VERBOSE_DATASETS.items():
        answer_keys[ds_name] = {}
        with open(BASE / 'data' / ds['answers']) as f:
            for line in f:
                d = json.loads(line)
                answer_keys[ds_name][d['id']] = d

    # Build all entries from verbose datasets
    ds_filter = None if args.datasets == 'all' else set(args.datasets.split(','))
    entries = []
    for ds_name, ds in VERBOSE_DATASETS.items():
        if ds_filter and ds_name not in ds_filter:
            continue
        with open(BASE / 'results_v3' / ds['file']) as f:
            samples = json.load(f)
        for sample in samples:
            sid = sample['id']
            if sid not in answer_keys[ds_name]:
                continue
            gt = extract_ground_truth(answer_keys[ds_name][sid]['answer'])
            entries.append({
                'sample_id': sid, 'dataset': ds_name, 'ds_type': ds['type'],
                'question': sample['verbose'], 'correct': gt,
            })

    # Resume
    existing = {}
    if args.resume and output_file.exists():
        with open(output_file) as f:
            for r in json.load(f):
                existing[r['sample_id']] = r
        print(f"Resuming: {len(existing)} already done")

    results = list(existing.values())
    todo = [e for e in entries if e['sample_id'] not in existing]
    print(f"Model: {args.model}, Total: {len(entries)}, Remaining: {len(todo)}", flush=True)

    for i, entry in enumerate(todo):
        response = evaluate_single(client, entry['question'], args.model, entry['ds_type'])
        predicted = extract_answer(response, entry['dataset']) if response else None

        result = {
            'sample_id': entry['sample_id'],
            'dataset': entry['dataset'],
            'correct_answer': entry['correct'],
            'predicted': predicted,
            'raw_response': response,
            'is_correct': check_answer(predicted, entry['correct']) if predicted else False,
        }
        results.append(result)

        if (i + 1) % 50 == 0 or i == len(todo) - 1:
            # Stats
            for ds in ['gsm8k', 'multiarith', 'arc_challenge']:
                ds_results = [r for r in results if r['dataset'] == ds]
                if ds_results:
                    acc = sum(r['is_correct'] for r in ds_results) / len(ds_results) * 100
                    print(f"  [{i+1}/{len(todo)}] {ds}: {acc:.1f}% ({len(ds_results)})", flush=True)
            # Checkpoint
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

        time.sleep(0.05)

    # Final
    print(f"\n{'='*60}")
    print(f"FINAL: {args.model} verbose control")
    for ds in ['gsm8k', 'multiarith', 'arc_challenge']:
        ds_results = [r for r in results if r['dataset'] == ds]
        acc = sum(r['is_correct'] for r in ds_results) / len(ds_results) * 100 if ds_results else 0
        print(f"  {ds}: {acc:.1f}% ({len(ds_results)})")
    print(f"Saved: {output_file}", flush=True)

if __name__ == '__main__':
    main()
