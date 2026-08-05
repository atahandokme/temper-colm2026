#!/usr/bin/env python3
"""Run sysprompt-vs-no-sysprompt reasoning eval on TEMPER-5400 for an API model
(OpenAI or DeepSeek). Outputs both nosys__<model>.json and sys__<model>.json
in results_v3/sysprompt_reasoning/, in the same schema as the vLLM harness.

Usage:
  python rebuttal_sysprompt_api.py --model gpt-4o-mini --provider openai
  python rebuttal_sysprompt_api.py --model deepseek-chat --provider deepseek
"""
import json, os, sys, time, argparse, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from openai import OpenAI

BASE = Path('/home/heck2/adokme3/emotion-translator-training/translator_evaluation')
sys.path.insert(0, str(BASE / 'scripts'))
from reasoning_utils import (extract_answer, extract_ground_truth, check_answer,
    SOLVE_PROMPT, SOLVE_PROMPT_MCQ_COT)
load_dotenv()

EM = ['anger', 'joy', 'sadness', 'fear', 'disgust', 'surprise']
SYS = "Only focus on the math problem itself. Ignore any emotional, dramatic, or non-functional language, and solve the underlying problem."

# Mirror the vLLM harness's ensure_mcq_choices: ARC's emotional/neutral rewrites
# do not preserve answer-choice letters; we restore them from the original.
import re as _re
def _extract_mcq_choices(original_text):
    pat = r'(?:^|\n)\s*([A-D])\.\s+(.+?)(?=\n\s*[A-D]\.\s|\Z)'
    m = _re.findall(pat, original_text, _re.DOTALL)
    if len(m) >= 2:
        return "\n".join(f"{ltr}. {txt.strip()}" for ltr, txt in m)
    return None
def _ensure_mcq_choices(text, original_choices):
    if original_choices is None: return text
    stripped = _re.split(r'\n\s*[A-D]\.\s', text)[0].strip()
    return stripped + "\n\n" + original_choices

def make_client(provider):
    if provider == 'deepseek':
        return OpenAI(api_key=os.environ['DEEPSEEK_API_KEY'], base_url='https://api.deepseek.com')
    return OpenAI(api_key=os.environ['OPENAI_API_KEY'])

def is_mcq(ds): return 'arc' in ds.lower()

def call_one(client, model, question, dataset, with_sys, max_tokens=2048):
    prompt_tmpl = SOLVE_PROMPT_MCQ_COT if is_mcq(dataset) else SOLVE_PROMPT
    user = prompt_tmpl.format(question=question)
    msgs = []
    if with_sys: msgs.append({"role": "system", "content": SYS})
    msgs.append({"role": "user", "content": user})
    for _ in range(4):
        try:
            r = client.chat.completions.create(model=model, messages=msgs,
                                               max_completion_tokens=max_tokens, temperature=0)
            return r.choices[0].message.content or ""
        except Exception as e:
            print(f"  retry: {str(e)[:120]}", flush=True); time.sleep(3)
    return ""

def run_problem(client, model, sample, gt, dataset, with_sys):
    """Return {condition_key: {correct, predicted, response}} for all 13 conditions."""
    out = {}
    ds_max = 128 if is_mcq(dataset) else 2048
    mcq_choices = _extract_mcq_choices(sample['original']) if is_mcq(dataset) else None
    def do(text):
        if mcq_choices is not None:
            text = _ensure_mcq_choices(text, mcq_choices)
        resp = call_one(client, model, text, dataset, with_sys, max_tokens=ds_max)
        pred = extract_answer(resp, dataset)
        return {"correct": check_answer(pred, gt), "predicted": pred, "response": resp[:3000]}
    out['original'] = do(sample['original'])
    for em in EM:
        emo_obj = sample['emotions'].get(em, {})
        if emo_obj.get('emotional_text'): out[f'emo_{em}'] = do(emo_obj['emotional_text'])
        if emo_obj.get('neutral_text'):   out[f'neu_{em}'] = do(emo_obj['neutral_text'])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--provider', choices=['openai','deepseek'], required=True)
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--out_name', default=None, help="override label in output filename")
    args = ap.parse_args()

    client = make_client(args.provider)
    label = args.out_name or args.model.replace('/', '-')

    # ground truth
    gt = {}
    for f in ['gsm8k_test.jsonl','multiarith.jsonl','arc_challenge.jsonl']:
        for line in open(BASE / 'data' / f):
            j = json.loads(line); gt[str(j['id'])] = extract_ground_truth(j['answer'])

    out_dir = BASE / 'results_v3' / 'sysprompt_reasoning'
    out_dir.mkdir(exist_ok=True)
    for cond, with_sys in [('nosys', False), ('sys', True)]:
        out_path = out_dir / f'{cond}__{label}.json'
        # resume support
        done_ids = set()
        if out_path.exists():
            existing = json.load(open(out_path))
            existing_samples = existing[0]['samples'] if isinstance(existing, list) else existing['samples']
            done_ids = {(s['id'], s['dataset']) for s in existing_samples}
            samples_acc = existing_samples
        else:
            samples_acc = []
        tasks = []
        for ds in ['gsm8k','multiarith','arc_challenge']:
            d = json.load(open(BASE / 'results_v3' / f'emotionbench_{ds}.json'))
            for s in d:
                if (s['id'], ds) in done_ids: continue
                if str(s['id']) not in gt: continue
                tasks.append((s, ds, gt[str(s['id'])]))
        print(f"[{label} | {cond}] todo={len(tasks)} (already done={len(done_ids)})", flush=True)
        def work(args_):
            s, ds, g = args_
            return {'id': s['id'], 'dataset': ds, 'ground_truth': g,
                    'conditions': run_problem(client, args.model, s, g, ds, with_sys)}
        if tasks:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                for i, fut in enumerate(as_completed([ex.submit(work, t) for t in tasks]), 1):
                    samples_acc.append(fut.result())
                    if i % 25 == 0:
                        json.dump([{'model': label, 'samples': samples_acc}], open(out_path, 'w'))
                        print(f"  [{cond}] {i}/{len(tasks)}", flush=True)
            json.dump([{'model': label, 'samples': samples_acc}], open(out_path, 'w'))
        print(f"[{label} | {cond}] saved {len(samples_acc)} samples -> {out_path}", flush=True)

if __name__ == '__main__':
    main()
