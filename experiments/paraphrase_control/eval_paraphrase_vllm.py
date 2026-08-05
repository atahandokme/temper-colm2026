#!/usr/bin/env python3
"""
Evaluate verbose/paraphrase control with vLLM.
Compares Original (O) vs Paraphrase (P) accuracy.
Supports prompt_style: base (math) and zero_shot_cot (ARC).
"""
import json, sys, argparse, yaml
from pathlib import Path
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).parent))
from reasoning_utils import (
    extract_answer, extract_ground_truth, check_answer,
    SOLVE_PROMPT, SOLVE_PROMPT_MCQ, SOLVE_PROMPT_BASE, SOLVE_PROMPT_MCQ_COT
)

BASE = Path(__file__).parent.parent
DATA_DIR = BASE / "data"

VERBOSE_FILES = {
    'gsm8k': 'emobench_verbose_gsm8k.json',
    'multiarith': 'emobench_verbose_multiarith.json',
    'arc_challenge': 'emobench_verbose_arc_challenge.json',
}

ANSWER_FILES = {
    'gsm8k': 'gsm8k_test.jsonl',
    'multiarith': 'multiarith.jsonl',
    'arc_challenge': 'arc_challenge.jsonl',
}

MCQ_DATASETS = {'arc_challenge', 'gpqa_diamond'}


def is_mcq(ds):
    return ds in MCQ_DATASETS


def load_ground_truth():
    gt = {}
    for ds, fname in ANSWER_FILES.items():
        fpath = DATA_DIR / fname
        if not fpath.exists():
            continue
        with open(fpath) as f:
            for line in f:
                entry = json.loads(line)
                sid = entry.get('id', entry.get('sample_id', ''))
                answer = entry.get('answer', entry.get('ground_truth', ''))
                # Extract just the final answer after #### for math datasets
                answer_str = str(answer).strip()
                if '####' in answer_str:
                    answer_str = answer_str.split('####')[-1].strip()
                gt[sid] = answer_str
    return gt


def main():
    parser = argparse.ArgumentParser(description="Verbose/paraphrase eval with vLLM")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    llm_config = config["llms"][0]
    model_path = llm_config["model_path"]
    model_name = llm_config["name"]
    tp_size = config.get("tensor_parallel_size", 1)
    datasets = config.get("datasets", ["gsm8k", "multiarith", "arc_challenge"])
    prompt_style = config.get("prompt_style", "base")
    output_file = BASE / config["output_file"]

    gen_config = config.get("generation", {})
    temp = gen_config.get("temperature", 0.0)
    max_tokens_math = gen_config.get("max_new_tokens", 2048)
    max_tokens_mcq = gen_config.get("max_new_tokens_mcq", 512)

    print(f"\n[{model_name}] Loading with vLLM (tp={tp_size})...")
    max_model_len = gen_config.get("max_input_length", 4096) + max_tokens_math
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=0.90,
    )
    tokenizer = llm.get_tokenizer()
    print(f"[{model_name}] Loaded", flush=True)

    gt = load_ground_truth()

    sp_math = SamplingParams(temperature=temp, top_p=0.95 if temp > 0 else 1.0, max_tokens=max_tokens_math)
    sp_mcq = SamplingParams(temperature=temp, top_p=0.95 if temp > 0 else 1.0, max_tokens=max_tokens_mcq)

    def make_prompt(question, ds):
        if is_mcq(ds):
            if prompt_style == "base":
                template = SOLVE_PROMPT_MCQ
            else:
                template = SOLVE_PROMPT_MCQ_COT
        else:
            if prompt_style == "base":
                template = SOLVE_PROMPT_BASE
            else:
                template = SOLVE_PROMPT
        user_msg = template.format(question=question)
        messages = [{"role": "user", "content": user_msg}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    all_results = {"model": model_name, "datasets": {}}

    for ds in datasets:
        verbose_file = BASE / "results_v3" / VERBOSE_FILES[ds]
        if not verbose_file.exists():
            print(f"[{model_name}] Skipping {ds} (no verbose file)")
            continue

        samples = json.load(open(verbose_file))
        print(f"\n[{model_name}] {ds}: {len(samples)} samples", flush=True)

        # Build prompts: original + verbose for each sample
        prompts = []
        prompt_map = []  # (sample_idx, condition)

        for si, s in enumerate(samples):
            prompts.append(make_prompt(s['original'], ds))
            prompt_map.append((si, 'original'))

            verbose_text = s.get('verbose', {}).get('verbose_text', '')
            if verbose_text:
                prompts.append(make_prompt(verbose_text, ds))
                prompt_map.append((si, 'verbose'))

        print(f"  Total prompts: {len(prompts)}")
        sp = sp_mcq if is_mcq(ds) else sp_math

        outputs = llm.generate(prompts, sp)

        # Process results
        ds_results = []
        for idx, output in enumerate(outputs):
            si, condition = prompt_map[idx]
            s = samples[si]
            response = output.outputs[0].text.strip()
            predicted = extract_answer(response, ds)
            ground_truth = gt.get(s['id'], s.get('ground_truth', ''))
            correct = check_answer(predicted, ground_truth)

            if len(ds_results) <= si:
                ds_results.append({
                    'id': s['id'],
                    'ground_truth': ground_truth,
                    'conditions': {}
                })
            ds_results[si]['conditions'][condition] = {
                'correct': correct,
                'predicted': str(predicted) if predicted else None,
                'response': response,
            }

        # Compute accuracy
        o_acc = sum(1 for r in ds_results if r['conditions'].get('original', {}).get('correct', False)) / len(ds_results) * 100
        p_acc = sum(1 for r in ds_results if r['conditions'].get('verbose', {}).get('correct', False)) / len(ds_results) * 100

        print(f"  Original: {o_acc:.1f}% | Paraphrase: {p_acc:.1f}% | Diff: {o_acc-p_acc:+.1f}pp", flush=True)
        all_results['datasets'][ds] = ds_results

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump([all_results], f)
    print(f"\n✓ All results saved to: {output_file}", flush=True)


if __name__ == "__main__":
    main()
