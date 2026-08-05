"""
Reasoning Test with vLLM: Fast offline inference with PagedAttention.

Submits all prompts per dataset to vLLM at once for maximum throughput
via continuous batching and tensor parallelism.

Config format same as 06c_reasoning_test_batched.py, with optional extras:
  tensor_parallel_size: 2   (default: 1)
  vllm_quantization: "bitsandbytes"  (default: None)
"""
import json
import sys
import re
import yaml
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reasoning_utils import (
    extract_answer, extract_ground_truth, check_answer,
    load_clean_samples, print_results, SOLVE_PROMPT, SOLVE_PROMPT_MCQ,
    SOLVE_PROMPT_BASE, SOLVE_PROMPT_MCQ_BASE, SOLVE_PROMPT_MCQ_COT
)
from few_shot_examples import get_few_shot_prompt


def load_emotionbench_samples(emotionbench_files, data_dir):
    """Load samples from EmotionBench JSON files + ground truth from data/*.jsonl."""
    ground_truth = {}
    for ds_file in ["gsm8k_test.jsonl", "svamp.jsonl", "multiarith.jsonl",
                     "gpqa_diamond.jsonl", "arc_challenge.jsonl"]:
        ds_path = data_dir / ds_file
        if ds_path.exists():
            with open(ds_path) as f:
                for line in f:
                    sample = json.loads(line)
                    ground_truth[sample['id']] = sample['answer']

    test_samples = []
    for eb_file in emotionbench_files:
        eb_path = Path(eb_file)
        if not eb_path.is_absolute():
            eb_path = data_dir.parent / eb_file
        with open(eb_path) as f:
            problems = json.load(f)

        fname = eb_path.stem
        ds_name = fname.replace("emotionbench_", "")

        no_answer = 0
        for p in problems:
            answer = ground_truth.get(p['id'])
            if answer is None:
                no_answer += 1
                continue
            test_samples.append({
                'id': p['id'],
                'dataset': ds_name,
                'original': p['original'],
                'answer': answer,
                'emotions': p['emotions'],
                'clean_emotions': list(p['emotions'].keys()),
            })

        loaded = len(problems) - no_answer
        print(f"  {ds_name}: {loaded}/{len(problems)} samples loaded"
              f"{f', {no_answer} skipped (no ground truth)' if no_answer else ''}")

    print(f"\n  Total EmotionBench samples: {len(test_samples)}")
    return test_samples


def is_mcq_dataset(ds_name):
    return 'gpqa' in ds_name.lower() or 'arc' in ds_name.lower()


def extract_mcq_choices(original_text):
    """Extract A/B/C/D answer choices from original MCQ text."""
    pattern = r'(?:^|\n)\s*([A-D])\.\s+(.+?)(?=\n\s*[A-D]\.\s|\Z)'
    matches = re.findall(pattern, original_text, re.DOTALL)
    if len(matches) >= 2:
        return "\n".join(f"{letter}. {text.strip()}" for letter, text in matches)
    return None


def ensure_mcq_choices(text, original_choices):
    """Always replace answer choices with original ones for MCQ."""
    if original_choices is None:
        return text
    # Strip any existing A/B/C/D choices from the text
    stripped = re.split(r'\n\s*[A-D]\.\s', text)[0].strip()
    return stripped + "\n\n" + original_choices


def main():
    parser = argparse.ArgumentParser(description="vLLM reasoning test")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=None,
                        help="Override GPU from config (used as first GPU)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("=" * 80)
    print("REASONING TEST (vLLM)")
    print(f"Config: {args.config}")
    print("=" * 80)
    sys.stdout.flush()

    base_dir = Path(__file__).parent.parent

    # Load test samples
    if "emotionbench_files" in config:
        data_dir = base_dir / "data"
        test_samples = load_emotionbench_samples(config["emotionbench_files"], data_dir)
    else:
        results_path = base_dir / config["source_results"]
        samples_config = config.get("samples", {})
        quality_filters = config.get("quality_filters", None)
        skip_number_filters = config.get("skip_number_filters", False)
        test_samples = load_clean_samples(
            results_path,
            max_per_dataset=samples_config.get("max_per_dataset", 500),
            datasets_filter=samples_config.get("datasets"),
            source_model=config.get("source_model", "ce_only"),
            quality_filters=quality_filters,
            skip_number_filters=skip_number_filters
        )

    gen_config = config.get("generation", {})
    llm_config = config["llms"][0]
    model_name = llm_config["name"]
    model_path = llm_config["model_path"]
    quantization = llm_config.get("quantization", None)

    # vLLM-specific config
    tp_size = config.get("tensor_parallel_size", 1)
    vllm_quant = config.get("vllm_quantization", None)

    # Map config quantization to vLLM quantization
    if vllm_quant is None and quantization == "4bit":
        vllm_quant = "bitsandbytes"

    temp = gen_config.get("temperature", 0.0)
    max_new_tokens_math = gen_config.get("max_new_tokens", 2048)
    max_new_tokens_mcq = gen_config.get("max_new_tokens_mcq", 128)
    max_model_len = gen_config.get("max_input_length", 4096) + max_new_tokens_math

    # Load vLLM model
    print(f"\nLoading {model_path} with vLLM (tp={tp_size}, quant={vllm_quant})...")
    sys.stdout.flush()

    from vllm import LLM, SamplingParams

    llm_kwargs = {
        "model": model_path,
        "tensor_parallel_size": tp_size,
        "max_model_len": max_model_len,
        "trust_remote_code": True,
        "dtype": "auto",
        "enforce_eager": True,  # skip torch.compile for PyTorch 2.4 compat
    }
    if vllm_quant:
        llm_kwargs["quantization"] = vllm_quant
    # Optional: hf_overrides to patch broken model configs
    # (e.g., Mistral-7B-Instruct-v0.3 ships with head_dim=None which vLLM can't handle)
    hf_overrides = config.get("hf_overrides")
    if hf_overrides:
        llm_kwargs["hf_overrides"] = hf_overrides

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    print(f"[{model_name}] ✓ Loaded with vLLM")
    sys.stdout.flush()

    # Build sampling params
    sp_math = SamplingParams(
        temperature=temp if temp > 0 else 0,
        top_p=gen_config.get("top_p", 0.95) if temp > 0 else 1.0,
        max_tokens=max_new_tokens_math,
    )
    sp_mcq = SamplingParams(
        temperature=temp if temp > 0 else 0,
        top_p=gen_config.get("top_p", 0.95) if temp > 0 else 1.0,
        max_tokens=max_new_tokens_mcq,
    )

    prompt_style = config.get("prompt_style", "zero_shot_cot")  # default: current behavior
    system_prompt = config.get("system_prompt", None)  # optional: prepend a system message
    # For models whose chat template does NOT accept the system role (e.g. Gemma-2),
    # set system_prompt_as_prefix: true to prepend system_prompt to the user message.
    system_as_prefix = config.get("system_prompt_as_prefix", False)
    def _attach_system(messages, user_idx=-1):
        """Either prepend a system message or fold system text into the user msg."""
        if not system_prompt:
            return messages
        if system_as_prefix:
            messages = list(messages)
            i = user_idx if user_idx >= 0 else len(messages) - 1
            messages[i] = {"role": messages[i]["role"],
                           "content": f"{system_prompt}\n\n{messages[i]['content']}"}
            return messages
        return [{"role": "system", "content": system_prompt}] + messages
    def make_prompt(question, dataset=""):
        if prompt_style == "few_shot_cot":
            # Few-shot CoT: exemplars + question (no chat template, raw prompt)
            few_shot = get_few_shot_prompt(question, dataset)
            if few_shot is not None:
                messages = _attach_system([{"role": "user", "content": few_shot}])
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # Fall back to zero-shot CoT for unsupported datasets
            prompt_template = SOLVE_PROMPT_MCQ if is_mcq_dataset(dataset) else SOLVE_PROMPT
        elif prompt_style == "base":
            prompt_template = SOLVE_PROMPT_MCQ_BASE if is_mcq_dataset(dataset) else SOLVE_PROMPT_BASE
        else:  # zero_shot_cot (default)
            prompt_template = SOLVE_PROMPT_MCQ_COT if is_mcq_dataset(dataset) else SOLVE_PROMPT
        user_msg = prompt_template.format(question=question)
        messages = _attach_system([{"role": "user", "content": user_msg}])
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Group by dataset
    by_dataset = {}
    for s in test_samples:
        by_dataset.setdefault(s['dataset'], []).append(s)

    results = {"model": model_name, "samples": []}

    for ds_name, ds_samples in by_dataset.items():
        is_mcq = is_mcq_dataset(ds_name)
        sampling_params = sp_mcq if is_mcq else sp_math
        ds_max_tokens = max_new_tokens_mcq if is_mcq else max_new_tokens_math

        print(f"\n[{model_name}] Starting {ds_name} ({len(ds_samples)} samples, max_tokens={ds_max_tokens})...")
        sys.stdout.flush()

        # Build ALL prompts for this dataset at once
        all_prompts = []
        prompt_map = []  # (sample_idx, condition_key)

        for si, sample in enumerate(ds_samples):
            mcq_choices = None
            if is_mcq:
                mcq_choices = extract_mcq_choices(sample['original'])

            # Original
            all_prompts.append(make_prompt(sample['original'], sample['dataset']))
            prompt_map.append((si, "original"))

            # Emotional + Neutralized -- iterate over whatever emotion keys are present
            # (standard 6 for the benchmark; custom keys like 'gross' for control sets)
            std = ["anger", "joy", "sadness", "fear", "disgust", "surprise"]
            present = list(sample['emotions'].keys())
            for emotion in ([e for e in std if e in present] or present):
                emo_data = sample['emotions'][emotion]
                emo_text = ensure_mcq_choices(emo_data['emotional_text'], mcq_choices)
                neu_text = ensure_mcq_choices(emo_data['neutral_text'], mcq_choices)
                all_prompts.append(make_prompt(emo_text, sample['dataset']))
                prompt_map.append((si, f"emo_{emotion}"))
                all_prompts.append(make_prompt(neu_text, sample['dataset']))
                prompt_map.append((si, f"neu_{emotion}"))

        print(f"  Total prompts: {len(all_prompts)} ({len(ds_samples)} samples × ~13 conditions)")
        print(f"  Generating with vLLM...")
        sys.stdout.flush()

        # Generate in chunks with checkpoints to avoid losing progress on crash
        sample_results = [{
            "id": s['id'],
            "dataset": s['dataset'],
            "ground_truth": extract_ground_truth(s['answer']),
            "conditions": {}
        } for s in ds_samples]

        chunk_size = config.get("checkpoint_chunk_size", len(all_prompts))  # default: all at once
        for chunk_start in range(0, len(all_prompts), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(all_prompts))
            chunk_prompts = all_prompts[chunk_start:chunk_end]
            chunk_map = prompt_map[chunk_start:chunk_end]

            outputs = llm.generate(chunk_prompts, sampling_params)

            for output, (si, cond_key) in zip(outputs, chunk_map):
                response = output.outputs[0].text
                gt = sample_results[si]["ground_truth"]
                pred = extract_answer(response, ds_name)
                sample_results[si]["conditions"][cond_key] = {
                    "correct": check_answer(pred, gt),
                    "predicted": pred,
                    "response": response[:3000]
                }

            if chunk_size < len(all_prompts):
                # Save checkpoint
                checkpoint_file = base_dir / config["output_file"]
                checkpoint_file = checkpoint_file.with_name(checkpoint_file.stem + f"_checkpoint_{ds_name}.json")
                checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                with open(checkpoint_file, 'w') as f:
                    json.dump(sample_results, f, indent=2)
                print(f"    Checkpoint saved: {chunk_end}/{len(all_prompts)} prompts", flush=True)

        # Print dataset summary
        n = len(sample_results)
        acc = {}
        for s in sample_results:
            for cond, data in s['conditions'].items():
                acc.setdefault(cond, []).append(1 if data['correct'] else 0)

        orig_acc = sum(acc['original']) / len(acc['original']) * 100
        emo_accs, neu_accs = [], []
        for emotion in ["anger", "joy", "sadness", "fear", "disgust", "surprise"]:
            ek, nk = f"emo_{emotion}", f"neu_{emotion}"
            ea = sum(acc[ek]) / len(acc[ek]) * 100 if ek in acc and acc[ek] else 0
            na = sum(acc[nk]) / len(acc[nk]) * 100 if nk in acc and acc[nk] else 0
            emo_accs.append(ea)
            neu_accs.append(na)

        avg_emo = sum(emo_accs) / len(emo_accs)
        avg_neu = sum(neu_accs) / len(neu_accs)
        print(f"\n{'─' * 80}")
        print(f"  [{model_name}] {ds_name} — {n} samples DONE")
        print(f"  Original: {orig_acc:.1f}% | Avg Emotional: {avg_emo:.1f}% | Avg Neutralized: {avg_neu:.1f}%")
        print(f"  Drop: {orig_acc - avg_emo:+.1f}pp | Recovery: {avg_neu - avg_emo:+.1f}pp")
        print(f"{'─' * 80}")
        sys.stdout.flush()

        results["samples"].extend(sample_results)

        # Save intermediate
        output_file = base_dir / config["output_file"]
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump([results], f, indent=2)
        print(f"  [Saved intermediate to {output_file}]")
        sys.stdout.flush()

    # Save final
    output_file = base_dir / config["output_file"]
    with open(output_file, 'w') as f:
        json.dump([results], f, indent=2)
    print(f"\n✓ All results saved to: {output_file}")

    print_results([results])


if __name__ == "__main__":
    main()
