"""
Reasoning Test via API: OpenAI-compatible endpoints (GPT-4o-mini, DeepSeek-V3).

Sends all EmotionBench prompts to an API endpoint with async batching.

Usage:
  python scripts/06c_reasoning_test_api.py --config configs/emotionbench_gpt4omini_api.yaml
"""
import json
import sys
import re
import yaml
import argparse
import asyncio
import time
from pathlib import Path
from dotenv import load_dotenv
import os

sys.path.insert(0, str(Path(__file__).parent))
from reasoning_utils import (
    extract_answer, extract_ground_truth, check_answer,
    SOLVE_PROMPT, SOLVE_PROMPT_MCQ
)


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
    """If text is missing A/B/C/D labels, append original choices."""
    if original_choices is None:
        return text
    if len(re.findall(r'(?:^|\n)\s*[A-D]\.\s', text)) >= 2:
        return text
    return text.strip() + "\n\n" + original_choices


async def call_api(client, model, messages, max_tokens, temperature, semaphore, retry=5, provider="openai"):
    """Call API with retry and rate limiting."""
    async with semaphore:
        for attempt in range(retry):
            try:
                if provider == "gemini":
                    # Use google-genai SDK
                    user_content = messages[0]["content"]
                    response = await asyncio.to_thread(
                        lambda: client.models.generate_content(
                            model=model,
                            contents=user_content,
                            config={"temperature": temperature, "max_output_tokens": max_tokens}
                        )
                    )
                    return response.text or ""
                elif provider == "anthropic":
                    response = await client.messages.create(
                        model=model,
                        max_tokens=max_tokens,
                        messages=messages,
                        temperature=temperature,
                    )
                    return response.content[0].text
                else:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    return response.choices[0].message.content
            except Exception as e:
                if attempt < retry - 1:
                    wait = 2 ** (attempt + 1)  # 2, 4, 8, 16s
                    if "rate_limit" in str(e).lower() or "429" in str(e):
                        wait = max(wait, 10)  # at least 10s for rate limits
                    print(f"    Retry {attempt+1}/{retry} after error: {e}", flush=True)
                    await asyncio.sleep(wait)
                else:
                    print(f"    FAILED after {retry} retries: {e}", flush=True)
                    return ""


async def run_dataset(client, model_id, ds_name, ds_samples, gen_config, max_concurrent, provider="openai", checkpoint_file=None, checkpoint_every=600):
    """Run all prompts for a dataset via async API calls."""
    is_mcq = is_mcq_dataset(ds_name)
    max_tokens = gen_config.get("max_new_tokens_mcq", 128) if is_mcq else gen_config.get("max_new_tokens", 2048)
    temp = gen_config.get("temperature", 0.0)

    print(f"\n  [{ds_name}] Building prompts for {len(ds_samples)} samples...", flush=True)

    # Build all prompts
    all_tasks = []  # (sample_idx, condition_key, messages)
    for si, sample in enumerate(ds_samples):
        mcq_choices = extract_mcq_choices(sample['original']) if is_mcq else None

        # Original
        prompt_template = SOLVE_PROMPT_MCQ if is_mcq else SOLVE_PROMPT
        user_msg = prompt_template.format(question=sample['original'])
        all_tasks.append((si, "original", [{"role": "user", "content": user_msg}]))

        # Emotional + Neutralized
        clean_emos = set(sample.get('clean_emotions',
                       ["anger", "joy", "sadness", "fear", "disgust", "surprise"]))
        for emotion in ["anger", "joy", "sadness", "fear", "disgust", "surprise"]:
            if emotion not in clean_emos:
                continue
            emo_data = sample['emotions'][emotion]
            emo_text = ensure_mcq_choices(emo_data['emotional_text'], mcq_choices)
            neu_text = ensure_mcq_choices(emo_data['neutral_text'], mcq_choices)

            user_emo = prompt_template.format(question=emo_text)
            all_tasks.append((si, f"emo_{emotion}", [{"role": "user", "content": user_emo}]))
            user_neu = prompt_template.format(question=neu_text)
            all_tasks.append((si, f"neu_{emotion}", [{"role": "user", "content": user_neu}]))

    print(f"  [{ds_name}] Total prompts: {len(all_tasks)} — sending with concurrency={max_concurrent}...", flush=True)

    # Initialize results
    sample_results = [{
        "id": s['id'],
        "dataset": s['dataset'],
        "ground_truth": extract_ground_truth(s['answer']),
        "conditions": {}
    } for s in ds_samples]

    # Send all requests with semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)
    completed = 0
    total = len(all_tasks)
    start_time = time.time()

    async def process_task(si, cond_key, messages):
        nonlocal completed
        response = await call_api(client, model_id, messages, max_tokens, temp, semaphore, provider=provider)
        gt = sample_results[si]["ground_truth"]
        pred = extract_answer(response, ds_name)
        sample_results[si]["conditions"][cond_key] = {
            "correct": check_answer(pred, gt),
            "predicted": pred,
            "response": response[:3000]
        }
        completed += 1
        if completed % 100 == 0 or completed == total:
            elapsed = time.time() - start_time
            rate = completed / elapsed
            eta = (total - completed) / rate if rate > 0 else 0
            print(f"    [{ds_name}] {completed}/{total} done ({rate:.1f}/s, ETA: {eta:.0f}s)", flush=True)
        if checkpoint_file and checkpoint_every and completed % checkpoint_every == 0:
            with open(checkpoint_file, 'w') as _f:
                json.dump(sample_results, _f, indent=2)
            print(f"    [{ds_name}] Checkpoint saved at {completed}/{total}", flush=True)

    tasks = [process_task(si, ck, msgs) for si, ck, msgs in all_tasks]
    await asyncio.gather(*tasks)

    # Print dataset summary
    n = len(sample_results)
    acc = {}
    for s in sample_results:
        for cond, data in s['conditions'].items():
            acc.setdefault(cond, []).append(1 if data['correct'] else 0)

    emotions = ["anger", "joy", "sadness", "fear", "disgust", "surprise"]
    orig_acc = sum(acc['original']) / len(acc['original']) * 100
    emo_accs, neu_accs = [], []
    for emotion in emotions:
        ek, nk = f"emo_{emotion}", f"neu_{emotion}"
        ea = sum(acc[ek]) / len(acc[ek]) * 100 if ek in acc and acc[ek] else 0
        na = sum(acc[nk]) / len(acc[nk]) * 100 if nk in acc and acc[nk] else 0
        emo_accs.append(ea)
        neu_accs.append(na)

    avg_emo = sum(emo_accs) / len(emo_accs)
    avg_neu = sum(neu_accs) / len(neu_accs)
    print(f"\n{'─' * 80}")
    print(f"  [{ds_name}] {n} samples DONE")
    print(f"  Original: {orig_acc:.1f}% | Avg Emotional: {avg_emo:.1f}% | Avg Neutralized: {avg_neu:.1f}%")
    print(f"  Drop: {orig_acc - avg_emo:+.1f}pp | Recovery: {avg_neu - avg_emo:+.1f}pp")
    print(f"{'─' * 80}", flush=True)

    return sample_results


async def main_async():
    parser = argparse.ArgumentParser(description="API reasoning test")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load API keys from .env
    env_path = Path(__file__).parent.parent.parent.parent / "emotion-math-data" / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    print("=" * 80)
    print("REASONING TEST (API)")
    print(f"Config: {args.config}")
    print("=" * 80, flush=True)

    base_dir = Path(__file__).parent.parent

    # Load test samples
    data_dir = base_dir / "data"
    test_samples = load_emotionbench_samples(config["emotionbench_files"], data_dir)

    llm_config = config["llms"][0]
    model_name = llm_config["name"]
    model_id = llm_config["model_id"]
    provider = llm_config.get("provider", "openai")

    gen_config = config.get("generation", {})
    max_concurrent = config.get("max_concurrent", 20)

    # Set up API client
    if provider == "gemini":
        from google import genai
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("ERROR: GOOGLE_API_KEY not found")
            sys.exit(1)
        client = genai.Client(api_key=api_key)
    elif provider == "anthropic":
        from anthropic import AsyncAnthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY not found")
            sys.exit(1)
        client = AsyncAnthropic(api_key=api_key)
    else:
        from openai import AsyncOpenAI
        if provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            base_url = "https://api.deepseek.com"
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = None
        else:
            raise ValueError(f"Unknown provider: {provider}")

        if not api_key:
            print(f"ERROR: API key not found for provider '{provider}'")
            sys.exit(1)

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = AsyncOpenAI(**client_kwargs)

    print(f"\nProvider: {provider} | Model: {model_id} | Concurrency: {max_concurrent}")
    sys.stdout.flush()

    # Group by dataset
    by_dataset = {}
    for s in test_samples:
        by_dataset.setdefault(s['dataset'], []).append(s)

    results = {"model": model_name, "samples": []}

    for ds_name, ds_samples in by_dataset.items():
        checkpoint_file = str(base_dir / config["output_file"]).replace('.json', f'_checkpoint_{ds_name}.json')
        sample_results = await run_dataset(client, model_id, ds_name, ds_samples, gen_config, max_concurrent, provider, checkpoint_file=checkpoint_file)
        results["samples"].extend(sample_results)

        # Save intermediate
        output_file = base_dir / config["output_file"]
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump([results], f, indent=2)
        print(f"  [Saved intermediate to {output_file}]", flush=True)

    # Save final
    output_file = base_dir / config["output_file"]
    with open(output_file, 'w') as f:
        json.dump([results], f, indent=2)
    print(f"\n✓ All results saved to: {output_file}")

    from reasoning_utils import print_results
    print_results([results])


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
