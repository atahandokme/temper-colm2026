"""
DeepSeek Gold Translation: Have DeepSeek translate GSM8K test set directly.

Uses the same prompts as our 8B translator, so we get "gold standard"
translations with no math errors — isolating the pure effect of emotional
language on reasoning from translation quality issues.

Output format matches eval_v3 format for direct use with reasoning tests.

Usage:
  python 19_deepseek_gold_translate.py \
    --input data/gsm8k_test.jsonl \
    --output results_v3/eval_v3_gsm8k_deepseek_gold.json \
    --max_questions 1319
"""

import json
import asyncio
import argparse
import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

MAX_CONCURRENT_REQUESTS = 15
# GROSS-BUT-NEUTRAL control: unpleasant/rare CONTENT, flat emotionless TONE.
# Tests whether unusual/gross style (not emotion) drives degradation. Must classify NEUTRAL.
TARGET_EMOTIONS = ["gross"]


# === PROMPTS (strict version — prevents DeepSeek from solving) ===

def create_emotion_prompt(text, emotion):
    return f"""Rewrite this math problem so its CONTENT is unpleasant and gross (decay, rot, bodily functions, slime, garbage, vermin, mold) and uses rare, unusual vocabulary, BUT keep the NARRATOR'S TONE completely flat, clinical, and emotionless. Describe the gross things in a detached, matter-of-fact, encyclopedic register. The narrator must NOT express any feeling (no anger, fear, disgust, excitement, sadness, joy).

RULES:
1. Keep ALL numbers exactly the same
2. Keep math words exactly: half, twice, double, triple, ratio, percent, and relationships like "X more than"
3. Keep the question at the end
4. Do NOT solve or show calculations
5. NO emotional words, exclamations, or affect — only neutral, clinical phrasing of unpleasant content
6. Output ONLY the rewritten problem, nothing else

Original: {text}

Gross but emotionally-neutral version:"""


def create_neutralization_prompt(text):
    return f"""Rewrite this math problem in a neutral tone.

RULES:
1. Keep ALL numbers exactly the same
2. Keep math words exactly: half, twice, double, triple, ratio, percent
3. Keep the question at the end
4. Do NOT solve or show calculations
5. Output ONLY the rewritten problem, nothing else

Original: {text}

Neutral version:"""


def extract_numbers(text):
    """Extract all numbers from text."""
    return set(re.findall(r'\d+(?:\.\d+)?', text))


def has_answer_leakage(text, answer_nums=None):
    """Check if text contains '####' pattern (answer leakage)."""
    return '####' in text


async def call_api(client, semaphore, prompt, temperature=0.7, max_tokens=1024, max_retries=3):
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    print(f"\n  API error after {max_retries} retries: {e}")
                    return None


async def translate_one(client, semaphore, original, emotion, temperature=0.7):
    """Forward translate (emotion) + back translate (neutral) one question."""
    # Forward: neutral -> emotional
    fwd_prompt = create_emotion_prompt(original, emotion)
    emotional_text = await call_api(client, semaphore, fwd_prompt, temperature=temperature)
    if not emotional_text:
        return None

    # Back: emotional -> neutral
    back_prompt = create_neutralization_prompt(emotional_text)
    neutral_text = await call_api(client, semaphore, back_prompt, temperature=0.3)
    if not neutral_text:
        return None

    # Quality checks
    orig_nums = extract_numbers(original)
    emo_nums = extract_numbers(emotional_text)
    neu_nums = extract_numbers(neutral_text)

    return {
        'emotional_text': emotional_text,
        'neutral_text': neutral_text,
        'numbers_preserved': orig_nums.issubset(emo_nums),
        'roundtrip_numbers_preserved': orig_nums.issubset(neu_nums),
        'no_answer_leakage': not has_answer_leakage(emotional_text) and not has_answer_leakage(neutral_text),
    }


async def main_async(args):
    # Load API key
    env_path = Path(__file__).parent.parent.parent.parent / "emotion-math-data" / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not found. Check .env file.")
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # Load GSM8K test data
    data_path = Path(__file__).parent.parent / "data" / args.input
    if not data_path.exists():
        data_path = Path(args.input)
    print(f"Loading questions from: {data_path}")

    questions = []
    with open(data_path) as f:
        for line in f:
            questions.append(json.loads(line))
    if args.max_questions > 0:
        questions = questions[:args.max_questions]

    n_total = len(questions) * len(TARGET_EMOTIONS)
    print(f"Questions: {len(questions)}, Emotions: {len(TARGET_EMOTIONS)}")
    print(f"Total translations: {n_total} forward + {n_total} back = {n_total * 2} API calls")
    sys.stdout.flush()

    # Checkpoint support
    output_path = Path(__file__).parent.parent / args.output
    checkpoint_path = output_path.with_suffix('.checkpoint.json')
    completed = {}
    if checkpoint_path.exists():
        completed = json.load(open(checkpoint_path))
        print(f"Resuming from checkpoint: {len(completed)} questions done")

    # Process questions
    results = []
    pbar = tqdm(questions, desc="Translating")

    for q in pbar:
        q_id = q['id']

        if q_id in completed:
            results.append(completed[q_id])
            continue

        original = q['question']
        sample_result = {
            'id': q_id,
            'original': original,
            'emotions': {}
        }

        # Launch all 6 emotions concurrently for this question
        tasks = {}
        for emotion in TARGET_EMOTIONS:
            tasks[emotion] = asyncio.create_task(
                translate_one(client, semaphore, original, emotion, temperature=args.temperature)
            )

        for emotion in TARGET_EMOTIONS:
            result = await tasks[emotion]
            if result is not None:
                sample_result['emotions'][emotion] = result
            else:
                # Fallback: mark as failed
                sample_result['emotions'][emotion] = {
                    'emotional_text': original,
                    'neutral_text': original,
                    'numbers_preserved': True,
                    'roundtrip_numbers_preserved': True,
                    'no_answer_leakage': True,
                    'api_failed': True,
                }

        results.append(sample_result)
        completed[q_id] = sample_result

        # Checkpoint every 50 questions
        if len(results) % 50 == 0:
            json.dump(completed, open(checkpoint_path, 'w'), indent=2)
            pbar.set_postfix(saved=len(results))

        # Progress stats every 100
        if len(results) % 100 == 0:
            n_ok = sum(1 for r in results for e in r['emotions'].values()
                      if e.get('numbers_preserved', False))
            n_total_done = sum(len(r['emotions']) for r in results)
            pct = n_ok / max(n_total_done, 1) * 100
            print(f"\n  {len(results)}/{len(questions)}: {pct:.1f}% numbers preserved")
            sys.stdout.flush()

    # Save in eval format
    dataset_name = args.dataset or Path(args.input).stem.split('_')[0]
    eval_output = [{
        'model': 'deepseek_gold',
        'datasets': {
            dataset_name: results
        }
    }]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(eval_output, f, indent=2)
    print(f"\nSaved {len(results)} samples to {output_path}")

    # Cleanup checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Checkpoint cleaned up.")

    # Summary stats
    total_pairs = sum(len(r['emotions']) for r in results)
    nums_ok = sum(1 for r in results for e in r['emotions'].values()
                 if e.get('numbers_preserved', False))
    rt_ok = sum(1 for r in results for e in r['emotions'].values()
               if e.get('roundtrip_numbers_preserved', False))
    no_leak = sum(1 for r in results for e in r['emotions'].values()
                 if e.get('no_answer_leakage', True))
    api_fails = sum(1 for r in results for e in r['emotions'].values()
                   if e.get('api_failed', False))

    print(f"\n{'='*60}")
    print(f"DEEPSEEK GOLD TRANSLATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Questions: {len(results)}")
    print(f"  Total pairs: {total_pairs}")
    print(f"  Numbers preserved (fwd): {nums_ok}/{total_pairs} ({nums_ok/max(total_pairs,1)*100:.1f}%)")
    print(f"  Numbers preserved (rt):  {rt_ok}/{total_pairs} ({rt_ok/max(total_pairs,1)*100:.1f}%)")
    print(f"  No answer leakage:       {no_leak}/{total_pairs}")
    print(f"  API failures:            {api_fails}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='gsm8k_test.jsonl',
                        help='Input JSONL (GSM8K test questions)')
    parser.add_argument('--output', type=str, default='results_v3/eval_v3_gsm8k_deepseek_gold.json',
                        help='Output JSON in eval format')
    parser.add_argument('--max_questions', type=int, default=1319,
                        help='Max questions to translate (0 = all)')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Forward translation temperature')
    parser.add_argument('--dataset', type=str, default='',
                        help='Dataset name for output format (auto-detected from input filename if empty)')
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
