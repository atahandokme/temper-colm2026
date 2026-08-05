"""Shared utilities for reasoning test scripts."""
import json
import re
from pathlib import Path


# Word-form numbers to digit mapping (single words)
WORD_TO_NUM = {
    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
    'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
    'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
    'eighteen': '18', 'nineteen': '19', 'twenty': '20', 'thirty': '30',
    'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
    'eighty': '80', 'ninety': '90', 'hundred': '100', 'thousand': '1000',
    'million': '1000000', 'half': '0.5', 'quarter': '0.25',
    'third': '0.333', 'twice': '2', 'double': '2', 'triple': '3',
}

# Multi-word number phrases (checked via substring match on lowered text)
_ONES = ['zero','one','two','three','four','five','six','seven','eight','nine',
         'ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen',
         'seventeen','eighteen','nineteen']
_TENS = ['','','twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety']
MULTIWORD_NUM = {}
for _i in range(20, 100):
    _t, _o = divmod(_i, 10)
    if _o != 0:
        MULTIWORD_NUM[str(_i)] = _TENS[_t] + '-' + _ONES[_o]
for _i in [200, 300, 400, 500, 600, 700, 800, 900]:
    MULTIWORD_NUM[str(_i)] = _ONES[_i // 100] + ' hundred'

# Fraction word mappings
FRAC_WORDS = {
    '1/2': ['half', 'one-half', 'one half'],
    '1/3': ['third', 'one-third', 'one third', 'a third'],
    '2/3': ['two-thirds', 'two thirds'],
    '1/4': ['quarter', 'one-quarter', 'one quarter', 'a quarter'],
    '3/4': ['three-quarters', 'three quarters'],
    '1/5': ['one-fifth', 'one fifth', 'a fifth'],
    '2/5': ['two-fifths', 'two fifths'],
    '3/5': ['three-fifths', 'three fifths'],
    '1/6': ['one-sixth', 'one sixth', 'a sixth'],
    '5/8': ['five-eighths', 'five eighths'],
    '1/8': ['one-eighth', 'one eighth', 'an eighth'],
    '1/10': ['one-tenth', 'one tenth', 'a tenth'],
}

META_COMMENTARY_PATTERNS = [
    r'\bNote\s*:', r'\bI tried\b', r'\bI maintained\b', r'\bI attempted\b',
    r'\bI made sure\b', r'\bI kept\b', r'\bI rewrote\b', r'\bI preserved\b',
]
URL_PATTERNS = [r'https?://', r'www\.', r'\.com\b', r'\.org\b', r'\.net\b']
ARTIFACT_PATTERNS = [r'#[a-zA-Z]\w*\b', r'\bAntworten\b', r'\bFollow\s+us\b', r'@\w+']

SOLVE_PROMPT = "Solve the following math problem step by step. After your solution, write your final numeric answer after ####.\n\nQuestion: {question}"

SOLVE_PROMPT_MCQ = "Answer the following multiple-choice question. Reply with ONLY the letter of the correct answer (A, B, C, or D). Do not explain.\n\nQuestion: {question}\n\nAnswer:"

# Base prompt: no chain-of-thought, just answer
SOLVE_PROMPT_BASE = "Answer the following math problem. Write your final numeric answer after ####.\n\nQuestion: {question}"

SOLVE_PROMPT_MCQ_BASE = "Answer the following multiple-choice question. Reply with ONLY the letter of the correct answer (A, B, C, or D). Do not explain.\n\nQuestion: {question}\n\nAnswer:"

# Zero-shot CoT for MCQ
SOLVE_PROMPT_MCQ_COT = "Answer the following multiple-choice question. Think step by step, then state your final answer as a single letter (A, B, C, or D).\n\nQuestion: {question}\n\nAnswer:"


def extract_numbers_v2(text):
    """Extract all numbers from text, including word-form numbers."""
    numbers = set(re.findall(r'\b\d+\.?\d*\b', text))
    text_lower = text.lower()
    for word, digit in WORD_TO_NUM.items():
        if re.search(r'\b' + word + r'\b', text_lower):
            numbers.add(digit)
    return numbers


def _extract_numbers_v3(text):
    """Extract numbers handling fractions, decimals without leading zero, and commas."""
    # 1. Extract fractions and record their positions to avoid double-counting
    fracs = set()
    frac_positions = set()
    for m in re.finditer(r'\b(\d+/\d+)\b', text):
        fracs.add(m.group(1))
        for i in range(m.start(), m.end()):
            frac_positions.add(i)

    # 2. Decimals without leading zero: .5 → 0.5
    for m in re.finditer(r'(?<!\d)\.(\d+)', text):
        fracs.add('0.' + m.group(1))
        for i in range(m.start(), m.end()):
            frac_positions.add(i)

    # 3. Regular numbers, skipping positions claimed by fractions
    nums = set()
    for m in re.finditer(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', text):
        if not any(i in frac_positions for i in range(m.start(), m.end())):
            nums.add(m.group())

    return fracs | nums


def _normalize_number(n):
    """Strip commas and leading zeros."""
    return n.replace(',', '').lstrip('0') or '0'


def check_numbers_preserved_v2(original, generated):
    """Check number preservation with word-to-digit normalization."""
    orig_numbers = extract_numbers_v2(original)
    gen_numbers = extract_numbers_v2(generated)
    return orig_numbers.issubset(gen_numbers)


def check_numbers_preserved_v3(original, generated):
    """Improved MVP: handles number→word, fractions→words, comma formatting, .5→0.5."""
    orig_nums = _extract_numbers_v3(original)
    gen_nums = _extract_numbers_v3(generated)
    gen_normalized = set(_normalize_number(n) for n in gen_nums)
    gen_lower = generated.lower()

    for num in orig_nums:
        norm = _normalize_number(num)

        # Direct match (with normalization)
        if norm in gen_normalized:
            continue

        # Fraction → word match
        if num in FRAC_WORDS and any(w in gen_lower for w in FRAC_WORDS[num]):
            continue

        # Single-word number match
        if norm in WORD_TO_NUM.values():
            # Reverse lookup: find all words that map to this number
            for word, digit in WORD_TO_NUM.items():
                if digit == norm and re.search(r'\b' + word + r'\b', gen_lower):
                    break
            else:
                pass  # fall through
            if any(digit == norm and re.search(r'\b' + word + r'\b', gen_lower)
                   for word, digit in WORD_TO_NUM.items()):
                continue

        # Multi-word number match (e.g. "forty-eight", "two hundred")
        if norm in MULTIWORD_NUM:
            phrase = MULTIWORD_NUM[norm]
            if phrase in gen_lower or phrase.replace('-', ' ') in gen_lower:
                continue

        # "1" → "a/an/each/single/every" (extremely common paraphrase)
        if norm == '1' and any(w in gen_lower for w in ['one', 'a ', 'an ', 'each ', 'single ', 'every ']):
            continue

        # Large numbers with commas (23786 → 23,786)
        if norm.isdigit() and len(norm) > 3:
            with_commas = '{:,}'.format(int(norm))
            if with_commas in generated:
                continue

        # "X times" → multiplier words
        if norm == '2' and any(w in gen_lower for w in ['twice', 'double']):
            continue
        if norm == '3' and any(w in gen_lower for w in ['triple', 'thrice']):
            continue
        if norm == '4' and 'quadruple' in gen_lower:
            continue

        return False

    return True


def check_text_quality(text):
    """Check text for meta-commentary, URLs, and artifacts."""
    failures = {}
    for pattern in META_COMMENTARY_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            failures['meta_commentary'] = True
            break
    for pattern in URL_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            failures['urls'] = True
            break
    for pattern in ARTIFACT_PATTERNS:
        if re.search(pattern, text):
            failures['artifacts'] = True
            break
    return failures


def extract_answer_mcq(text):
    """Extract multiple-choice answer (A/B/C/D) from LLM response."""
    # Strip chat template artifacts that models insert after "Answer:"
    # e.g. "Answer:\nassistant\nB" or "Answer:assistant\n\nB" or "Answer:\nmodel\nD"
    text = re.sub(r'(?i)(answer\s*:)\s*(assistant|model)\s*', r'\1 ', text)
    # Try #### format first
    hash_matches = re.findall(r'####\s*([A-Da-d])\b', text)
    if hash_matches:
        return hash_matches[-1].upper()
    # Try \boxed{X}
    boxed_matches = re.findall(r'\\boxed\{([A-Da-d])\}', text)
    if boxed_matches:
        return boxed_matches[-1].upper()
    # Try "the answer is X" patterns (letter must not be followed by alpha)
    # Allow optional markdown bold (**) around the letter
    answer_matches = re.findall(
        r'(?:final answer is|the answer is|answer is|answer:)\s*[:.]?\s*\*{0,2}\(?([A-Da-d])\)?',
        text, re.IGNORECASE
    )
    if answer_matches:
        return answer_matches[-1].upper()
    # Try standalone letter at end of response (last non-empty line)
    last_lines = text.strip().split('\n')
    for line in reversed(last_lines):
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^([A-Da-d])[).\s]*$', line)
        if m:
            return m.group(1).upper()
    # Fallback: last occurrence of a standalone A/B/C/D
    all_letters = re.findall(r'\b([A-Da-d])\b', text)
    if all_letters:
        return all_letters[-1].upper()
    return None


def extract_answer(text, dataset_name):
    """Extract answer from LLM response."""
    if dataset_name and ('gpqa' in dataset_name.lower() or 'arc' in dataset_name.lower()):
        return extract_answer_mcq(text)
    # Numeric extraction for math datasets
    boxed_matches = re.findall(r'\\boxed\{(-?[\d,]+\.?\d*)\}', text)
    if boxed_matches:
        return boxed_matches[-1].replace(',', '')
    hash_matches = re.findall(r'####\s*(-?[\d,]+\.?\d*)', text)
    if hash_matches:
        return hash_matches[-1].replace(',', '')
    final_matches = re.findall(
        r'(?:final answer is|final answer:)\s*[:.]?\s*\$?(-?[\d,]+\.?\d*)',
        text, re.IGNORECASE
    )
    if final_matches:
        return final_matches[-1].replace(',', '')
    answer_matches = re.findall(
        r'(?:the answer is|answer is|answer:)\s*[:.]?\s*\$?(-?[\d,]+\.?\d*)',
        text, re.IGNORECASE
    )
    if answer_matches:
        return answer_matches[-1].replace(',', '')
    numbers = re.findall(r'-?\d[\d,]*\.?\d*', text)
    if numbers:
        return numbers[-1].replace(',', '')
    return None


def extract_ground_truth(answer_str):
    """Extract answer from ground truth (numeric or MCQ letter)."""
    answer_str = str(answer_str).strip()
    # Check for single letter answer (MCQ)
    if re.match(r'^[A-Da-d]$', answer_str):
        return answer_str.upper()
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', answer_str)
    if match:
        return match.group(1).replace(',', '')
    numbers = re.findall(r'-?[\d,]+\.?\d*', answer_str)
    if numbers:
        return numbers[-1].replace(',', '')
    return None


def check_answer(predicted, ground_truth):
    """Check if predicted answer matches ground truth."""
    if predicted is None or ground_truth is None:
        return False
    try:
        return abs(float(str(predicted).replace(',', '')) - float(str(ground_truth).replace(',', ''))) < 0.01
    except ValueError:
        return str(predicted).strip().upper() == str(ground_truth).strip().upper()


def load_clean_samples(results_path, max_per_dataset=50, datasets_filter=None,
                       source_model="ce_only", quality_filters=None,
                       skip_number_filters=False):
    """Load model results with per-emotion quality filtering."""
    with open(results_path) as f:
        data = json.load(f)

    ce_only_data = None
    for model_data in data:
        if model_data['model'] == source_model:
            ce_only_data = model_data
            break
    if ce_only_data is None:
        raise ValueError(f"{source_model} not found in results")

    data_dir = Path(__file__).parent.parent / "data"
    ground_truth = {}
    for ds_file in ["gsm8k_test.jsonl", "svamp.jsonl", "multiarith.jsonl", "gpqa_diamond.jsonl"]:
        ds_path = data_dir / ds_file
        if ds_path.exists():
            with open(ds_path) as f:
                for line in f:
                    sample = json.loads(line)
                    ground_truth[sample['id']] = sample['answer']

    qf = quality_filters or {}
    use_quality_filters = len(qf) > 0

    reject_counts = {
        'no_answer': 0, 'numbers_forward': 0, 'numbers_roundtrip': 0,
        'numbers_rescued': 0, 'no_answer_leakage': 0, 'has_question': 0,
        'forward_bertscore': 0, 'roundtrip_bertscore': 0, 'length_ratio': 0,
        'meta_commentary': 0, 'urls': 0, 'artifacts': 0,
    }
    emotion_clean_counts = {}

    test_samples = []
    for ds_name, samples in ce_only_data['datasets'].items():
        if datasets_filter and ds_name not in datasets_filter:
            continue
        ds_count = 0
        ds_total = 0
        for sample in samples:
            if ds_count >= max_per_dataset:
                break
            ds_total += 1

            answer = ground_truth.get(sample['id'], None)
            if answer is None:
                reject_counts['no_answer'] += 1
                continue

            clean_emotions = set()
            for emo, result in sample['emotions'].items():
                emo_ok = True

                if not skip_number_filters:
                    fwd_ok = result.get('numbers_preserved', False)
                    rt_ok = result.get('roundtrip_numbers_preserved', False)

                    if not fwd_ok:
                        fwd_ok = check_numbers_preserved_v2(sample['original'], result.get('emotional_text', ''))
                        if fwd_ok:
                            reject_counts['numbers_rescued'] += 1
                    if not rt_ok:
                        rt_ok = check_numbers_preserved_v2(sample['original'], result.get('neutral_text', ''))
                        if rt_ok:
                            reject_counts['numbers_rescued'] += 1

                    if not fwd_ok:
                        reject_counts['numbers_forward'] += 1
                        emo_ok = False
                    if not rt_ok and emo_ok:
                        reject_counts['numbers_roundtrip'] += 1
                        emo_ok = False

                if emo_ok and use_quality_filters:
                    if qf.get('no_answer_leakage', False) and not result.get('no_answer_leakage', True):
                        reject_counts['no_answer_leakage'] += 1
                        emo_ok = False
                    if emo_ok and qf.get('has_question', False) and not result.get('has_question', True):
                        reject_counts['has_question'] += 1
                        emo_ok = False
                    min_fwd_bert = qf.get('min_forward_bertscore', 0)
                    if emo_ok and min_fwd_bert > 0 and result.get('forward_bertscore', 1.0) < min_fwd_bert:
                        reject_counts['forward_bertscore'] += 1
                        emo_ok = False
                    min_rt_bert = qf.get('min_roundtrip_bertscore', 0)
                    if emo_ok and min_rt_bert > 0 and result.get('roundtrip_bertscore', 1.0) < min_rt_bert:
                        reject_counts['roundtrip_bertscore'] += 1
                        emo_ok = False
                    max_len_ratio = qf.get('max_length_ratio', 0)
                    if emo_ok and max_len_ratio > 0:
                        emo_text = result.get('emotional_text', '')
                        orig_len = max(len(sample['original']), 1)
                        if len(emo_text) / orig_len > max_len_ratio:
                            reject_counts['length_ratio'] += 1
                            emo_ok = False
                        if emo_ok:
                            neu_text = result.get('neutral_text', '')
                            if len(neu_text) / orig_len > max_len_ratio:
                                reject_counts['length_ratio'] += 1
                                emo_ok = False
                    if emo_ok and (qf.get('no_meta_commentary', False) or qf.get('no_urls', False) or qf.get('no_artifacts', False)):
                        for text_field in ['emotional_text', 'neutral_text']:
                            text = result.get(text_field, '')
                            quality_issues = check_text_quality(text)
                            if qf.get('no_meta_commentary', False) and 'meta_commentary' in quality_issues:
                                reject_counts['meta_commentary'] += 1
                                emo_ok = False
                                break
                            if qf.get('no_urls', False) and 'urls' in quality_issues:
                                reject_counts['urls'] += 1
                                emo_ok = False
                                break
                            if qf.get('no_artifacts', False) and 'artifacts' in quality_issues:
                                reject_counts['artifacts'] += 1
                                emo_ok = False
                                break

                if emo_ok:
                    clean_emotions.add(emo)
                    emotion_clean_counts[emo] = emotion_clean_counts.get(emo, 0) + 1

            if len(clean_emotions) > 0:
                test_samples.append({
                    'id': sample['id'],
                    'dataset': ds_name,
                    'original': sample['original'],
                    'answer': answer,
                    'emotions': sample['emotions'],
                    'clean_emotions': list(clean_emotions),
                })
                ds_count += 1

        print(f"  {ds_name}: {ds_count}/{ds_total} samples (at least 1 clean emotion)")

    print(f"\nFiltering summary (per-emotion):")
    print(f"  Total samples with ≥1 clean emotion: {len(test_samples)}")
    print(f"  Clean emotions per type:")
    for emo in ["anger", "joy", "sadness", "fear", "disgust", "surprise"]:
        print(f"    {emo:<10}: {emotion_clean_counts.get(emo, 0)}")
    for reason, count in reject_counts.items():
        if count > 0:
            label = reason.replace('_', ' ').title()
            if reason == 'numbers_rescued':
                print(f"  + {label}: {count} (recovered by word-to-digit normalization)")
            else:
                print(f"  - Rejected emotion-pairs ({label}): {count}")

    return test_samples


def print_results(all_results):
    """Print accuracy comparison table."""
    print("\n" + "=" * 100)
    print("REASONING TEST RESULTS: Does Emotional Translation Preserve Math Reasoning?")
    print("=" * 100)

    emotions = ["anger", "joy", "sadness", "fear", "disgust", "surprise"]

    for result in all_results:
        model = result['model']
        by_dataset = {}
        for sample in result['samples']:
            by_dataset.setdefault(sample['dataset'], []).append(sample)

        print(f"\n{'━' * 100}")
        print(f"  MODEL: {model} ({len(result['samples'])} total samples)")
        print(f"{'━' * 100}")

        all_orig, all_emo, all_neu = [], [], []

        for ds_name, ds_samples in by_dataset.items():
            n = len(ds_samples)
            acc = {}
            for sample in ds_samples:
                for cond, data in sample['conditions'].items():
                    acc.setdefault(cond, []).append(1 if data['correct'] else 0)

            orig_acc = sum(acc['original']) / len(acc['original']) * 100
            all_orig.extend(acc['original'])

            print(f"\n  ┌── {ds_name} ({n} samples) ──")
            print(f"  │ {'Original':<25} {orig_acc:>6.1f}%  (n={len(acc['original'])})")

            for emotion in emotions:
                emo_key = f'emo_{emotion}'
                neu_key = f'neu_{emotion}'
                ea = sum(acc[emo_key]) / len(acc[emo_key]) * 100 if emo_key in acc and acc[emo_key] else 0
                en = len(acc.get(emo_key, []))
                na = sum(acc[neu_key]) / len(acc[neu_key]) * 100 if neu_key in acc and acc[neu_key] else 0
                nn = len(acc.get(neu_key, []))
                if emo_key in acc:
                    all_emo.extend(acc[emo_key])
                if neu_key in acc:
                    all_neu.extend(acc[neu_key])
                print(f"  │ {'Emo ' + emotion:<25} {ea:>6.1f}% (n={en:<4})  {'Neu ' + emotion:<25} {na:>6.1f}% (n={nn})")

            emo_accs = [sum(acc.get(f'emo_{e}', [])) / max(len(acc.get(f'emo_{e}', [])), 1) * 100 for e in emotions]
            neu_accs = [sum(acc.get(f'neu_{e}', [])) / max(len(acc.get(f'neu_{e}', [])), 1) * 100 for e in emotions]
            avg_emo = sum(emo_accs) / len(emo_accs)
            avg_neu = sum(neu_accs) / len(neu_accs)
            print(f"  │ {'── Avg Emotional':<25} {avg_emo:>6.1f}%          {'── Avg Neutralized':<25} {avg_neu:>6.1f}%")
            drop = orig_acc - avg_emo
            recovery = avg_neu - avg_emo
            print(f"  │ Drop: {drop:+.1f}pp | Recovery: {recovery:+.1f}pp")
            print(f"  └{'─' * 80}")

        total_orig = sum(all_orig) / len(all_orig) * 100 if all_orig else 0
        total_emo = sum(all_emo) / len(all_emo) * 100 if all_emo else 0
        total_neu = sum(all_neu) / len(all_neu) * 100 if all_neu else 0
        print(f"\n  OVERALL ({len(result['samples'])} samples): Original={total_orig:.1f}% | Emotional={total_emo:.1f}% | Neutralized={total_neu:.1f}%")
        print(f"  Emotion drop: {total_orig - total_emo:+.1f}pp | Neutralization recovery: {total_neu - total_emo:+.1f}pp")

    print("\n" + "=" * 100)
    import sys
    sys.stdout.flush()
