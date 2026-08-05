"""
Evaluate Emotion Translations on Math Reasoning Datasets.

Evaluates emotion translator quality across multiple metrics:
    - Emotion accuracy: Does the translated text match the target emotion?
    - Emotion confidence: Classifier confidence on the predicted emotion.
    - Number preservation (MVP): Are all mathematical values preserved?
    - BERTScore: Semantic similarity between original and translated text.
    - Round-trip fidelity: Quality of neutral -> emotional -> neutral cycle.
    - Answer leakage: Does the translation accidentally reveal the answer?

See Section 4 of the paper for evaluation methodology.

Usage:
    python evaluate_translations.py --config ./configs/eval_config.yaml

Dependencies:
    torch, transformers, peft, bert_score, pyyaml, numpy, tqdm
"""

import json
import re
import torch
import yaml
import argparse
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from pathlib import Path
from tqdm import tqdm
from bert_score import score as bert_score


TARGET_EMOTIONS = ["anger", "joy", "sadness", "fear", "disgust", "surprise"]

# Word-form numbers to digit mapping (case-insensitive)
# See reasoning_utils.py for the full MVP checker used in the reasoning tests
WORD_TO_NUM = {
    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
    'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
    'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
    'eighteen': '18', 'nineteen': '19', 'twenty': '20', 'thirty': '30',
    'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
    'eighty': '80', 'ninety': '90', 'hundred': '100', 'thousand': '1000',
    'million': '1000000', 'half': '0.5', 'quarter': '0.25',
    'third': '0.333', 'twice': '2', 'double': '2', 'triple': '3', 'thrice': '3',
    'one-half': '0.5', 'one-third': '0.333', 'one-fourth': '0.25',
    'one-quarter': '0.25', 'three-fourths': '0.75', 'three-quarters': '0.75',
    'two-thirds': '0.667', 'one-fifth': '0.2', 'one-sixth': '0.167',
    'one-eighth': '0.125', 'one-tenth': '0.1',
}

# Math quantifiers that carry mathematical meaning beyond raw numbers
MATH_QUANTIFIERS = [
    'per day', 'per hour', 'per minute', 'per week', 'per month', 'per year',
    'per person', 'per pound', 'per kilogram', 'per mile', 'per km',
    'per gallon', 'per liter',
    'every day', 'every week', 'every month', 'every year', 'every hour',
    'a day', 'an hour', 'a week', 'a month', 'a year',
    'one-half', 'one-third', 'one-fourth', 'one-quarter',
    'two-thirds', 'three-quarters', 'three-fourths', 'one-fifth',
    'each', 'half', 'twice', 'double', 'triple', 'thrice', 'quadruple',
    'quarter', 'third', 'percent', 'percentage',
]

# Answer leakage patterns
ANSWER_LEAKAGE_PATTERNS = [
    r'Answer:', r'####', r'= \d+',
    r'the answer is \d+', r'equals \d+'
]


# ---------------------------------------------------------------------------
# Number Extraction and Preservation (See Section 4.1)
# ---------------------------------------------------------------------------

def normalize_number(s):
    """Convert a number string to a canonical float for comparison.
    Handles fractions (1/2), decimals (0.5), and integers (3).
    """
    s = s.strip()
    if '/' in s:
        parts = s.split('/')
        if len(parts) == 2:
            try:
                return float(parts[0]) / float(parts[1])
            except (ValueError, ZeroDivisionError):
                pass
    try:
        return float(s)
    except ValueError:
        return None


def extract_numbers(text, include_word_forms=True):
    """Extract all numbers from text as normalized floats.

    Args:
        text: Input text to extract numbers from.
        include_word_forms: If True, also match word-form numbers.
            Use False for original text (avoids false positives).
            Use True for generated text (catches paraphrased numbers).
    """
    raw_numbers = set()

    # Extract fractions first (1/2, 3/4) and remove from text
    remaining = text
    for m in re.findall(r'\b(\d+/\d+)\b', text):
        raw_numbers.add(m)
        remaining = remaining.replace(m, ' ', 1)

    # Match digit-form integers and decimals
    for m in re.findall(r'\b\d+\.?\d*\b', remaining):
        raw_numbers.add(m)

    # Match word-form numbers (case-insensitive)
    if include_word_forms:
        text_lower = text.lower()
        for word, digit in WORD_TO_NUM.items():
            if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
                raw_numbers.add(digit)

    # Normalize to floats (round to 3 decimals for fraction equivalence)
    normalized = set()
    for n in raw_numbers:
        val = normalize_number(n)
        if val is not None:
            normalized.add(round(val, 3))
    return normalized


def extract_math_quantifiers(text):
    """Extract math quantifiers present in text (case-insensitive)."""
    text_lower = text.lower()
    found = set()
    for q in MATH_QUANTIFIERS:
        if re.search(r'\b' + re.escape(q) + r'\b', text_lower):
            found.add(q)
    return found


def has_answer_leakage(text):
    """Check if text contains answer leakage patterns."""
    return any(re.search(p, text, re.IGNORECASE) for p in ANSWER_LEAKAGE_PATTERNS)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class TranslationEvaluator:
    """Evaluator for a single emotion translator model on math datasets.

    Performs forward translation (neutral -> emotional), back translation
    (emotional -> neutral), and computes quality metrics for both.
    """

    def __init__(self, model_name, base_model_path, lora_path, device,
                 emotion_classifier, temperature=0.7, gen_config=None):
        self.model_name = model_name
        self.device = device
        self.emotion_classifier = emotion_classifier
        self.temperature = temperature
        self.gen_config = gen_config or {
            "dynamic_tokens": True, "max_new_tokens": 512,
            "top_p": 0.9, "do_sample": True
        }

        # Load tokenizer from base model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with optional LoRA
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype=torch.bfloat16, device_map=device
        )
        if lora_path:
            self.model = PeftModel.from_pretrained(
                self.model, lora_path, adapter_name="translator"
            )
        self.model.eval()

    def generate(self, prompt, source_text=None, temperature=None):
        """Generate text from prompt."""
        temp = temperature if temperature is not None else self.temperature
        max_new_tokens = self.gen_config["max_new_tokens"]

        # Dynamic token limit based on source length
        if self.gen_config.get("dynamic_tokens", True) and source_text is not None:
            source_tokens = len(self.tokenizer.encode(source_text))
            max_new_tokens = max(100, source_tokens * 2)

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temp,
                do_sample=self.gen_config.get("do_sample", True),
                top_p=self.gen_config.get("top_p", 0.9),
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        return generated_text.strip()

    def create_emotion_prompt(self, text, emotion):
        """Create forward translation prompt (neutral -> emotional)."""
        return f"Translate this text from neutral to {emotion}.\n\nInput: {text}\n\nOutput:"

    def create_neutralization_prompt(self, text):
        """Create back translation prompt (emotional -> neutral)."""
        return f"Translate this text to neutral.\n\nInput: {text}\n\nOutput:"

    def evaluate_emotion(self, text, target_emotion):
        """Classify emotion using pre-trained classifier."""
        results = self.emotion_classifier(text, truncation=True, max_length=512)[0]
        if isinstance(results, list):
            top_result = results[0]
        else:
            top_result = results
        predicted = top_result['label'].lower()
        confidence = top_result['score']
        return {
            "predicted": predicted,
            "target": target_emotion.lower(),
            "confidence": confidence,
            "correct": predicted == target_emotion.lower()
        }

    def check_math_preservation(self, original, generated):
        """Check if mathematical content is preserved (See Section 4.1)."""
        orig_numbers = extract_numbers(original, include_word_forms=False)
        gen_numbers = extract_numbers(generated, include_word_forms=True)
        return {
            "numbers_preserved": orig_numbers.issubset(gen_numbers),
            "has_question": "?" in generated,
            "no_answer_leakage": not has_answer_leakage(generated),
            "orig_number_count": len(orig_numbers),
            "gen_number_count": len(gen_numbers)
        }

    def evaluate_sample(self, sample, dataset_name):
        """Evaluate a single sample across all target emotions."""
        question = sample["question"]
        sample_results = {
            "id": sample["id"],
            "dataset": dataset_name,
            "original": question,
            "emotions": {}
        }

        for emotion in TARGET_EMOTIONS:
            # Forward: neutral -> emotional
            forward_prompt = self.create_emotion_prompt(question, emotion)
            emotional_text = self.generate(forward_prompt, source_text=question)

            # Back: emotional -> neutral
            back_prompt = self.create_neutralization_prompt(emotional_text)
            neutral_text = self.generate(back_prompt, source_text=emotional_text)

            # Compute metrics
            emotion_eval = self.evaluate_emotion(emotional_text, emotion)
            math_check = self.check_math_preservation(question, emotional_text)
            rt_math_check = self.check_math_preservation(question, neutral_text)

            sample_results["emotions"][emotion] = {
                "emotional_text": emotional_text,
                "neutral_text": neutral_text,
                "emotion_accuracy": emotion_eval["correct"],
                "emotion_confidence": emotion_eval["confidence"],
                "predicted_emotion": emotion_eval["predicted"],
                "numbers_preserved": math_check["numbers_preserved"],
                "roundtrip_numbers_preserved": rt_math_check["numbers_preserved"],
                "has_question": math_check["has_question"],
                "no_answer_leakage": math_check["no_answer_leakage"],
            }

        return sample_results

    def evaluate_dataset(self, dataset_path, n_samples=500):
        """Evaluate on a dataset with batched BERTScore computation."""
        dataset_name = dataset_path.stem.replace("_test", "")

        # Load dataset
        samples = []
        with open(dataset_path, 'r') as f:
            for line in f:
                samples.append(json.loads(line))

        # Random subsample for evaluation
        import random
        random.seed(42)
        if len(samples) > n_samples:
            samples = random.sample(samples, n_samples)

        # Step 1: Generate all translations
        results = []
        for sample in tqdm(samples, desc=f"Generating [{self.model_name}]"):
            result = self.evaluate_sample(sample, dataset_name)
            results.append(result)

        # Step 2: Batch BERTScore computation (much faster than per-sample)
        forward_candidates, forward_references = [], []
        roundtrip_candidates, roundtrip_references = [], []

        for result in results:
            question = result["original"]
            for emotion in TARGET_EMOTIONS:
                forward_candidates.append(result["emotions"][emotion]["emotional_text"])
                forward_references.append(question)
                roundtrip_candidates.append(result["emotions"][emotion]["neutral_text"])
                roundtrip_references.append(question)

        # Forward BERTScore: original vs emotional
        _, _, forward_f1s = bert_score(
            forward_candidates, forward_references,
            lang="en", device=self.device, batch_size=64
        )

        # Round-trip BERTScore: original vs neutralized
        _, _, roundtrip_f1s = bert_score(
            roundtrip_candidates, roundtrip_references,
            lang="en", device=self.device, batch_size=64
        )

        # Attach scores to results
        idx = 0
        for result in results:
            for emotion in TARGET_EMOTIONS:
                result["emotions"][emotion]["forward_bertscore"] = forward_f1s[idx].item()
                result["emotions"][emotion]["roundtrip_bertscore"] = roundtrip_f1s[idx].item()
                idx += 1

        return results


# ---------------------------------------------------------------------------
# Metrics Aggregation
# ---------------------------------------------------------------------------

def aggregate_metrics(results):
    """Aggregate per-sample metrics into dataset-level averages."""
    metrics = {
        "emotion_accuracy": [], "emotion_confidence": [],
        "numbers_preserved": [], "forward_bertscore": [],
        "roundtrip_bertscore": [], "roundtrip_numbers_preserved": [],
        "has_question": [], "no_answer_leakage": [],
    }

    for sample in results:
        for emotion, data in sample["emotions"].items():
            metrics["emotion_accuracy"].append(1 if data["emotion_accuracy"] else 0)
            metrics["emotion_confidence"].append(data["emotion_confidence"])
            metrics["numbers_preserved"].append(1 if data["numbers_preserved"] else 0)
            metrics["forward_bertscore"].append(data["forward_bertscore"])
            metrics["roundtrip_bertscore"].append(data["roundtrip_bertscore"])
            metrics["roundtrip_numbers_preserved"].append(
                1 if data["roundtrip_numbers_preserved"] else 0)
            metrics["has_question"].append(1 if data["has_question"] else 0)
            metrics["no_answer_leakage"].append(1 if data["no_answer_leakage"] else 0)

    return {key: np.mean(values) if values else 0.0 for key, values in metrics.items()}


def print_results_table(all_model_results):
    """Print formatted results table (See Table 1 in the paper)."""
    print("\n" + "=" * 110)
    print("TRANSLATION EVALUATION RESULTS")
    print("=" * 110)

    header = (
        f"{'Model':<20} {'Dataset':<12} {'Emo Acc':>8} {'Emo Conf':>8} "
        f"{'Num Pres':>8} {'FwdBERT':>8} {'RT BERT':>8} {'RT Num':>8}"
    )
    print(header)
    print("-" * 110)

    for model_result in all_model_results:
        model_name = model_result["model"]
        for dataset_name, results in model_result["datasets"].items():
            metrics = aggregate_metrics(results)
            print(
                f"{model_name:<20} {dataset_name:<12} "
                f"{metrics['emotion_accuracy']:>8.3f} {metrics['emotion_confidence']:>8.3f} "
                f"{metrics['numbers_preserved']:>8.3f} {metrics['forward_bertscore']:>8.3f} "
                f"{metrics['roundtrip_bertscore']:>8.3f} "
                f"{metrics['roundtrip_numbers_preserved']:>8.3f}"
            )
    print("=" * 110)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate emotion translators")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to evaluation config YAML")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_dir = Path('./data/')
    results_dir = Path('./results/')
    results_dir.mkdir(parents=True, exist_ok=True)

    gen_config = config.get("generation", {})
    datasets = [data_dir / ds for ds in config["datasets"]]
    n_samples = config.get("samples_per_dataset", 10)

    all_model_results = []
    for model_cfg in config["models"]:
        device = f"cuda:{model_cfg['gpu_id']}"

        # Load emotion classifier
        emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            device=model_cfg['gpu_id'],
            top_k=None
        )

        evaluator = TranslationEvaluator(
            model_name=model_cfg["name"],
            base_model_path=model_cfg["base_model"],
            lora_path=model_cfg.get("lora_path"),
            device=device,
            emotion_classifier=emotion_classifier,
            temperature=model_cfg.get("temperature", 0.7),
            gen_config=gen_config
        )

        model_results = {"model": model_cfg["name"], "datasets": {}}
        for dataset_path in datasets:
            dataset_name = dataset_path.stem.replace("_test", "")
            dataset_results = evaluator.evaluate_dataset(dataset_path, n_samples=n_samples)
            model_results["datasets"][dataset_name] = dataset_results

        all_model_results.append(model_results)

    # Save results
    output_file = results_dir / config.get("output_file", "evaluation_results.json")
    with open(output_file, 'w') as f:
        json.dump(all_model_results, f, indent=2)

    print_results_table(all_model_results)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
