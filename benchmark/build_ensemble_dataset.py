"""
Build Ensemble Dataset: For each problem × emotion, select the translator
with the highest forward BERTScore across all 5 trained models.

Produces a single high-quality emotional+neutral dataset per benchmark
for use in multi-model reasoning evaluation.

Usage:
  python build_ensemble_dataset.py
  python build_ensemble_dataset.py --dataset gsm8k
  python build_ensemble_dataset.py --dataset multiarith
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent.parent / "results_v3"
OUTPUT_DIR = Path(__file__).parent.parent / "results_v3"

TARGET_EMOTIONS = ["anger", "joy", "sadness", "fear", "disgust", "surprise"]

MODELS = {
    "ce_only": "eval_v3_{dataset}_ce_only.json",
    "emo7_005": "eval_v3_{dataset}_emo7_005.json",
    "emo7_05": "eval_v3_{dataset}_emo7_05.json",
    "emo100_002": "eval_v3_{dataset}_emo100_002.json",
    "emo100_02": "eval_v3_{dataset}_emo100_02.json",
}

DATASETS = {
    "gsm8k": {"key": "gsm8k", "n": 1319},
    "multiarith": {"key": "multiarith", "n": 420},
    "arc_challenge": {"key": "arc_challenge", "n": 1172},
    "gpqa_diamond": {"key": "gpqa_diamond", "n": 198},
}


def load_model_data(dataset_name):
    """Load all 5 models' eval data for a dataset."""
    ds_key = DATASETS[dataset_name]["key"]
    all_data = {}

    for model_name, file_pattern in MODELS.items():
        filename = file_pattern.format(dataset=ds_key)
        filepath = RESULTS_DIR / filename
        if not filepath.exists():
            print(f"  WARNING: {filepath} not found")
            continue

        with open(filepath) as f:
            data = json.load(f)

        samples = data[0]["datasets"][list(data[0]["datasets"].keys())[0]]
        all_data[model_name] = {s["id"]: s for s in samples}
        print(f"  Loaded {model_name}: {len(samples)} samples")

    return all_data


def build_ensemble(dataset_name):
    """Build ensemble dataset by selecting best translator per (problem, emotion)."""
    print(f"\n{'='*60}")
    print(f"Building ensemble: {dataset_name}")
    print(f"{'='*60}")

    all_data = load_model_data(dataset_name)
    if not all_data:
        print("ERROR: No data loaded")
        return

    # Get all problem IDs (from first model)
    first_model = list(all_data.keys())[0]
    problem_ids = list(all_data[first_model].keys())
    print(f"\n  Problems: {len(problem_ids)}")

    ensemble = []
    stats = defaultdict(int)
    missing = 0

    for pid in problem_ids:
        # Get original from any model
        ref_sample = None
        for model_name in all_data:
            if pid in all_data[model_name]:
                ref_sample = all_data[model_name][pid]
                break
        if ref_sample is None:
            missing += 1
            continue

        sample = {
            "id": pid,
            "original": ref_sample["original"],
            "dataset": ref_sample.get("dataset", dataset_name),
            "emotions": {},
        }

        for emotion in TARGET_EMOTIONS:
            # Select best PAIR (emotional + neutral from same model).
            # The neutral is a neutralization of that model's emotional output,
            # so they must stay together.
            # Score = roundtrip BERTScore (best proxy for mathematical integrity).
            best_score = -1
            best_model = None
            best_data = None

            for model_name, model_samples in all_data.items():
                if pid not in model_samples:
                    continue
                edata = model_samples[pid].get("emotions", {}).get(emotion)
                if edata is None:
                    continue

                rt_bs = edata.get("roundtrip_bertscore", 0)

                if rt_bs > best_score:
                    best_score = rt_bs
                    best_model = model_name
                    best_data = edata

            if best_data is not None:
                sample["emotions"][emotion] = {
                    "emotional_text": best_data["emotional_text"],
                    "neutral_text": best_data["neutral_text"],
                    "forward_bertscore": best_data.get("forward_bertscore"),
                    "roundtrip_bertscore": best_data.get("roundtrip_bertscore"),
                    "selection_score": best_score,
                    "source_model": best_model,
                    "emotion_accuracy": best_data.get("emotion_accuracy"),
                    "emotion_confidence": best_data.get("emotion_confidence"),
                    "numbers_preserved": best_data.get("numbers_preserved"),
                    "roundtrip_numbers_preserved": best_data.get("roundtrip_numbers_preserved"),
                    "roundtrip_nli": best_data.get("roundtrip_nli"),
                }
                stats[best_model] += 1

        ensemble.append(sample)

    # Print stats
    print(f"\n  Ensemble built: {len(ensemble)} problems")
    if missing:
        print(f"  Skipped {missing} (missing from all models)")

    print(f"\n  Pair selection (by roundtrip BERTScore):")
    total_sel = sum(stats.values())
    for model_name, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"    {model_name}: {count} ({count/total_sel*100:.1f}%)")

    # Compute ensemble quality stats
    fwd_bs = []
    emo_acc = []
    num_pres = []
    for s in ensemble:
        for e, d in s["emotions"].items():
            fwd_bs.append(d["forward_bertscore"])
            if d.get("emotion_accuracy") is not None:
                emo_acc.append(1 if d["emotion_accuracy"] else 0)
            if d.get("numbers_preserved") is not None:
                num_pres.append(1 if d["numbers_preserved"] else 0)

    print(f"\n  Ensemble quality:")
    print(f"    Mean fwd BERTScore: {sum(fwd_bs)/len(fwd_bs):.4f}")
    if emo_acc:
        print(f"    Emotion accuracy:   {sum(emo_acc)/len(emo_acc)*100:.1f}%")
    if num_pres:
        print(f"    Number preservation: {sum(num_pres)/len(num_pres)*100:.1f}%")

    # Save
    output_path = OUTPUT_DIR / f"ensemble_{dataset_name}.json"
    with open(output_path, "w") as f:
        json.dump(ensemble, f, indent=2)
    print(f"\n  Saved: {output_path}")

    # Also save as JSONL for easy reasoning evaluation
    jsonl_path = OUTPUT_DIR / f"ensemble_{dataset_name}.jsonl"
    with open(jsonl_path, "w") as f:
        for s in ensemble:
            f.write(json.dumps(s) + "\n")
    print(f"  Saved: {jsonl_path}")

    return ensemble


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", choices=["gsm8k", "multiarith", "arc_challenge", "gpqa_diamond", "all"])
    args = parser.parse_args()

    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        build_ensemble(ds)


if __name__ == "__main__":
    main()
