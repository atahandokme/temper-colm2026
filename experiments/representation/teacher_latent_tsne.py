"""
Quick t-SNE plots for the paper using the teacher's 100-dim bottleneck.
Only needs the small DistilRoBERTa teacher — no LLM loading needed.

Usage: python run_tsne.py --gpu_id 6
"""

import json
import sys
import os
import numpy as np
import torch
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

# Add parent paths for model imports
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'models'))

from emotion_teacher.bottleneck import BottleneckTeacher

EMOTION_COLORS = {
    'anger': '#e74c3c',
    'disgust': '#27ae60',
    'fear': '#8e44ad',
    'joy': '#f1c40f',
    'sadness': '#3498db',
    'surprise': '#e67e22',
}


def load_samples(results_dir, max_per_emotion=200):
    """Load emotional translations from all 5 models."""
    model_files = {
        'CE-Only': 'eval_v3_gsm8k_ce_only.json',
        'Emo7-L': 'eval_v3_gsm8k_emo7_005.json',
        'Emo7-H': 'eval_v3_gsm8k_emo7_05.json',
        'Emo100-L': 'eval_v3_gsm8k_emo100_002.json',
        'Emo100-H': 'eval_v3_gsm8k_emo100_02.json',
    }

    all_data = {}
    for model_name, fname in model_files.items():
        path = os.path.join(results_dir, fname)
        if not os.path.exists(path):
            print(f"  Skipping {model_name}: {path} not found")
            continue

        with open(path) as f:
            data = json.load(f)

        by_emotion = defaultdict(list)
        for sample in data[0]['datasets']['gsm8k']:
            for emotion, result in sample['emotions'].items():
                by_emotion[emotion].append({
                    'text': result['emotional_text'],
                    'neutral': result['neutral_text'],
                    'original': sample['original'],
                    'emotion': emotion,
                    'id': sample['id'],
                })

        # Subsample
        np.random.seed(42)
        subsampled = []
        for emotion, samples in by_emotion.items():
            n = min(max_per_emotion, len(samples))
            indices = np.random.choice(len(samples), n, replace=False)
            for i in indices:
                subsampled.append(samples[i])

        all_data[model_name] = subsampled
        print(f"  {model_name}: {len(subsampled)} samples ({len(subsampled)//6} per emotion)")

    return all_data


def get_teacher_latents(texts, teacher, batch_size=64):
    """Get 100-dim latents from the bottleneck teacher."""
    latents = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        z = teacher.get_latent(batch)  # [B, 100]
        latents.append(z.cpu().numpy())
    return np.concatenate(latents, axis=0)


def get_teacher_probs(texts, teacher, batch_size=64):
    """Get 7-dim emotion probabilities from teacher."""
    probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        p = teacher.get_probs(batch)  # [B, 7]
        probs.append(p.cpu().numpy())
    return np.concatenate(probs, axis=0)


def plot_single_tsne(coords, emotions, title, save_path):
    """Single t-SNE scatter plot."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for emotion in ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']:
        mask = np.array([e == emotion for e in emotions])
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=EMOTION_COLORS[emotion],
            label=emotion.capitalize(),
            alpha=0.5,
            s=12,
            edgecolors='none',
        )

    ax.legend(loc='best', fontsize=10, framealpha=0.9, markerscale=2)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=10)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=10)
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_multi_tsne(model_coords, model_emotions, suptitle, save_path, ncols=3):
    """Multi-panel t-SNE comparison across models."""
    n = len(model_coords)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows))
    if nrows == 1:
        axes = [axes] if ncols == 1 else axes
    axes_flat = np.array(axes).flatten()

    for idx, (model_name, coords) in enumerate(model_coords.items()):
        ax = axes_flat[idx]
        emotions = model_emotions[model_name]

        for emotion in ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']:
            mask = np.array([e == emotion for e in emotions])
            if mask.sum() == 0:
                continue
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=EMOTION_COLORS[emotion],
                label=emotion.capitalize(),
                alpha=0.5,
                s=8,
                edgecolors='none',
            )

        ax.set_title(model_name, fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(True, alpha=0.1)

    # Hide unused axes
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Single legend at bottom
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6,
              fontsize=10, bbox_to_anchor=(0.5, -0.02), markerscale=2)

    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=6)
    parser.add_argument('--max_per_emotion', type=int, default=200)
    args = parser.parse_args()

    results_dir = str(ROOT / 'translator_evaluation' / 'results_v3')
    output_dir = str(ROOT / 'paper' / 'figures')
    os.makedirs(output_dir, exist_ok=True)

    device = f'cuda:{args.gpu_id}'

    # Load teacher
    ckpt = str(ROOT / 'outputs' / 'finetuned_teacher' / 'bottleneck_head.pt')
    if not os.path.exists(ckpt):
        ckpt = str(ROOT / 'outputs' / 'bottleneck_checkpoint' / 'bottleneck_head.pt')
    print(f"Loading teacher with checkpoint: {ckpt}")
    teacher = BottleneckTeacher(latent_dim=100, checkpoint_path=ckpt, device=torch.device(device))

    # Load samples from all 5 models
    print("\nLoading samples...")
    all_data = load_samples(results_dir, max_per_emotion=args.max_per_emotion)

    # =========================================================
    # PLOT 1: Teacher bottleneck space (100-dim) for ONE model
    # =========================================================
    print("\n=== Plot 1: Teacher Bottleneck Space (Emo100-L) ===")
    model_key = 'Emo100-L'
    texts = [s['text'] for s in all_data[model_key]]
    emotions = [s['emotion'] for s in all_data[model_key]]

    latents = get_teacher_latents(texts, teacher)
    print(f"  Latent shape: {latents.shape}")

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    coords = tsne.fit_transform(latents)
    plot_single_tsne(
        coords, emotions,
        'Emotion Bottleneck Space (100-dim)\nEmo100-L Emotional Translations',
        os.path.join(output_dir, 'tsne_bottleneck_emo100l.pdf')
    )

    # =========================================================
    # PLOT 2: Teacher 7-dim probability space for ONE model
    # =========================================================
    print("\n=== Plot 2: Teacher 7-dim Probability Space (Emo100-L) ===")
    probs = get_teacher_probs(texts, teacher)
    print(f"  Probs shape: {probs.shape}")

    tsne7 = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    coords7 = tsne7.fit_transform(probs)
    plot_single_tsne(
        coords7, emotions,
        'Emotion Probability Space (7-dim)\nEmo100-L Emotional Translations',
        os.path.join(output_dir, 'tsne_7dim_emo100l.pdf')
    )

    # =========================================================
    # PLOT 3: Comparison across all 5 models (100-dim bottleneck)
    # =========================================================
    print("\n=== Plot 3: 5-Model Comparison (100-dim Bottleneck) ===")
    model_coords = {}
    model_emotions = {}

    for model_name, samples in all_data.items():
        print(f"  Processing {model_name}...")
        mtexts = [s['text'] for s in samples]
        memos = [s['emotion'] for s in samples]

        mlats = get_teacher_latents(mtexts, teacher)
        mtsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        mcoords = mtsne.fit_transform(mlats)

        model_coords[model_name] = mcoords
        model_emotions[model_name] = memos

    plot_multi_tsne(
        model_coords, model_emotions,
        'Teacher Bottleneck Representations Across 5 Translator Variants',
        os.path.join(output_dir, 'tsne_5model_comparison.pdf'),
        ncols=3
    )

    # =========================================================
    # PLOT 4: Original vs Emotional vs Neutral (Emo100-L only)
    # =========================================================
    print("\n=== Plot 4: Original vs Emotional vs Neutral ===")
    samples = all_data['Emo100-L']
    orig_texts = [s['original'] for s in samples]
    emo_texts = [s['text'] for s in samples]
    neu_texts = [s['neutral'] for s in samples]
    emo_labels = [s['emotion'] for s in samples]

    # Get latents for all three
    all_texts = orig_texts + emo_texts + neu_texts
    all_latents = get_teacher_latents(all_texts, teacher)
    n = len(orig_texts)

    # Labels: "Original", "Emotional", "Neutral"
    type_labels = ['Original']*n + ['Emotional']*n + ['Neutral']*n

    tsne_all = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    coords_all = tsne_all.fit_transform(all_latents)

    fig, ax = plt.subplots(figsize=(8, 6))
    type_colors = {'Original': '#95a5a6', 'Emotional': '#e74c3c', 'Neutral': '#3498db'}
    type_alphas = {'Original': 0.3, 'Emotional': 0.5, 'Neutral': 0.4}

    for ttype in ['Original', 'Emotional', 'Neutral']:
        mask = np.array([t == ttype for t in type_labels])
        ax.scatter(
            coords_all[mask, 0], coords_all[mask, 1],
            c=type_colors[ttype],
            label=('Neutralized' if ttype == 'Neutral' else ttype),
            alpha=type_alphas[ttype],
            s=10,
            edgecolors='none',
        )

    ax.legend(loc='best', fontsize=11, framealpha=0.9, markerscale=2.5)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=10)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=10)
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tsne_orig_emo_neu.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: tsne_orig_emo_neu.pdf")

    print("\n=== All plots done! ===")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
