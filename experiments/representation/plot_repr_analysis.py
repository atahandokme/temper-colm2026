#!/usr/bin/env python3
"""Regenerate repr analysis figures with 'Paraphrase' instead of 'Verbose'."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 12

DATA_PATH = '/home/heck2/adokme3/emotion-translator-training/translator_evaluation/results_v3/repr_analysis_4models_verbose.json'
OUT_DIR = '/home/heck2/adokme3/emotion-translator-training/paper_colm/figures'

with open(DATA_PATH) as f:
    data = json.load(f)

models = ['Llama-3.1-8B', 'Qwen-2.5-7B', 'Gemma-2-9B', 'Mistral-7B']
model_short = ['Llama-8B', 'Qwen-7B', 'Gemma-9B', 'Mistral-7B']
emotions = ['anger', 'joy', 'sadness', 'fear', 'disgust', 'surprise']
emotions_cap = ['Anger', 'Joy', 'Sadness', 'Fear', 'Disgust', 'Surprise']
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']  # seaborn-style

# ============================================================
# Figure 1: repr_analysis_cosine.pdf - bar chart cosine distances
# ============================================================
fig, ax = plt.subplots(figsize=(12, 5))

categories = emotions_cap + ['Avg Emo', 'Neutral.', 'Paraphrase']
n_cats = len(categories)
n_models = len(models)
bar_width = 0.18
x = np.arange(n_cats)

for mi, (model, mshort) in enumerate(zip(models, model_short)):
    d = data[model]
    vals = []
    # Per-emotion
    for emo in emotions:
        vals.append(d[f'emotional_{emo}']['mean'])
    # Avg Emo
    avg_emo = np.mean([d[f'emotional_{emo}']['mean'] for emo in emotions])
    vals.append(avg_emo)
    # Neutral (average across emotions)
    avg_neu = np.mean([d[f'neutral_{emo}']['mean'] for emo in emotions])
    vals.append(avg_neu)
    # Paraphrase (was "verbose")
    vals.append(d['verbose']['mean'])

    ax.bar(x + mi * bar_width - (n_models-1)*bar_width/2, vals, bar_width,
           label=mshort, color=colors[mi], alpha=0.85)

# Add separator line
ax.axvline(x=5.5, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=6.5, color='gray', linestyle='--', alpha=0.5)

# Labels
ax.text(2.5, 0.295, 'Per-emotion', ha='center', fontsize=11, color='#CC0000', style='italic')
ax.text(7.5, 0.295, 'Controls', ha='center', fontsize=11, color='#555555', style='italic')

ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=30, ha='right')
ax.set_ylabel('Cosine distance from original')
ax.legend(loc='upper left', fontsize=10)
ax.set_ylim(0, 0.31)

plt.tight_layout()
for ext in ['pdf', 'png']:
    fig.savefig(f'{OUT_DIR}/repr_analysis_cosine.{ext}', dpi=200, bbox_inches='tight')
print("Saved repr_analysis_cosine")
plt.close()

# ============================================================
# Figure 2: repr_analysis_probe.pdf - 4-class probe F1 bar chart
# ============================================================
# Data from appendix table
probe_f1 = {
    'Llama-8B': {'Original': .742, 'Emotional': .998, 'Neutralized': .824, 'Paraphrase': .894},
    'Qwen-7B': {'Original': .715, 'Emotional': .997, 'Neutralized': .805, 'Paraphrase': .874},
    'Gemma-9B': {'Original': .722, 'Emotional': .998, 'Neutralized': .802, 'Paraphrase': .902},
    'Mistral-7B': {'Original': .707, 'Emotional': .998, 'Neutralized': .797, 'Paraphrase': .882},
}
binary_acc = 99.9  # mean

conditions = ['Original', 'Emotional', 'Neutralized', 'Paraphrase']
cond_colors = ['#AAAAAA', '#E74C3C', '#2ECC71', '#3498DB']
markers = ['o', 's', 'D', '^']

fig, ax = plt.subplots(figsize=(7, 6))
x = np.arange(len(conditions))
bar_width = 0.6

means = [np.mean([probe_f1[m][c] for m in probe_f1]) for c in conditions]

bars = ax.bar(x, means, bar_width, color=cond_colors, alpha=0.8, edgecolor='white', linewidth=0.5)

# Add per-model dots
for mi, m in enumerate(probe_f1):
    for ci, c in enumerate(conditions):
        ax.scatter(ci + (mi-1.5)*0.08, probe_f1[m][c],
                  marker=markers[mi], color='#555555', s=40, zorder=5, alpha=0.7)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, means)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
            f'{val:.2f}', ha='center', fontsize=13, fontweight='bold', color=cond_colors[i])

# Binary probe box
ax.text(0.5, 0.98, f'Binary probe (emo vs. rest): {binary_acc}%',
        transform=ax.transAxes, ha='center', va='top', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFE0E0', edgecolor='#FFAAAA'))

# Chance line
ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.text(3.6, 0.26, 'chance', color='gray', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(conditions)
ax.set_ylabel('4-class probe F1')
ax.set_ylim(0, 1.1)

plt.tight_layout()
for ext in ['pdf', 'png']:
    fig.savefig(f'{OUT_DIR}/repr_analysis_probe.{ext}', dpi=200, bbox_inches='tight')
print("Saved repr_analysis_probe")
plt.close()

# ============================================================
# Figure 3: hidden_shift_4models_verbose.pdf - 4-panel plot
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

model_titles = ['Llama-3.1-8B-Instruct', 'Qwen2.5-7B-Instruct',
                'Gemma-2-9B-it', 'Mistral-7B-Instruct-v0.3']

for mi, (model, title) in enumerate(zip(models, model_titles)):
    ax = axes[mi]
    d = data[model]

    x = np.arange(len(emotions) + 1)  # +1 for paraphrase
    labels = emotions_cap + ['Paraphrase']

    # Emotional bars
    emo_vals = [d[f'emotional_{e}']['mean'] for e in emotions]
    emo_errs = [d[f'emotional_{e}']['std'] for e in emotions]

    # Neutral bars
    neu_vals = [d[f'neutral_{e}']['mean'] for e in emotions]
    neu_errs = [d[f'neutral_{e}']['std'] for e in emotions]

    # Paraphrase
    par_val = d['verbose']['mean']
    par_err = d['verbose']['std']

    bar_width = 0.35

    # Emotional (6 emotions only)
    ax.bar(np.arange(6), emo_vals, bar_width, yerr=emo_errs,
           color='#E74C3C', alpha=0.8, label='Emotional', capsize=2)
    # Neutral (6 emotions only)
    ax.bar(np.arange(6) + bar_width, neu_vals, bar_width, yerr=neu_errs,
           color='#2ECC71', alpha=0.8, label='Neutralized', capsize=2)
    # Paraphrase (single bar)
    ax.bar(6.5, par_val, bar_width*2, yerr=par_err,
           color='#3498DB', alpha=0.8, label='Paraphrase', capsize=2)

    ax.set_xticks(list(np.arange(6) + bar_width/2) + [6.5])
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Cosine distance from original', fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold')
    if mi == 0:
        ax.legend(fontsize=9)

plt.tight_layout()
for ext in ['pdf', 'png']:
    fig.savefig(f'{OUT_DIR}/hidden_shift_4models_verbose.{ext}', dpi=200, bbox_inches='tight')
print("Saved hidden_shift_4models_verbose")
plt.close()

print("\nDone! All figures regenerated with 'Paraphrase' instead of 'Verbose'.")
