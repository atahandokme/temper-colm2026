#!/usr/bin/env python3
"""Solver-space t-SNE (Option B): Llama-3.1-8B final-layer mean-pooled states for
all 900 benchmark problems in four conditions (original, emotional x6, neutralized x6,
verbose paraphrase), mirroring the Sec 5.5 pipeline (12,600 states). Output figure
mirrors the teacher-space appendix figure: emotions colored, orig gray, neu blue, verb green.

Usage: python camera_solver_tsne.py --gpu 6
Output: paper/figures/tsne_solver_orig_emo_neu.pdf + .npz of embeddings
"""
import json, os, argparse
from pathlib import Path

BASE = Path('/home/heck2/adokme3/emotion-translator-training/translator_evaluation')
FIGS = Path('/home/heck2/adokme3/emotion-translator-training/paper/figures')
EMOTIONS = ["anger", "joy", "sadness", "fear", "disgust", "surprise"]
EMOTION_COLORS = {'anger': '#e74c3c', 'disgust': '#27ae60', 'fear': '#8e44ad',
                  'joy': '#f1c40f', 'sadness': '#3498db', 'surprise': '#e67e22'}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=6)
    ap.add_argument('--batch', type=int, default=32)
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import numpy as np, torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    texts, labels = [], []   # label: ('orig',None) ('emo',emotion) ('neu',emotion) ('verb',None)
    for ds in ['gsm8k', 'multiarith', 'arc_challenge']:
        bench = json.load(open(BASE / 'results_v3' / f'emotionbench_{ds}.json'))
        for r in bench:
            texts.append(r['original']); labels.append(('orig', None))
            for em in EMOTIONS:
                texts.append(r['emotions'][em]['emotional_text']); labels.append(('emo', em))
                texts.append(r['emotions'][em]['neutral_text']); labels.append(('neu', em))
        verb = json.load(open(BASE / 'results_v3' / f'emobench_verbose_{ds}.json'))
        for r in verb:
            texts.append(r['verbose']['verbose_text']); labels.append(('verb', None))
    print(f'{len(texts)} texts', flush=True)

    mn = 'meta-llama/Llama-3.1-8B-Instruct'
    tok = AutoTokenizer.from_pretrained(mn); tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(mn, torch_dtype=torch.bfloat16,
                                             device_map='cuda', output_hidden_states=True).eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), args.batch):
            b = tok(texts[i:i+args.batch], return_tensors='pt', padding=True,
                    truncation=True, max_length=512).to('cuda')
            h = m(**b).hidden_states[-1]
            mask = b['attention_mask'].unsqueeze(-1)
            embs.append(((h*mask).sum(1)/mask.sum(1)).float().cpu())
            if (i // args.batch) % 30 == 0: print(f'  {i}/{len(texts)}', flush=True)
    X = torch.cat(embs).numpy()
    np.savez(BASE / 'results_v3' / 'solver_tsne_embeddings.npz', X=X,
             kinds=[l[0] for l in labels], emos=[str(l[1]) for l in labels])

    from sklearn.manifold import TSNE
    print('running t-SNE...', flush=True)
    Y = TSNE(n_components=2, perplexity=40, init='pca', random_state=0).fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    kinds = np.array([l[0] for l in labels]); emos = np.array([str(l[1]) for l in labels])
    ax.scatter(*Y[kinds=='orig'].T, s=5, c='#7f8c8d', alpha=0.55, label='Original', linewidths=0)
    ax.scatter(*Y[kinds=='neu'].T, s=5, c='#2c5f8a', alpha=0.35, label='Neutralized', linewidths=0)
    ax.scatter(*Y[kinds=='verb'].T, s=5, c='#16a085', alpha=0.55, label='Paraphrase (verbose)', linewidths=0)
    for em in EMOTIONS:
        pts = Y[(kinds=='emo') & (emos==em)]
        ax.scatter(*pts.T, s=5, c=EMOTION_COLORS[em], alpha=0.5, label=f'Emotional: {em}', linewidths=0)
    ax.legend(markerscale=2.5, fontsize=8, loc='best', framealpha=0.9)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    plt.tight_layout()
    out = FIGS / 'tsne_solver_orig_emo_neu.pdf'
    plt.savefig(out, bbox_inches='tight'); plt.savefig(str(out).replace('.pdf', '.png'), dpi=180, bbox_inches='tight')
    print(f'saved {out}', flush=True)

if __name__ == '__main__':
    main()
