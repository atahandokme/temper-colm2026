#!/usr/bin/env python3
"""Generate emotional translations from a trained Llama-3.1-8B LoRA adapter on the
TEMPER-5400 benchmark, then classify each output with the j-hartmann emotion classifier.

Usage:
  python rebuttal_generate_new_variants.py --adapter outputs_extreme/emo100_10/best \\
      --label emo100_10 --gpu 3
Outputs:
  results_v3/lambda_curve/<label>__emotionbench_gen.json
"""
import json, os, argparse, re
from pathlib import Path
BASE = Path('/home/heck2/adokme3/emotion-translator-training/translator_evaluation')
TRAIN_BASE = Path('/home/heck2/adokme3/emotion-translator-training')
EMOTIONS = ["anger", "joy", "sadness", "fear", "disgust", "surprise"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adapter', required=True, help="path to LoRA adapter dir (best or epoch2)")
    ap.add_argument('--label', required=True, help="output label, e.g. emo100_10")
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--base_model', default='meta-llama/Llama-3.1-8B-Instruct')
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['TORCH_COMPILE_DISABLE'] = '1'

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer
    adapter_path = str(TRAIN_BASE / args.adapter) if not os.path.isabs(args.adapter) else args.adapter
    out_dir = BASE / 'results_v3' / 'lambda_curve'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'{args.label}__emotionbench_gen.json'

    # Load all problems (900)
    problems = []
    for ds in ['gsm8k', 'multiarith', 'arc_challenge']:
        with open(BASE / 'results_v3' / f'emotionbench_{ds}.json') as f:
            for r in json.load(f):
                problems.append({'id': r['id'], 'dataset': ds, 'original': r['original']})
    print(f"[{args.label}] loaded {len(problems)} problems")

    tok = AutoTokenizer.from_pretrained(args.base_model)
    llm = LLM(model=args.base_model, dtype='auto', tensor_parallel_size=1,
              enable_lora=True, max_lora_rank=16, max_model_len=2048, enforce_eager=True)
    sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=400)
    lora_req = LoRARequest(args.label, 1, adapter_path)

    # Build forward prompts (translator's exact inference template)
    prompts = []
    keys = []
    for prob in problems:
        for emotion in EMOTIONS:
            raw = f"Translate this text from neutral to {emotion}.\nInput: {prob['original']}\nOutput:"
            p = tok.apply_chat_template([{"role": "user", "content": raw}],
                                        tokenize=False, add_generation_prompt=True)
            prompts.append(p)
            keys.append((prob, emotion))

    print(f"[{args.label}] generating {len(prompts)} translations with LoRA...")
    outs = llm.generate(prompts, sp, lora_request=lora_req)
    results = []
    for (prob, emotion), o in zip(keys, outs):
        text = o.outputs[0].text.strip()
        # clean any prompt-template residue
        for marker in ['Input:', 'Translate this text', 'Output:']:
            if marker in text and text.startswith(marker) is False:
                text = text.split(marker)[0].strip()
        orig_nums = set(re.findall(r'\d+(?:\.\d+)?', prob['original']))
        emo_nums = set(re.findall(r'\d+(?:\.\d+)?', text))
        results.append({
            'id': prob['id'], 'dataset': prob['dataset'], 'emotion': emotion,
            'original': prob['original'], 'emotional_text': text,
            'numbers_preserved': orig_nums.issubset(emo_nums),
        })
    json.dump(results, open(out_file, 'w'))
    print(f"[{args.label}] saved {len(results)} translations -> {out_file}")

    # Free vLLM, then run classifier on the same GPU
    del llm
    import torch; torch.cuda.empty_cache()

    from transformers import AutoModelForSequenceClassification
    import torch.nn.functional as F
    clf_name = 'j-hartmann/emotion-english-distilroberta-base'
    ctok = AutoTokenizer.from_pretrained(clf_name)
    cm = AutoModelForSequenceClassification.from_pretrained(clf_name, use_safetensors=True).eval().cuda()
    labels = [cm.config.id2label[i] for i in range(cm.config.num_labels)]
    bs = 32
    for i in range(0, len(results), bs):
        batch = results[i:i+bs]
        enc = ctok([r['emotional_text'] for r in batch], truncation=True, max_length=512,
                   padding=True, return_tensors='pt').to('cuda')
        with torch.no_grad():
            p = F.softmax(cm(**enc).logits, dim=-1)
        for j, r in enumerate(batch):
            scores = {labels[k]: float(p[j][k]) for k in range(len(labels))}
            top = max(scores, key=scores.get)
            r['predicted_emotion'] = top
            r['emotion_accuracy'] = (top == r['emotion'])
            r['emotion_confidence'] = scores[r['emotion']]
            r['top_emotion_conf'] = scores[top]
    json.dump(results, open(out_file, 'w'))
    print(f"[{args.label}] classification done; saved with emotion fields")

if __name__ == '__main__':
    main()
