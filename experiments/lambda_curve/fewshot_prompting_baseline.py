#!/usr/bin/env python3
"""Few-shot emotionalization baseline for MG3d's Q1: 'Is training necessary
vs few-shot prompting LLMs with intensity parameters?'

Generates emotional rewrites of TEMPER-5400 problems using either:
  (a) base Llama-3.1-8B-Instruct via vLLM (within-architecture control)
  (b) DeepSeek-V3 via API
under a few-shot prompt with 2 (neutral->emotional) exemplars per emotion
and an explicit intensity slider (1/5..5/5).

For each (problem, emotion, intensity) it records the rewrite + classifier
emotion confidence + numbers_preserved flag. Output schema matches the
emotionbench files so the same reasoning harness can eval it later.
"""
import json, os, sys, argparse, re, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
BASE = Path(__file__).parent
load_dotenv()

EM = ["anger", "joy", "sadness", "fear", "disgust", "surprise"]
INTENSITIES = [1, 3, 5]
EXEMPLARS = json.load(open(BASE / "rebuttal_fewshot_exemplars.json"))

def build_prompt(text, emotion, intensity):
    ex = EXEMPLARS[emotion]
    sys_inst = (
        f"You rewrite math problems in a {emotion} tone at a target intensity level on a 1-5 scale "
        f"(1 = subtle/barely noticeable, 3 = clearly emotional, 5 = very intense). "
        f"You MUST keep every number, fraction, percent, and quantitative relationship "
        f"(e.g., 'half', 'twice', 'X more than') exactly as in the original. "
        f"Keep the question at the end. Output ONLY the rewritten problem, nothing else."
    )
    shots = ""
    for i, (neu, emo) in enumerate(ex[:2], 1):
        shots += f"\n\nExample {i} - Neutral:\n{neu}\nExample {i} - {emotion} (intensity {intensity}/5):\n{emo}"
    user = (
        f"{sys_inst}\n"
        f"{shots}\n\n"
        f"Now rewrite the following neutral problem in a {emotion} tone at intensity {intensity}/5.\n\n"
        f"Neutral:\n{text}\n{emotion} (intensity {intensity}/5):"
    )
    return user

def nums(s): return set(re.findall(r'\d+(?:\.\d+)?', s))

def clean_output(out):
    # Strip common prefixes the model adds before the rewrite
    out = out.strip()
    # remove anything before "Neutral:" or starting with "Example"
    for pat in ["Example ", "Neutral:"]:
        idx = out.find(pat)
        if idx > 0: out = out[:idx]
    # remove trailing "Neutral:" lines
    return out.strip()

def run_local_llama(problems, out_path, gpu=7, model_path="meta-llama/Llama-3.1-8B-Instruct"):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path, dtype="auto", tensor_parallel_size=1, enforce_eager=True,
              max_model_len=2048)
    sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=400)

    # build all prompts
    prompts = []
    keys = []  # (prob_idx, emotion, intensity)
    for pi, p in enumerate(problems):
        for emotion in EM:
            for intensity in INTENSITIES:
                user = build_prompt(p["original"], emotion, intensity)
                msgs = [{"role": "user", "content": user}]
                prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
                keys.append((pi, emotion, intensity))
    print(f"generated {len(prompts)} prompts; running vLLM...", flush=True)
    outs = llm.generate(prompts, sp)
    for (pi, emotion, intensity), o in zip(keys, outs):
        text = clean_output(o.outputs[0].text)
        problems[pi].setdefault("fewshot", {}).setdefault(emotion, {})[str(intensity)] = {
            "emotional_text": text,
            "numbers_preserved": nums(problems[pi]["original"]).issubset(nums(text)),
        }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(problems, open(out_path, "w"))
    print(f"saved -> {out_path}")

def run_api(problems, out_path, provider, model, workers=12):
    from openai import OpenAI
    if provider == "deepseek":
        client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    else:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    # resume
    if Path(out_path).exists():
        problems = json.load(open(out_path))
        print(f"resuming from {out_path}")
    tasks = []
    for pi, p in enumerate(problems):
        for emotion in EM:
            for intensity in INTENSITIES:
                if p.get("fewshot", {}).get(emotion, {}).get(str(intensity)): continue
                tasks.append((pi, emotion, intensity))
    print(f"todo: {len(tasks)} generations", flush=True)
    def call(pi, emotion, intensity):
        user = build_prompt(problems[pi]["original"], emotion, intensity)
        for _ in range(4):
            try:
                r = client.chat.completions.create(model=model, messages=[{"role":"user","content":user}],
                                                   temperature=0.7, max_tokens=400)
                return r.choices[0].message.content
            except Exception as e:
                print(f"  retry: {str(e)[:120]}", flush=True); time.sleep(3)
        return ""
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_to_k = {ex.submit(call, *t): t for t in tasks}
        for f in as_completed(fut_to_k):
            pi, emotion, intensity = fut_to_k[f]
            text = clean_output(f.result() or "")
            problems[pi].setdefault("fewshot", {}).setdefault(emotion, {})[str(intensity)] = {
                "emotional_text": text,
                "numbers_preserved": nums(problems[pi]["original"]).issubset(nums(text)),
            }
            done += 1
            if done % 200 == 0:
                json.dump(problems, open(out_path, "w"))
                print(f"  {done}/{len(tasks)}", flush=True)
    json.dump(problems, open(out_path, "w"))
    print(f"saved -> {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["local","api"], required=True)
    ap.add_argument("--provider", choices=["openai","deepseek"], default="deepseek")
    ap.add_argument("--model", default=None, help="model name/path; default Llama-3.1-8B-Instruct for local, deepseek-chat for api")
    ap.add_argument("--gpu", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # load TEMPER-5400 problems (just original + id + dataset)
    problems = []
    for ds in ["gsm8k","multiarith","arc_challenge"]:
        for r in json.load(open(BASE / "results_v3" / f"emotionbench_{ds}.json")):
            problems.append({"id": r["id"], "dataset": ds, "original": r["original"]})
    print(f"loaded {len(problems)} problems", flush=True)
    if args.mode == "local":
        m = args.model or "meta-llama/Llama-3.1-8B-Instruct"
        run_local_llama(problems, args.out, gpu=args.gpu, model_path=m)
    else:
        m = args.model or ("deepseek-chat" if args.provider=="deepseek" else "gpt-4o-mini")
        run_api(problems, args.out, args.provider, m)

if __name__ == "__main__":
    main()
