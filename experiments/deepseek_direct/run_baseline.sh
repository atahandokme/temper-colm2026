#!/bin/bash
# Reasoning eval on a DeepSeek baseline (one model per GPU). Usage:
#   ./rebuttal_run_baseline.sh <LABEL> <DATASETS_CSV> <MODEL_NAME> <MODEL_PATH> <GPU>
# LABEL dir = results_v3/baseline/<LABEL>/emotionbench_<ds>.json
set -e
LABEL="$1"; DATASETS="$2"; MODEL_NAME="$3"; MODEL_PATH="$4"; GPU="$5"
PY=/nethome/adokme3/.conda/envs/emotion_llama/bin/python
cd /home/heck2/adokme3/emotion-translator-training/translator_evaluation
mkdir -p configs/baseline results_v3/baseline_reasoning logs_rebuttal

OUT="results_v3/baseline_reasoning/${LABEL}__${MODEL_NAME}.json"
if [ -f "$OUT" ]; then echo "[skip] $OUT"; exit 0; fi
CFG="configs/baseline/${LABEL}__${MODEL_NAME}.yaml"
{
  echo "emotionbench_files:"
  IFS=','; for ds in $DATASETS; do echo "  - \"results_v3/baseline/${LABEL}/emotionbench_${ds}.json\""; done
  echo "prompt_style: zero_shot_cot"
  echo "llms:"
  echo "  - name: \"${MODEL_NAME}\""
  echo "    model_path: \"${MODEL_PATH}\""
  echo "tensor_parallel_size: 1"
  echo "generation:"
  echo "  temperature: 0.0"
  echo "  max_new_tokens: 2048"
  echo "  max_new_tokens_mcq: 128"
  echo "output_file: \"${OUT}\""
} > "$CFG"
echo "[run] $LABEL $MODEL_NAME gpu=$GPU -> $OUT"
CUDA_VISIBLE_DEVICES=$GPU TOKENIZERS_PARALLELISM=false TORCH_COMPILE_DISABLE=1 \
  $PY scripts/06c_reasoning_test_vllm.py --config "$CFG"
echo "[done] $LABEL $MODEL_NAME"
