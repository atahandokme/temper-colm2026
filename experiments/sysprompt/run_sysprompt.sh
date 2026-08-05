#!/bin/bash
# System-prompt vs neutralization: run the TEMPER benchmark (emotionbench_*) through
# reasoning WITH and WITHOUT a mitigation system prompt, same models. Usage:
#   ./rebuttal_run_sysprompt.sh <MODEL_NAME> <MODEL_PATH> <GPU>
set -e
MODEL_NAME="$1"; MODEL_PATH="$2"; GPU="$3"
PY=/nethome/adokme3/.conda/envs/emotion_llama/bin/python
cd /home/heck2/adokme3/emotion-translator-training/translator_evaluation
mkdir -p configs/sysprompt results_v3/sysprompt_reasoning logs_rebuttal
SYS="Only focus on the math problem itself. Ignore any emotional, dramatic, or non-functional language, and solve the underlying problem."

for COND in nosys sys; do
  OUT="results_v3/sysprompt_reasoning/${COND}__${MODEL_NAME}.json"
  [ -f "$OUT" ] && { echo "[skip] $OUT"; continue; }
  CFG="configs/sysprompt/${COND}__${MODEL_NAME}.yaml"
  {
    echo "emotionbench_files:"
    echo "  - \"results_v3/emotionbench_gsm8k.json\""
    echo "  - \"results_v3/emotionbench_multiarith.json\""
    echo "  - \"results_v3/emotionbench_arc_challenge.json\""
    echo "prompt_style: zero_shot_cot"
    [ "$COND" = "sys" ] && echo "system_prompt: \"${SYS}\""
    echo "llms:"
    echo "  - name: \"${MODEL_NAME}\""
    echo "    model_path: \"${MODEL_PATH}\""
    echo "tensor_parallel_size: 1"
    echo "generation: {temperature: 0.0, max_new_tokens: 2048, max_new_tokens_mcq: 128}"
    echo "output_file: \"${OUT}\""
  } > "$CFG"
  echo "[run] $COND $MODEL_NAME gpu=$GPU"
  CUDA_VISIBLE_DEVICES=$GPU TOKENIZERS_PARALLELISM=false TORCH_COMPILE_DISABLE=1 \
    $PY scripts/06c_reasoning_test_vllm.py --config "$CFG"
done
echo "[done] $MODEL_NAME"
