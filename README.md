# TEMPER: Testing Emotional Perturbation in Quantitative Reasoning

Official code release for the COLM 2026 paper **TEMPER: Testing Emotional Perturbation in Quantitative Reasoning**.

**Benchmark:** [TEMPER-5400 on Hugging Face](https://huggingface.co/datasets/atahandokme/temper-5400): 5,400 verified emotion–neutral pairs over 900 problems from GSM8K, MultiArith, and ARC-Challenge.

TEMPER shows that emotional framing of quantitative problems degrades LLM reasoning by 2–10 percentage points across eighteen models (1B to frontier scale), even when all numbers, quantifiers, and relationships are preserved — and that neutralizing the emotional text recovers most of the lost performance.

## Repository Structure

Each directory corresponds to a component of the paper.

| Directory | Paper Section | Contents |
|---|---|---|
| `training/` | §3, Appendix A | Translator training (cross-entropy + emotion-latent auxiliary loss); bottleneck teacher training and module |
| `evaluation/` | §5, Appendix G | Reasoning harness (vLLM and API backends); answer extraction and math-value-preservation utilities; few-shot exemplars; translation quality metrics |
| `benchmark/` | §5 | Ensemble selection across the five translator variants; Hugging Face dataset builder |
| `experiments/lambda_curve/` | §3.2, Appendix C | Intermediate-λ variant generation; two-stage fidelity filter (Claude Haiku judge + GPT-4o adjudication); math-faithful subset construction; few-shot prompting baselines |
| `experiments/style_controls/` | §5.2, Appendix K | Gross-but-neutral, archaic, and bureaucratic control generation; perplexity measurement; deployment-aware naturalness judge |
| `experiments/sysprompt/` | §5.1, Appendix O | System-prompt-vs-neutralization evaluation |
| `experiments/deepseek_direct/` | §5.4, Appendix R | Directly prompted DeepSeek-V3 baseline and error-audit pipeline |
| `experiments/representation/` | §5.5, Appendix S | Teacher-latent t-SNE; solver hidden-state extraction; representational analysis figures |
| `experiments/paraphrase_control/` | §5.2, Appendix J | Non-emotional paraphrase evaluation |
| `data/samples/` | — | Ten problems per source dataset with all six emotional translations and neutralizations |
| `data/audit/` | Appendices C, L, R | Fidelity audit records: the 286-case decreased-length failure list, hand-verified GSM8K error audit, λ=70 adjudication audit, and the 20 user-style reference messages |

## Environment

Python 3.10 with PyTorch, Transformers, PEFT, and vLLM (see `requirements.txt`). API-based scripts read credentials from environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`) or a `.env` file via `python-dotenv`. Scripts were developed on a shared GPU cluster and some retain their original data paths, which document the exact pipeline layout. Prompt templates for every pipeline stage are provided in the paper appendix.

## Translator Prompt Formats

The trained translators are sensitive to the exact inference prompt.

- **Emotionalize:** `Translate this text from neutral to {emotion}.` (temperature 0.7)
- **Neutralize:** `Rewrite this text in a neutral tone.` (temperature 0.3)

Both are raw completion prompts followed by `Input: {text}` and `Output:` markers. Other phrasings can degrade output quality substantially.

## Citation

```bibtex
@article{dokme2026temper,
  title={TEMPER: Testing Emotional Perturbation in Quantitative Reasoning},
  author={Dokme, Atahan and Reichman, Benjamin and Heck, Larry},
  journal={arXiv preprint arXiv:2604.07801},
  year={2026}
}
```

## License

Code is released under the MIT License. The TEMPER-5400 dataset is released under CC BY 4.0.
