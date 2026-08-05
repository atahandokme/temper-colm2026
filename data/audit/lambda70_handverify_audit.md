# λ=70 hand-verify audit (backfill, Aug 4 2026)

Backfills the missing λ=70 entry in `results_v3/lambda_curve/handverify/` (λ ∈ {10, 40, 60, 80} were sampled during the rebuttal; 70 was skipped). Sample: 50 Haiku-BROKEN cases drawn from `adjudication/emo100_70__gpt4o_adj.json` with `rebuttal_handverify_sample.py --lam 70 --seed 70`. Each case was re-verified blind (GPT-4o verdicts hidden during judging), then compared against the adjudicator. Per-case verdicts + reasons: `handverify/emo100_70__handverify_verdicts.json`.

## Verification criterion

Same as the GPT-4o adjudicator's operational question: *would a competent solver, given the eval-time prompt, reach the original ground-truth answer?* Note the eval-time detail that matters for ARC: `scripts/06c_reasoning_test_vllm.py::ensure_mcq_choices` strips any A–D choices from the rewrite and re-appends the ORIGINAL choices for MCQ datasets. Mangled/dropped options therefore do NOT break the evaluation — only stem semantics can. Several Haiku BROKEN flags (and at least one GPT-4o REAL_ERROR, case 45) cite dropped options and are false alarms under the eval-time criterion.

## Results (n=50)

| Verdict | Fable (blind) | GPT-4o (same 50) |
|---|---|---|
| truly broken | 30 | 25 |
| false alarm | 16 | 25 |
| borderline | 4 | — |

- **Binary agreement: 34/46 (74%)** on non-borderline cases.
- Sample true-broken rate: 60–68% (30–34/50 counting borderlines) vs GPT-4o's 50% on this sample; population-level GPT-4o gives 53.3% (2,098/3,939) → the reported λ=70 true-broken rate of 38.9% (of all 5,400) is, if anything, a mild **underestimate**. This is consistent with the appendix's conservative-λ argument (fidelity collapses at high λ) — it strengthens, not weakens, that justification.

## Systematic disagreement pattern (12 cases)

**GPT-4o is too lenient on silent constraint omission (9 cases: 2, 4, 7, 8, 15, 17, 21, 43, 48).** All are GSM8K/MultiArith rewrites that preserve every surface number but silently drop a constraint or the question itself: truck capacity 2,250 lb (case 2), "eats twice as much as his girlfriend" (7), the daily frequency of the 15-min call (15), "how much would Cameron save?" removed entirely (21), the trips question replaced by "Can he make the terror of the two tables unspoken?" (48). These mirror the DeepSeek-direct systematic error classes (relationship drops, premise changes) found in the GSM8K rebuttal audit — the adjudicator under-catches omissions because all the digits are still present.

**GPT-4o is too strict on 3 recoverable cases (23, 34, 45)**, flagging garbled-but-computable stems (34: $140/mo, first half at full price, 10% less second half — fully recoverable) and one ARC option-drop that eval-time choice restoration repairs (45).

## Caveat for the λ-curve appendix

The math-faithful subset at high λ is Haiku-NOT-BROKEN ∪ GPT-4o rescues. This sample suggests ~1/3 of GPT-4o's rescues at λ=70 (9 of 25 here) silently lack a constraint, i.e. some residual contamination survives into the λ=70 subset. Two honest framings for the camera-ready: (i) report the λ-curve drops as computed, noting adjudicated fidelity is imperfect at extreme λ and the true-broken rates are lower bounds; (ii) note the direction — residual contamination can only inflate the measured drop at λ ≥ 40, which is one more reason the paper's conservative λ=0.02 operating point (where Haiku+manual review agree fidelity is ~99%) is the right regime for the headline claims. The paper's main results are unaffected: at λ=0.02 the pipeline includes full manual review.
