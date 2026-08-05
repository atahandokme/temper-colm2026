# Hand-verification of GPT-4o-flagged REAL_ERROR cases (GSM8K, forward emotionalization, 2400 each)

Verified manually by reading full original vs rewrite. Verdict: REAL = math/relationship/premise changed; FALSE = equivalent or pre-existing original issue.

## DeepSeek-direct — 17 GPT-4o flags -> 13 REAL, 4 FALSE

| id | emotion | verdict | note |
|----|---------|---------|------|
| gsm8k_531 | fear | FALSE | "3 times fewer" preserved verbatim; GPT-4o confused by original's own phrasing |
| gsm8k_146 | joy | FALSE | "half of half" = 1/4 — equivalent |
| gsm8k_334 | anger | FALSE | "RJ has 1 gun" is an inconsistency in the ORIGINAL, copied faithfully |
| gsm8k_334 | surprise | FALSE | same original RJ/Jason issue |
| gsm8k_201 | sadness | **REAL** | premise INVERTED: "John never wins... no $5000... no 5% raise" |
| gsm8k_777 | joy | **REAL** | "$6 of cost" -> "half the cost" ($5) |
| gsm8k_777 | anger | **REAL** | -> "half the cost" |
| gsm8k_777 | disgust | **REAL** | "$6 of cost" -> "$6 off cost" ($4) |
| gsm8k_777 | surprise | **REAL** | -> "half the cost" |
| gsm8k_777 | sadness | **REAL** | -> "half the cost" |
| gsm8k_777 | fear | **REAL** | -> "half the cost" |
| gsm8k_1212 | joy | **REAL** | "1/8 the capacity" -> "half the capacity" |
| gsm8k_1212 | anger | **REAL** | -> "half of one-eighth" (=1/16) |
| gsm8k_1091 | sadness | **REAL** | premise INVERTED: "earned nothing", "no one came to buy" |
| gsm8k_1179 | fear | **REAL** | "7 more than he drills" -> "twice drilled plus 7" |
| gsm8k_427 | fear | **REAL** | "keeps 1/3" -> "keeps exactly half" |
| gsm8k_427 | surprise | **REAL** | "keeps 1/3" -> "keeping half" |

DeepSeek REAL errors cluster on 6 distinct problems (notably gsm8k_777 broken across ALL 6 emotions).
Error types: fraction/relationship swaps (1/8->half, 1/3->half, $6->half, "n more"->"twice plus n") and
sadness-driven premise inversions ("earned nothing", "never wins").

True error rate: **13 / 2400 = 0.54%**

## Trained-translator (benchmark) — 3 GPT-4o flags -> 0 clear REAL, 2 borderline, 1 FALSE

| id | emotion | verdict | note |
|----|---------|---------|------|
| gsm8k_1018 | anger | BORDERLINE | "spends a total of 18 pages" garbled but 18/3/10 preserved; math recoverable |
| gsm8k_424 | sadness | BORDERLINE | added "buses only half full" (flavor) but asks capacity; numbers 60/30/15/4/6/10 intact |
| gsm8k_434 | fear | FALSE | "spent 3 bags of chips" = bought 3; all numbers ($5,40c,75c,3,1%) preserved |

No fraction/relationship/number changes. Flags are phrasing awkwardness, not math corruption.

True error rate: **0-2 / 2400 = 0.00-0.08%**

## Summary (hand-verified true math-error rate, forward emotionalization, GSM8K)

| Source | GPT-4o flags | hand-verified REAL | rate |
|--------|--------------|--------------------|------|
| DeepSeek-direct (one-shot, rules prompt) | 17 | 13 | **0.54%** |
| Trained translator (post-QC) | 3 | 0-2 (borderline) | **0.00-0.08%** |

~7-13x fewer real errors for the trained translator. DeepSeek errors are systematic
(fraction/relationship swaps + premise inversions), not random.
NOTE: translator is post quality-control retry pipeline; DeepSeek is one-shot (the fair baseline reviewers proposed).
