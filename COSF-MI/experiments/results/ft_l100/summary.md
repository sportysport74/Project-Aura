# Distill A/B results

Every checkpoint re-scored with `experiments/ab_eval.py`. Arms with 2 seeds show mean (seed values). The arms are not like for like: `new` trains on the licence-clean pilot set only (9,890 steps, no TinyChat), `old` on TinyChat + the old distill set (3,949 steps). So the comparison leads with the arithmetic battery, the readable set and story forgetting.


## Arithmetic battery (greedy, exact match on the last number)

| cell | n | base | l100 |
|---|---|---|---|
| + 1-digit | 30 | 33% | 30% (37%, 23%) |
| + 2-digit | 40 | 18% | 31% (35%, 28%) |
| + 3-digit | 40 | 15% | 25% (30%, 20%) |
| - 1-digit | 30 | 47% | 45% (50%, 40%) |
| - 2-digit | 40 | 12% | 55% (52%, 58%) |
| - 3-digit | 40 | 12% | 40% (38%, 42%) |
| × 1-digit | 30 | 27% | 57% (60%, 53%) |
| × 2-digit | 40 | 0% | 1% (0%, 2%) |
| × 3-digit | 40 | 0% | 0% (0%, 0%) |
| ÷ 1-digit divisor | 40 | 25% | 24% (25%, 22%) |
| ÷ 2-digit divisor | 40 | 8% | 28% (22%, 32%) |
| ALL | 410 | 17% | 30% (30%, 29%) |
| ends with `Answer: <gold>` | | 17% | 30% (30%, 29%) |

## Readable set (40 prompts)

Side by side in `readable.md`. Maths prompts answered correctly (last number = gold, of 10):

| decoding | base | l100 |
|---|---|---|
| greedy | 1 | 4 (3, 5) |
| sampled | 2 | 3 (4, 2) |
| sampled_rep | 0 | 2 (2, 2) |

## Story val (forgetting), before -> after

Forgetting set: 512 windows of data/corpus_v1/tok24k_v2 val (the base's pretraining corpus).

Before = the base `l100_on_s1` scored by this code: 2.8406.

| arm | story val after | change |
|---|---|---|
| base | 2.8406 | +0.0000 |
| l100 | 2.9029 (2.9027, 2.9031) | +0.0623 (+0.0621, +0.0625) |

## Other frozen val sets (reply-only loss, lower is better)

TinyChat val is **not comparable** between arms: `old` trains on TinyChat and `new` never sees it. Old distill val favours `old` the same way, and pilot val favours `new`.

| val set | base | base | l100 |
|---|---|---|---|
| chat (TinyChat) (not comparable) | 3.5070 | 3.5070 | 3.3497 (3.3513, 3.3482) |
| old distill | 3.3499 | 3.3499 | 3.1293 (3.1334, 3.1252) |
| pilot arith | 1.3928 | 1.3928 | 0.5480 (0.5479, 0.5480) |
| pilot chat | 3.0276 | 3.0276 | 2.2538 (2.2538, 2.2539) |
| pilot explain | 3.0174 | 3.0174 | 2.3636 (2.3638, 2.3633) |
| pilot gsm8k | 1.8052 | 1.8052 | 1.2467 (1.2480, 1.2453) |
| pilot oasst2 | 3.1592 | 3.1592 | 2.6823 (2.6818, 2.6828) |
| pilot story | 2.2682 | 2.2682 | 1.7287 (1.7282, 1.7292) |
| pilot all | 2.4140 | 2.4140 | 1.7605 (1.7604, 1.7606) |

## math_eval exam (120 questions, its own near-greedy decoding with repetition penalty)

| kind | base | l100 |
|---|---|---|
| add 1-digit | 2 | 4 (4, 3) |
| add 2-digit | 0 | 0 (0, 0) |
| subtract | 0 | 0 (0, 1) |
| multiply | 0 | 0 (0, 1) |
| sum of 3 | 0 | 0 (0, 1) |
| word problem | 0 | 1 (1, 1) |
| TOTAL | 2 | 6 (5, 7) |

## Format and repetition, greedy (n = 400 prompts)

| metric | base | l100 |
|---|---|---|
| ends in "?" (chat/explain prompts) | 12% | 12% (11%, 13%) |
| ends in "?" (story/maths prompts) | 1% | 0% (0%, 0%) |
| turn ended (wrote "You:") | 89% | 95% (95%, 95%) |
| distinct-2 per reply | 0.764 | 0.818 (0.816, 0.820) |
| distinct-3 per reply | 0.820 | 0.869 (0.868, 0.870) |
| distinct-2 across replies | 0.314 | 0.397 (0.397, 0.398) |
| replies with a repeated 4-gram | 64% | 54% (53%, 55%) |
| replies with a 4-gram 3+ times | 32% | 20% (18%, 23%) |
| mean words | 55 | 51 (51, 52) |
| most common 3-word opener share | 6% | 12% (12%, 12%) |
| most common opener | "it's like a" | "once upon a"; "once upon a" |

## Format and repetition, sampled (n = 400 prompts)

| metric | base | l100 |
|---|---|---|
| ends in "?" (chat/explain prompts) | 24% | 27% (29%, 25%) |
| ends in "?" (story/maths prompts) | 2% | 0% (0%, 0%) |
| turn ended (wrote "You:") | 91% | 98% (98%, 98%) |
| distinct-2 per reply | 0.896 | 0.925 (0.925, 0.925) |
| distinct-3 per reply | 0.945 | 0.966 (0.966, 0.966) |
| distinct-2 across replies | 0.502 | 0.556 (0.557, 0.554) |
| replies with a repeated 4-gram | 36% | 25% (26%, 24%) |
| replies with a 4-gram 3+ times | 10% | 4% (4%, 4%) |
| mean words | 61 | 52 (51, 53) |
| most common 3-word opener share | 3% | 11% (11%, 11%) |
| most common opener | "sure aura sure" | "once upon a"; "once upon a" |

## Format and repetition, sampled_rep (n = 400 prompts)

| metric | base | l100 |
|---|---|---|
| ends in "?" (chat/explain prompts) | 18% | 67% (68%, 66%) |
| ends in "?" (story/maths prompts) | 3% | 0% (1%, 0%) |
| turn ended (wrote "You:") | 98% | 99% (100%, 99%) |
| distinct-2 per reply | 0.990 | 0.989 (0.990, 0.988) |
| distinct-3 per reply | 0.997 | 0.997 (0.998, 0.996) |
| distinct-2 across replies | 0.579 | 0.585 (0.583, 0.587) |
| replies with a repeated 4-gram | 4% | 3% (3%, 4%) |
| replies with a 4-gram 3+ times | 0% | 0% (0%, 0%) |
| mean words | 57 | 59 (60, 58) |
| most common 3-word opener share | 3% | 12% (12%, 12%) |
| most common opener | "can you help" | "once upon a"; "once upon a" |
