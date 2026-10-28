# Distill A/B results

Every checkpoint re-scored with `experiments/ab_eval.py`. Arms with 2 seeds show mean (seed values). The arms are not like for like: `new` trains on the licence-clean pilot set only (9,890 steps, no TinyChat), `old` on TinyChat + the old distill set (3,949 steps). So the comparison leads with the arithmetic battery, the readable set and story forgetting.


## Arithmetic battery (greedy, exact match on the last number)

| cell | n | arith_plain | arith_scratch |
|---|---|---|---|
| + 1-digit | 30 | 85% (87%, 83%) | 37% (30%, 43%) |
| + 2-digit | 40 | 99% (100%, 98%) | 99% (98%, 100%) |
| + 3-digit | 40 | 96% (98%, 95%) | 85% (85%, 85%) |
| - 1-digit | 30 | 57% (60%, 53%) | 37% (33%, 40%) |
| - 2-digit | 40 | 96% (95%, 98%) | 96% (98%, 95%) |
| - 3-digit | 40 | 79% (82%, 75%) | 81% (82%, 80%) |
| × 1-digit | 30 | 0% (0%, 0%) | 3% (3%, 3%) |
| × 2-digit | 40 | 1% (2%, 0%) | 2% (2%, 2%) |
| × 3-digit | 40 | 0% (0%, 0%) | 0% (0%, 0%) |
| ÷ 1-digit divisor | 40 | 14% (15%, 12%) | 5% (2%, 8%) |
| ÷ 2-digit divisor | 40 | 15% (18%, 12%) | 0% (0%, 0%) |
| ALL | 410 | 49% (51%, 48%) | 42% (41%, 42%) |
| ends with `Answer: <gold>` | | 49% (51%, 48%) | 40% (39%, 40%) |

## Readable set (40 prompts)

Side by side in `readable.md`. Maths prompts answered correctly (last number = gold, of 10):

| decoding | arith_plain | arith_scratch |
|---|---|---|
| greedy | 5 (5, 5) | 3 (3, 3) |
| sampled | 4 (3, 5) | 4 (3, 4) |

## Story val (forgetting), before -> after

| arm | story val after | change |
|---|---|---|
| arith_plain | 2.7486 (2.7516, 2.7455) | - |
| arith_scratch | 2.7468 (2.7494, 2.7441) | - |

## Other frozen val sets (reply-only loss, lower is better)

TinyChat val is **not comparable** between arms: `old` trains on TinyChat and `new` never sees it. Old distill val favours `old` the same way, and pilot val favours `new`.

| val set | arith_plain | arith_scratch |
|---|---|---|
| chat (TinyChat) (not comparable) | 3.2646 (3.2646, 3.2646) | 3.3009 (3.2964, 3.3054) |
| old distill | 3.3028 (3.3106, 3.2950) | 3.3678 (3.3831, 3.3526) |
| pilot arith | 1.2527 (1.2497, 1.2557) | 1.3479 (1.3505, 1.3453) |
| pilot chat | 2.6479 (2.6453, 2.6505) | 2.6749 (2.6723, 2.6775) |
| pilot explain | 2.7498 (2.7475, 2.7521) | 2.7770 (2.7749, 2.7790) |
| pilot gsm8k | 1.6239 (1.6212, 1.6266) | 1.8590 (1.8585, 1.8595) |
| pilot oasst2 | 3.0223 (3.0216, 3.0230) | 3.0481 (3.0478, 3.0483) |
| pilot story | 1.8864 (1.8830, 1.8898) | 1.9058 (1.9021, 1.9094) |
| pilot all | 2.1129 (2.1100, 2.1157) | 2.1579 (2.1561, 2.1598) |

## math_eval exam (120 questions, its own near-greedy decoding with repetition penalty)

| kind | arith_plain | arith_scratch |
|---|---|---|
| add 1-digit | 12 (11, 12) | 0 (1, 0) |
| add 2-digit | 0 (0, 0) | 0 (0, 0) |
| subtract | 2 (3, 2) | 0 (0, 0) |
| multiply | 0 (0, 0) | 0 (0, 0) |
| sum of 3 | 0 (0, 0) | 0 (0, 1) |
| word problem | 0 (0, 1) | 0 (0, 0) |
| TOTAL | 14 (14, 15) | 1 (1, 1) |

## Format and repetition, greedy (n = 400 prompts)

| metric | arith_plain | arith_scratch |
|---|---|---|
| ends in "?" (chat/explain prompts) | 52% (51%, 52%) | 37% (35%, 39%) |
| ends in "?" (story/maths prompts) | 0% (0%, 1%) | 0% (0%, 1%) |
| turn ended (wrote "You:") | 57% (60%, 54%) | 42% (41%, 42%) |
| distinct-2 per reply | 0.711 (0.727, 0.695) | 0.661 (0.681, 0.640) |
| distinct-3 per reply | 0.769 (0.786, 0.752) | 0.726 (0.749, 0.704) |
| distinct-2 across replies | 0.312 (0.320, 0.305) | 0.292 (0.302, 0.282) |
| replies with a repeated 4-gram | 68% (66%, 71%) | 77% (77%, 77%) |
| replies with a 4-gram 3+ times | 44% (42%, 46%) | 52% (50%, 54%) |
| mean words | 75 (72, 78) | 93 (94, 93) |
| most common 3-word opener share | 13% (13%, 13%) | 13% (13%, 12%) |
| most common opener | "once upon a"; "once upon a" | "once upon a"; "once upon a" |

## Format and repetition, sampled (n = 400 prompts)

| metric | arith_plain | arith_scratch |
|---|---|---|
| ends in "?" (chat/explain prompts) | 46% (47%, 44%) | 36% (39%, 34%) |
| ends in "?" (story/maths prompts) | 0% (0%, 0%) | 0% (0%, 1%) |
| turn ended (wrote "You:") | 58% (59%, 56%) | 41% (44%, 38%) |
| distinct-2 per reply | 0.899 (0.892, 0.905) | 0.870 (0.881, 0.858) |
| distinct-3 per reply | 0.952 (0.946, 0.959) | 0.934 (0.943, 0.925) |
| distinct-2 across replies | 0.512 (0.512, 0.513) | 0.495 (0.505, 0.484) |
| replies with a repeated 4-gram | 42% (42%, 42%) | 51% (49%, 54%) |
| replies with a 4-gram 3+ times | 12% (13%, 11%) | 18% (15%, 20%) |
| mean words | 88 (83, 92) | 99 (98, 101) |
| most common 3-word opener share | 12% (11%, 12%) | 12% (12%, 12%) |
| most common opener | "once upon a"; "once upon a" | "once upon a"; "once upon a" |
