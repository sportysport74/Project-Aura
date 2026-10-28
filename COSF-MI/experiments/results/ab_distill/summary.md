# Distill A/B results

Every checkpoint re-scored with `experiments/ab_eval.py`. Arms with 2 seeds show mean (seed values). The arms are not like for like: `new` trains on the licence-clean pilot set only (9,890 steps, no TinyChat), `old` on TinyChat + the old distill set (3,949 steps). So the comparison leads with the arithmetic battery, the readable set and story forgetting.


## Arithmetic battery (greedy, exact match on the last number)

| cell | n | new | old | ref_aura_chat_distill |
|---|---|---|---|---|
| + 1-digit | 30 | 20% (20%, 20%) | 7% (7%, 7%) | 10% |
| + 2-digit | 40 | 26% (32%, 20%) | 1% (0%, 2%) | 2% |
| + 3-digit | 40 | 12% (15%, 10%) | 0% (0%, 0%) | 0% |
| - 1-digit | 30 | 17% (10%, 23%) | 0% (0%, 0%) | 0% |
| - 2-digit | 40 | 45% (30%, 60%) | 0% (0%, 0%) | 0% |
| - 3-digit | 40 | 16% (12%, 20%) | 0% (0%, 0%) | 0% |
| × 1-digit | 30 | 0% (0%, 0%) | 3% (7%, 0%) | 3% |
| × 2-digit | 40 | 1% (2%, 0%) | 0% (0%, 0%) | 0% |
| × 3-digit | 40 | 0% (0%, 0%) | 0% (0%, 0%) | 0% |
| ÷ 1-digit divisor | 40 | 2% (2%, 2%) | 0% (0%, 0%) | 0% |
| ÷ 2-digit divisor | 40 | 2% (2%, 2%) | 0% (0%, 0%) | 0% |
| ALL | 410 | 13% (12%, 14%) | 1% (1%, 1%) | 1% |
| ends with `Answer: <gold>` | | 13% (12%, 14%) | 0% (0%, 0%) | 0% |

## Readable set (40 prompts)

Side by side in `readable.md`. Maths prompts answered correctly (last number = gold, of 10):

| decoding | new | old | ref_aura_chat_distill |
|---|---|---|---|
| greedy | 2 (3, 2) | 0 (0, 0) | 0 |
| sampled | 2 (2, 1) | 0 (0, 0) | 0 |

## Story val (forgetting), before -> after

| arm | story val after | change |
|---|---|---|
| new | 2.7814 (2.7816, 2.7813) | - |
| old | 2.7890 (2.7891, 2.7889) | - |
| ref_aura_chat_distill | 2.7893 | - |

## Other frozen val sets (reply-only loss, lower is better)

TinyChat val is **not comparable** between arms: `old` trains on TinyChat and `new` never sees it. Old distill val favours `old` the same way, and pilot val favours `new`.

| val set | new | old | ref_aura_chat_distill |
|---|---|---|---|
| chat (TinyChat) (not comparable) | 3.3898 (3.3937, 3.3859) | 1.8814 (1.8813, 1.8815) | 1.8780 |
| old distill | 3.2740 (3.2828, 3.2653) | 1.4915 (1.4910, 1.4921) | 1.4910 |
| pilot arith | 0.6349 (0.6351, 0.6347) | 3.1310 (3.1168, 3.1451) | 3.0930 |
| pilot chat | 2.4875 (2.4883, 2.4868) | 3.6230 (3.6203, 3.6257) | 3.6169 |
| pilot explain | 2.5937 (2.5937, 2.5937) | 3.7686 (3.7661, 3.7710) | 3.7608 |
| pilot gsm8k | 1.4603 (1.4600, 1.4605) | 2.6131 (2.6067, 2.6195) | 2.5944 |
| pilot oasst2 | 2.9142 (2.9137, 2.9148) | 3.7111 (3.7099, 3.7124) | 3.7068 |
| pilot story | 1.7664 (1.7664, 1.7663) | 2.9968 (3.0039, 2.9896) | 2.9928 |
| pilot all | 1.8948 (1.8950, 1.8946) | 3.2910 (3.2900, 3.2919) | 3.2795 |

## math_eval exam (120 questions, its own near-greedy decoding with repetition penalty)

| kind | new | old | ref_aura_chat_distill |
|---|---|---|---|
| add 1-digit | 0 (0, 1) | 1 (0, 2) | 1 |
| add 2-digit | 0 (0, 0) | 0 (0, 0) | 0 |
| subtract | 0 (1, 0) | 1 (1, 1) | 1 |
| multiply | 0 (0, 1) | 0 (0, 1) | 2 |
| sum of 3 | 0 (0, 0) | 0 (0, 0) | 1 |
| word problem | 1 (1, 1) | 0 (0, 0) | 0 |
| TOTAL | 2 (2, 3) | 2 (1, 4) | 5 |

## Format and repetition, greedy (n = 400 prompts)

| metric | new | old | ref_aura_chat_distill |
|---|---|---|---|
| ends in "?" (chat/explain prompts) | 5% (6%, 4%) | 7% (6%, 8%) | 8% |
| ends in "?" (story/maths prompts) | 0% (0%, 0%) | 3% (3%, 3%) | 1% |
| turn ended (wrote "You:") | 96% (96%, 95%) | 98% (98%, 98%) | 99% |
| distinct-2 per reply | 0.806 (0.811, 0.800) | 0.963 (0.965, 0.961) | 0.972 |
| distinct-3 per reply | 0.861 (0.869, 0.854) | 0.975 (0.976, 0.973) | 0.982 |
| distinct-2 across replies | 0.367 (0.374, 0.360) | 0.439 (0.441, 0.438) | 0.445 |
| replies with a repeated 4-gram | 56% (57%, 55%) | 5% (5%, 5%) | 4% |
| replies with a 4-gram 3+ times | 23% (22%, 24%) | 2% (2%, 3%) | 1% |
| mean words | 47 (47, 48) | 14 (14, 14) | 13 |
| most common 3-word opener share | 12% (12%, 12%) | 7% (7%, 7%) | 8% |
| most common opener | "once upon a"; "once upon a" | "that sounds like"; "that sounds like" | "that sounds like" |

## Format and repetition, sampled (n = 400 prompts)

| metric | new | old | ref_aura_chat_distill |
|---|---|---|---|
| ends in "?" (chat/explain prompts) | 19% (18%, 20%) | 8% (10%, 7%) | 8% |
| ends in "?" (story/maths prompts) | 0% (0%, 0%) | 5% (3%, 7%) | 11% |
| turn ended (wrote "You:") | 98% (98%, 99%) | 99% (98%, 99%) | 99% |
| distinct-2 per reply | 0.932 (0.927, 0.936) | 0.986 (0.985, 0.988) | 0.985 |
| distinct-3 per reply | 0.971 (0.969, 0.974) | 0.994 (0.993, 0.995) | 0.995 |
| distinct-2 across replies | 0.542 (0.542, 0.543) | 0.623 (0.611, 0.635) | 0.621 |
| replies with a repeated 4-gram | 21% (23%, 18%) | 2% (2%, 1%) | 1% |
| replies with a 4-gram 3+ times | 4% (3%, 4%) | 0% (0%, 0%) | 0% |
| mean words | 51 (51, 51) | 14 (14, 14) | 14 |
| most common 3-word opener share | 10% (10%, 10%) | 6% (6%, 5%) | 5% |
| most common opener | "once upon a"; "once upon a" | "that sounds like"; "that sounds like" | "that sounds like" |
