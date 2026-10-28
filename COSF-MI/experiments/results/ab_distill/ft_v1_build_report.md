# Fine-tune set build report

Source `data/distill_pilot/full/kept.jsonl`, args `{'src': 'data/distill_pilot/full/kept.jsonl', 'out': 'data/distill_pilot/ft_v1', 'simile_cap': 0.3, 'quip_cap': 0.01, 'arith_share': 0.25, 'val_per_category': 2000, 'seed': 0}`.

## Sizes (samples / AURA tokens)

| category | start | simile cap | quip thin | train | val | train token share |
|---|---|---|---|---|---|---|
| explain | 186,131 / 19.35M | 115,265 / 11.97M | - | 113,265 / 11.76M | 2,000 / 0.21M | 34.4% |
| arith | 228,166 / 19.19M | 187,952 / 15.74M | 92,338 / 7.69M | 90,205 / 7.51M | 2,000 / 0.17M | 22.0% |
| chat | 106,302 / 10.31M | 54,422 / 5.25M | - | 52,422 / 5.06M | 2,000 / 0.19M | 14.8% |
| story | 54,906 / 9.05M | 54,906 / 9.05M | - | 52,906 / 8.72M | 2,000 / 0.33M | 25.5% |
| oasst2 | 3,798 / 0.42M | 2,990 / 0.33M | - | 2,790 / 0.31M | 200 / 0.02M | 0.9% |
| gsm8k | 5,403 / 0.94M | 5,403 / 0.94M | 5,214 / 0.90M | 4,914 / 0.85M | 300 / 0.05M | 2.5% |
| **total** | 584,706 / 59.25M | 420,938 / 43.28M | | 316,502 / 34.22M | 8,500 / 0.97M | |

Arith samples dropped because an eval uses their (op, a, b): 133.

## Simile openers (share of replies)

| category | before | after cap | in train |
|---|---|---|---|
| explain | 56.7% | 30.0% | 30.0% |
| arith | 42.3% | 30.0% | 25.6% |
| chat | 64.2% | 30.0% | 30.0% |
| story | 0.4% | 0.4% | 0.4% |
| oasst2 | 44.9% | 30.0% | 29.9% |
| gsm8k | 6.0% | 6.0% | 5.8% |

## Playful lines, arith (187,952 → 92,338 replies; max single feature after: 1.00%)

Top 20 lines before cover 24.7% of replies (sum of shares).

| # | line before | share | line after | share |
|---|---|---|---|---|
| 1 | sure thing | 9.61% | let's see how many each friend gets | 1.00% |
| 2 | that's a lot of <thing> | 2.11% | let's see how many she has left | 0.98% |
| 3 | wow that's a lot of <thing> | 1.30% | let's see how many he has left | 0.98% |
| 4 | let's add those <thing> together | 1.07% | let's see how many each gets | 0.98% |
| 5 | let's see how many each gets | 0.94% | sure thing | 0.90% |
| 6 | let's see how many each friend gets | 0.80% | let's see how many are left | 0.80% |
| 7 | let's see how many are left | 0.77% | let's do the math | 0.79% |
| 8 | how many would you have left | 0.75% | how many are left | 0.76% |
| 9 | <name> must have felt a bit lighter after giving away those <thing> | 0.73% | how many would you have left | 0.69% |
| 10 | how many are left | 0.73% | let's add those <thing> together | 0.62% |
| 11 | let's do the math | 0.72% | <name> must have felt a bit lighter after giving away those <thing> | 0.60% |
| 12 | let's see how many he has left | 0.72% | let's count them all together | 0.58% |
| 13 | each friend gets a fair share | 0.72% | each friend gets a fair share | 0.54% |
| 14 | let's see how many she has left | 0.69% | let's count them together | 0.45% |
| 15 | that's a lot of <thing> isn't it | 0.54% | let's count them up | 0.38% |
| 16 | let's count them all together | 0.52% | let's find out | 0.37% |
| 17 | let's count them together | 0.52% | let's see how many she has now | 0.33% |
| 18 | let's see how many candies each friend gets | 0.49% | let's count them all up | 0.30% |
| 19 | sure thing little buddy | 0.47% | let's see how many <thing> each friend gets | 0.27% |
| 20 | let's see how many she has now | 0.45% | how many do you have left | 0.26% |

| metaphor family | before | after |
|---|---|---|
| pile | 10.69% | 0.88% |
| candies | 10.55% | 0.97% |
| sure thing | 10.44% | 0.99% |
| that's a lot of | 10.12% | 0.99% |
| collection grew | 7.02% | 0.98% |
| mountain | 5.05% | 0.98% |
| tower / stacking | 4.45% | 0.99% |
| pizza / slices | 3.52% | 0.99% |
| jar | 3.47% | 0.98% |
| stash | 2.82% | 0.93% |
| snowball / hill | 2.62% | 0.95% |
| library | 2.32% | 0.98% |
| party | 1.97% | 1.00% |
| stars / sky | 1.74% | 0.98% |
| fair share | 1.62% | 0.98% |
| treasure | 1.52% | 0.98% |
| garden | 1.35% | 0.99% |
| felt lighter | 1.26% | 0.98% |
| puzzle | 0.51% | 0.69% |

## Playful lines, gsm8k (5,403 → 5,214 replies; max single feature after: 1.00%)

Top 20 lines before cover 5.0% of replies (sum of shares).

| # | line before | share | line after | share |
|---|---|---|---|---|
| 1 | let's break it down | 0.70% | let's break it down | 0.52% |
| 2 | let's count them up | 0.54% | now let's add them all up | 0.48% |
| 3 | now let's add them all up | 0.46% | now let's add them up | 0.38% |
| 4 | now let's add them up | 0.39% | let's count them up | 0.36% |
| 5 | let's think about this like a fun puzzle | 0.39% | let's break it down step by step | 0.29% |
| 6 | let's break it down step by step | 0.31% | now let's add it all up | 0.23% |
| 7 | let's do the math | 0.28% | let's think about this like a fun puzzle | 0.23% |
| 8 | now let's add it all up | 0.26% | let's do the math | 0.21% |
| 9 | okay let's think about this like a fun puzzle | 0.22% | let's think about this step by step | 0.17% |
| 10 | let's think about this step by step | 0.17% | okay let's think about this like a fun puzzle | 0.15% |
| 11 | that's a lot of <thing> | 0.17% | so we have | 0.15% |
| 12 | now let's add them all together | 0.17% | now let's add them all together | 0.13% |
| 13 | so we have | 0.15% | let's calculate that | 0.13% |
| 14 | let's calculate that | 0.13% | okay let's think this through | 0.12% |
| 15 | wow that's a lot of <thing> | 0.11% | sure thing | 0.12% |
| 16 | let's figure this out | 0.11% | let's think this through step by step | 0.12% |
| 17 | okay let's think this through | 0.11% | let's figure this out | 0.10% |
| 18 | sure thing | 0.11% | let's think about this like a treasure hunt | 0.10% |
| 19 | let's think this through step by step | 0.11% | oh no | 0.08% |
| 20 | okay let's figure this out | 0.09% | let's count it up | 0.08% |

| metaphor family | before | after |
|---|---|---|
| that's a lot of | 3.76% | 1.00% |
| puzzle | 1.39% | 0.98% |
| pizza / slices | 1.30% | 0.98% |
| candies | 0.81% | 0.84% |
| jar | 0.63% | 0.63% |
| party | 0.52% | 0.44% |
| treasure | 0.44% | 0.46% |
| pile | 0.44% | 0.46% |
| tower / stacking | 0.41% | 0.36% |
| stars / sky | 0.20% | 0.17% |
| mountain | 0.17% | 0.15% |
| sure thing | 0.11% | 0.12% |
| snowball / hill | 0.07% | 0.08% |
| stash | 0.07% | 0.06% |
| library | 0.07% | 0.06% |
| garden | 0.06% | 0.06% |
| felt lighter | 0.04% | 0.04% |
| fair share | 0.02% | 0.02% |

## Arith cut by operation

Target 8.90M tokens, available after thinning 7.51M: **below target, nothing cut** (all arith train kept).

| op | available | kept | tokens | full-run share | short of target |
|---|---|---|---|---|---|
| * | 9,323 | 9,323 | 0.94M | 14.9% | - |
| + | 23,821 | 23,821 | 1.86M | 32.9% | - |
| - | 31,194 | 31,194 | 2.40M | 30.0% | - |
| / | 25,867 | 25,867 | 2.32M | 22.2% | - |
