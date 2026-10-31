# Factuality eval: l100 base and ft_l100 fine-tunes (2026-10-30)

100 explain questions with plain reference answers and key facts (`experiments/factuality/questions.jsonl`, written by Claude
for this eval, 8 topics). One question is also an exact pilot-data prompt ("Why is the sky blue?"). The readable set's 10
explain prompts overlap in topic.

Pipeline: `experiments/factuality_gen.py` (both prompt encodings; greedy plus 3 sampled_rep seeds; 160-token cap), then
`experiments/factuality_grade.py`, graded by Qwen2.5-7B-Instruct (Apache-2.0, the pilot teacher), temperature 0. Score: CORRECT 1,
PARTIAL 0.5, otherwise 0.

## Grader validation

| check | result |
|---|---|
| control: each question's own reference answer | 1.000 |
| control: another question's reference | 0.000 |
| hand check, first-pass grader (20 rows, seed 0, `spot_check_v1_grader.md`) | 12/20 agree. All 8 misses gave PARTIAL to replies that state no key fact ("a plane is a big flat pancake") |
| fix 1 | the grader must quote the key-fact words it found; no quote means INCORRECT |
| fix 2 | the quote must be in the reply (≥80% of its words): the grader had quoted "it's halfway between the poles" and "The Pacific Ocean" for replies that said neither |
| hand check, final grader, the same 20 | 19/20 (miss: CORRECT where I'd say PARTIAL) |
| hand check, final grader, **20 fresh rows** (seed 1, `spot_check_seed1.md`) | **19/20** (miss: PARTIAL for a reply that only names "magnetic field") |

The 38/40 agreement means the grader is fit for use. Both remaining misses are lenient, so the scores are slightly optimistic, if
anything. The first-pass numbers (`summary_v1_grader.md`, base 0.12-0.20, fine-tunes 0.25-0.34) are superseded.

## Results (final grader, `summary.md`)

| model | joined greedy | joined sampled_rep (3 seeds) | pieces greedy | pieces sampled_rep |
|---|---|---|---|---|
| l100 base (2.2B tokens) | 0.030 | 0.063 (0.060, 0.085, 0.045) | 0.030 | 0.058 |
| ft_l100_s0 | 0.130 | 0.147 (0.15, 0.14, 0.15) | 0.105 | 0.140 |
| ft_l100_s1 | 0.100 | 0.142 (0.12, 0.12, 0.185) | 0.100 | 0.127 |

CORRECT verdicts: 0-2% everywhere. Every score is mostly PARTIAL.

## Reading

- **The 104M model knows almost none of these facts.** The base gets about 6%, and that's PARTIAL credit, not right answers. The pretraining
  corpus has this material (FineWeb-Edu, Cosmopedia), but 104M at 2.2B tokens hasn't stored it in a retrievable form.
- **The fine-tune about doubles it, to about 14%,** mostly by producing an on-topic sentence ("birds fly to find food and warmer places")
  where the base rambles. Still no complete correct answers.
- The prompt encoding doesn't matter after fine-tuning. For the base, joined is marginally better.
- The noise floor for later comparisons: sampled_rep seeds of one model differ by up to ±0.03 per seed. The 3-seed pooled mean differs
  by about 0.01 between fine-tune seeds. Compare checkpoints **paired over the same 100 questions** (bootstrap over questions), not by
  eyeballing.
