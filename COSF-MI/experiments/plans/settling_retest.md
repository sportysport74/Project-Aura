# Pre-registered: settling retest on multi-step arithmetic (approved by Jon and frozen 2026-11-08, before any run of it)

VISION.md §3b. Retest rule (Jon, 2026-11-04): zero-init readout, judged on a multi-step reasoning task, not on val loss. Jon's
changes (2026-11-08): zero-init for the repeat control too; a deeper-vanilla2 control at matched compute; a floor check that fixes
the test range before the main runs; pass = beat all three controls, plus a pre-registered difficulty trend; halting only if it passes.

Known when this was written: the previous settling test (2026-11-04, `settling_vanilla2.md`, RESULTS.md) made vanilla2 worse on val
loss (+0.035 at s25) at ~3× compute; its R4 repeat control matched compute at 9 repeats. Plain-format × and ÷ at 33M (oscillator, fine-
tuned, 2026-10-28) were near 0 for both operands ≥ 2 digits. No vanilla2 model has been trained on arithmetic alone. No result of any
arm below exists.

## Question

Does within-token Kuramoto settling help a standard transformer do multi-step arithmetic, beyond what the same extra compute buys
without oscillator dynamics?

## Task and data

- **Data:** `scripts/distill/build_arith_plain.py`, plain format ("47 × 38 = 1786", then "Answer: 1786"), mix
  `*:0.45,/:0.45,+:0.05,-:0.05`, operands 1-4 digits (÷ = (a × b) ÷ b), seed 0. Held-out policy unchanged: every `common.heldout()`
  triple (1 in 10) and every eval triple is never generated; non-fact problems appear at most once (small cells run out; the report
  records how many problems each cell got). Written as "You: … Aura: …" transcripts with `</s>` between problems, tokenized with the
  24k tokenizer into `data/arith_settle/{train,val}.bin` (new: `scripts/corpus/build_arith_stream.py`, same format as the corpus shards).
- **Training:** from scratch, arithmetic only, s25 size (d384, L8, 6 heads), seq 512 (packed), batch 32, lr 6e-4, warmup 500, cosine to 0,
  **200M tokens per run** (the sweep budget; problems repeat as epochs if the stream is shorter), bf16 + compile, 450 W.
- **Eval (new: `experiments/settle_arith_eval.py`):** greedy decoding from the prompt "You: a × b = ? Aura:" (same template as training),
  exact match on the number after "Answer:". **All held-out problems of a cell, up to 200** (cell = digits of a × digits of b; for ÷,
  digits of the quotient × digits of the divisor; if a cell has more than 200, 200 are drawn with seed 0), from `common.heldout()`, saved to
  `experiments/results/settle_retest/test_set.jsonl` with **the actual n per cell recorded**, so every arm sees the same problems. **Cells
  with n < 50 are secondary only** (never in the primary set). Facts (1×1) are excluded from the test.

## Step 1: floor check (before any main run; recorded below as Addendum 1 before step 2 starts)

- vanilla2 s25, seeds 1 and 2, as above. Eval every × and ÷ cell (2 ops × 15 non-fact cells).
- **Test range = every cell with n ≥ 50 where vanilla2's mean accuracy over the 2 seeds is in [20%, 60%].** These cells are the primary test set.
  Cells above 60% are reported as "easy", cells below 20% as "hard" (secondary).
- If fewer than 3 cells land in range: change nothing on my own; report to Jon with the full table and stop.

## Step 2: arms (s25 backbone; vanilla2 = `--projection linear --emb-scale sqrt --ffn-activation gelu --linear-init gpt`)

| arm | what it adds | starts as |
|---|---|---|
| A. vanilla2 | nothing (reuses step 1's seeds 1-2, adds seed 3) | — |
| B. settling | `--settle settle --settle-T 8 --settle-K 1.0` (no clamp, no controller) | exact no-op: W_out = 0 (new flag `--settle-zero-init`) |
| C. repeat control | `--settle repeat --settle-repeats N`: the same plain block applied N times, no dynamics | exact no-op: W_out = 0 |
| D. deeper vanilla2 | vanilla2 with more layers (new size entry `s25_deep`, d384, L = L_D) | normal init (GPT rule) |

- **Compute matching (fixed in the smoke test, before any 200M-token run):** 600-step probes at the 450 W cap, median tok/s. N = the repeat
  count whose tok/s is closest to arm B's; L_D = the layer count whose tok/s is closest to arm B's. If two values bracket B, take the one
  with **more** compute (so the controls are never under-matched). Recorded with params and tok/s per arm.
- Parameter counts differ (B and C add ~0.44M per layer; D adds whole blocks). Reported, not matched: the question is compute, as Jon set.
- 3 seeds per arm (1, 2, 3). Same data order per seed across arms.

## Decision rules (fixed before the floor check)

- **Primary metric:** exact-match accuracy pooled over the primary test cells (200 problems × cells × 3 seeds per arm).
- **Statistic (every CI in this plan):** two-level bootstrap, 10,000 resamples: resample the 3 seeds with replacement, then problems
  within each drawn seed (problems paired across arms: every arm sees the same test set, and seed k of one arm is paired with seed k of
  the other), 95% percentile CI of the difference.
- **Pass = B − A, B − C and B − D all have 95% CIs above 0, AND B beats each control on all 3 seeds individually** (seed k of B vs
  seed k of the control, primary-set accuracy). All conditions must hold (an intersection test, so no multiple-comparison correction). If B beats A only: extra compute helps, the oscillators aren't shown to be why. If C or D ties or beats B: same reading.
- **Difficulty trend (pre-registered, secondary; does not change the verdict):** order the primary cells by vanilla2's floor accuracy
  (step 1). Split them into the easier half and the harder half (by that order; the middle cell, if odd, goes to the harder half). Trend =
  (B − best control) on the harder half minus the same on the easier half, paired bootstrap 95% CI. "The gap grows with difficulty" only if
  that CI is above 0. "Best control" = whichever of A, C, D scores highest on the primary set.
- **Also reported:** the "easy" and "hard" cells, + and − cells, val loss on the arithmetic val split, tok/s and GPU-minutes per run, and
  the settling diagnostics (R, var(R), λ per layer and step; how far W_out moved from 0).

## Only if it passes: halting

A follow-up plan (written then, frozen before it runs): adaptive halting (stop settling when the phase change < ε), logging steps per
problem. Question: do harder cells use more settling steps? Not run under this plan.

## Run setup

- One queue script (`scripts/settle_retest_queue.sh`) through `run_train` (watchdog, telemetry, pause flag, `--resume auto`), ntfy alerts
  on any failure and at the end of each step. Step 2 waits for Jon to see Addendum 1 (step 1 ends the queue).
- **Smoke test of every arm** (20 steps, tiny eval of 5 problems per cell, scratch dir) through the same queue with `SMOKE=1`, plus unit
  tests: zero-init makes B and C exact no-ops at step 0 (output identical to vanilla2 with the same weights), `s25_deep` builds.
- Expected GPU time: vanilla2 s25 ~6-10 min per run (588k tok/s on text); B ~19 min; C and D matched to B. Step 1 ≈ 20 min; step 2
  ≈ 10 runs × ~19 min ≈ 3.5 h. Fits before the platform upgrade; runs save every 30 min.

## Licences

Programmatic arithmetic only (licence "ours"). No teacher, no third-party data.

## Changes before freezing (Jon, 2026-11-08)

- Every CI: two-level bootstrap (seeds, then problems within seed). Pass also needs B > each control on all 3 seeds individually.
- Test cells: all held-out problems up to 200 per cell, actual n recorded; cells with n < 50 are secondary only.
- Step 1 ends the queue; Jon reads Addendum 1 before step 2.

## Addendum 0 (2026-11-08, while building, before any run): implementation notes

- **Init bug found and fixed for this plan's arms.** With `--linear-init gpt` (part of vanilla2), the model re-initialized every
  `nn.Linear` inside each block after construction, **including the settling module's**: W_in got std 0.02 instead of 1/√d (so φ⁰ started
  near 0, not at unit-scale phases) and W_out got std 0.02 (not the GPT residual rule). This also applied to the 2026-11-04 settling runs
  (`settling_vanilla2.md`), so their settling started from near-zero phases. With `--settle-zero-init` the loop now skips the settling
  module, so arms B and C keep the module's own init (W_in 1/√d, W_out = 0). The old path (no zero-init) is unchanged, so the 11-04 runs
  stay reproducible. Unit test `test_zero_init_is_exact_noop`: B and C give logits identical to vanilla2 with the same weights at init.
- Arm D uses `--n-layers L_D` on the s25 size instead of a separate `s25_deep` size entry (same model).
- Data: `data/arith_settle/src` (build_arith_plain.py, n = 4,000,000, mix ×0.45 ÷0.45 +0.05 −0.05, seed 0) →
  `data/arith_settle/tok24k`: 185.4M train tokens, 4.0M problems; val 0.09M tokens, 2,000 problems. 200M tokens per run = 12,207 steps
  (32 × 512), so the stream is seen ~1.08 times (2 epochs, cut at the step budget).
- Eval prompt: "a × b = ?" (one of the four training templates), transcript encoded in one go ("You: … \nAura:"), greedy, max 40 new tokens.
- Unrelated pre-existing test failures: `tests/test_constants.py` (import error) and `test_phi_decomposition::test_assign_shells`, both
  from the initial commit; nothing in this plan touches them.

## Addendum 1 (2026-11-08 ~08:50): step 1 floor check and compute matching, before any step-2 run

Run `runs/20261108-080610` (08:06-08:21, no faults). vanilla2 s25, arithmetic only, 200M tokens, seeds 1-2: val loss 0.5105 / 0.5101.
Full table: `experiments/results/settle_retest/floor.md`.

- **Primary cells (n ≥ 50, mean in [20%, 60%]): 3, the minimum the plan allows to go on.**

  | cell | n | seed 1 | seed 2 | mean |
  |---|---|---|---|---|
  | ÷ 3x2 | 200 | 53.0% | 62.5% | 57.8% |
  | × 2x2 | 200 | 49.5% | 49.0% | 49.2% |
  | ÷ 4x1 | 200 | 41.0% | 55.5% | 48.3% |

  Primary set = 600 problems. **Difficulty order (frozen rule): easier half = ÷ 3x2; harder half = × 2x2, ÷ 4x1.**
- Just outside the range: ÷ 3x3 60.5%, ÷ 3x4 62.3% (easy); ÷ 4x2 17.8%, × 2x3 17.0% (hard). Easy: × with a 1-digit operand (74-99%),
  ÷ with a 1- or 2-digit quotient (96-100%). Hard: × with both operands ≥ 2 digits beyond 2x2 (1-17%), ÷ with a 4-digit quotient and a
  ≥ 2-digit divisor (13-18%). Secondary + / −: + 2x2 98%, + 3x3 92%, − 2x2 46%, − 3x3 33% (+ and − are 5% of the mix each).
- **Seed spread is large per cell:** up to 14.5 points (÷ 4x1: 41.0 vs 55.5; ÷ 3x4: 55.5 vs 69.0). The seed level of the two-level
  bootstrap will be wide, and the all-3-seeds rule is demanding.
- **Compute matching** (600-step probes at 450 W, median tok/s after warm-up; scratch dirs, nothing evaluated):

  | arm | tok/s | params |
  |---|---|---|
  | A vanilla2 L8 | 655.8k | 23.4M |
  | **B settling T = 8** | **190.8k** | 27.2M |
  | C repeat ×6 / ×7 / **×8** / ×9 / ×10 / ×12 | 226.6k / 196.6k / **188.4k** / 174.4k / 165.6k / 141.9k | 27.2M |
  | D vanilla2 L20 / L24 / L25 / L26 / **L27** / L28 / L32 | 244.4k / 198.2k / 202.0k / 181.1k / **189.2k** / 173.9k / 143.9k | 57.1M at L27 |

  **N = 8** (closest to B, 1.3% slower, so slightly more compute); **L_D = 27** (closest, 0.8% slower). The probes aren't monotone in L
  (L26 slower than L27), so the measurement noise is about ±5%, and the matching is good to that. **D has 2.1× B's parameters** (57.1M vs
  27.2M): compute is matched, parameters are not, as the plan says.
- Step 2 as frozen: A seed 3, then B (`--settle settle --settle-T 8 --settle-K 1.0 --settle-zero-init`), C (`--settle repeat
  --settle-repeats 8 --settle-zero-init`), D (`--n-layers 27`), seeds 1-3 each: 10 runs, ~19 min each for B/C/D ≈ 3 h. Waiting for Jon.
