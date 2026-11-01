# Plan: chirality A/B at m50 (PLANNED 2026-10-31, NOT RUN; after the current queue: 10B stage 1, then the vanilla baseline)

Question: can AURA learn handedness at all, and does input-dependent handedness help? See `experiments/VISION.md` §3.

## What we know going into it

1. **Signs are frozen at init** in the trained l100 (99.95% kept; `experiments/results/chirality/l100_on_s1.md`). χ multiplies the
   frequencies, so crossing zero kills the oscillator's driven phase, a barrier gradients don't cross.
2. **The shrinkage is mostly weight decay**: AdamW decay alone predicts |χ| = 0.672, and the observed median is 0.655.
3. **Without the per-token gate, χ adds no expressive power** (checked numerically 2026-10-31, max difference 1e-6). A chirality
   layer `A·(cos θ + χ·sin θ) + k·cos θ` with `θ = χ·(Ω·x) + Θ` equals a plain oscillator layer `A''·sin θ'' + k·cos θ''` with
   `Ω' = χΩ`, `Θ'' = Θ − δ + δ''` and `A'' = √(r² − k²)` (where `r·cos(θ − δ)` is the chirality layer's single sinusoid). That holds
   whenever r ≥ k = 0.1·φ ≈ 0.16, which every trained oscillator meets. So per-feature chirality only changes **training dynamics**: the
   zero barrier, weight decay, and the heartbeat, which measures R on θ and therefore sees per-feature offsets δ. That's the same kind of
   offset as the `live_control` arm, a link worth noting.
4. **With the per-token gate it's real:** `θ = g(x)·χ·(Ω·x) + Θ` and the sine term scaled by g(x), with g = tanh(W·x + b) per group of
   features. The direction of the driven phase now depends on the input, and that can't be absorbed into fixed parameters.

So B and C can only show optimization effects. **D is the actual test of learned handedness.** Arm E (gate on the current
parameterization) separates "the gate helps" from "the angle parameterization helps".

## Arms (m50, 50.2M params, 200M tokens, 3 seeds each, heartbeat on)

| arm | handedness | weight decay on it | gate | what it tests |
|---|---|---|---|---|
| A | χ = ±1 init, multiplies Ω and the sine term (today) | yes (0.01) | off | baseline |
| B | same as A | **no** | off | is the shrinkage decay? (expect |χ| to stay ~1 if so) |
| C | **angle β**: `θ = Ω·x + Θ`, output `A·√2·cos(θ − β) + k·cos θ`; β init ±π/4 (gives A's `cos ± sin` at init); β passes through 0 (pure cos) with nothing vanishing | no | off | can handedness move freely? (function class same as A) |
| D | C + gate: `θ = g·(Ω·x) + Θ`, output `A·√2·cos(θ − g·β) + k·cos θ` | no | **on** (groups 8, bias 1.0) | input-dependent handedness |
| E *(recommended extra)* | A + gate (the existing `chirality_context_groups` code) | no | on | the gate without the new parameterization |

Same recipe as the sweep: `data/corpus_v1/tok24k_v2`, seq 512, batch 32, lr 6e-4, warmup 500, cosine to 0 at 12,207 steps, val on the
whole v1 val split, `--save-final`. Paired by seed (the data order is fixed per epoch). Arms interleaved within each seed.

## Measurements

1. **Val loss**: per arm mean ± sd, Welch t and paired-by-seed t for D−C, E−A, B−A, C−B (`experiments/sweep_report.py` with new arms).
   Expectation from point 3: A ≈ B ≈ C within noise. Any gap there is an optimization effect, not capacity.
2. **Direction-sensitive benchmark** (new, to build: `experiments/direction_bench.py`). Each base gets the same short fine-tune probe
   (2k steps on synthetic data, licence: ours), then a held-out test:
   - **reversal**: "reverse: k l m n p →" with 4-8 random tokens, exact match;
   - **inverse relations** (reversal-curse style): train "N1 is the mentor of N2" for invented names, test forward queries in new
     phrasings and **inverse** queries ("Who is N2's mentor?" / "Whose mentor is N1?"), accuracy forward vs inverse;
   - **directed order**: shuffled chains "a > b, b > c, c > d", queries "is a > c?" and "is c > a?", against a symmetric control relation
     ("a is next to b") with the same surface form.
   Score: directional accuracy minus symmetric-control accuracy, per arm. **Pilot first** on the existing `checkpoints/sweep/m50_on_s1`
   and on a vanilla m50 (from the queued baseline) to make sure the tasks are neither at floor nor at ceiling for m50.
3. **Does chirality move?** (`experiments/chirality_drift.py`, extended): B: |χ| distribution and sign flips vs init; C/D: β distribution,
   |β − β₀|, and the fraction that crossed 0 (switched handedness); D/E: the gate on 512 val sequences: mean g per layer and group,
   fraction of tokens with g < 0 (direction reversed), variance across tokens. If g stays ≈ tanh(1.0) = 0.76 for every token, the gate
   isn't being used.

## Decision rules (fixed before the run)

- **Learned handedness exists** if in D (or E) the gate varies across tokens (more than 5% of tokens with g < 0 in some layers) **and**
  D beats C on the direction benchmark by more than the probe's seed spread, paired by seed.
- **It helps language modelling** only if D − C (or E − A) val loss is negative with paired p < 0.05 over 3 seeds. Otherwise report it as
  unresolved at n = 3, not as "no effect".
- If B's |χ| stays ~1 while A's shrinks, the shrinkage was decay (expected), and chirality should be excluded from decay from then on.

## Cost (450 W cap; m50 ≈ 150k tok/s measured-ish, the gate adds maybe 5-10%)

| item | time |
|---|---|
| code: `--chirality-wd`, `--chirality-param {sign,angle}`, gate flags in `train_overnight.py`; arms in `sweep_queue.sh`; tests that angle at β = ±π/4 equals sign at χ = ±1 for the same Ω row signs | ~half a day, CPU |
| direction benchmark + probe script + pilot on 2 existing m50 checkpoints | ~half a day + ~30 min GPU |
| 12 runs (A-D) × ~23 min, or 15 with E | **4.6 h** (5.8 h with E) |
| probes + benchmark, 12-15 bases × ~6 min | 1.2-1.5 h |
| **total GPU** | **≈ 6-7.5 h** (one night) |

Disk: m50 final weights 0.2 GB × 15 + full states 0.6 GB × 15 ≈ 12 GB; delete the full states after the run.
