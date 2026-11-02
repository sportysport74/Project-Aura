# Pre-registered decision rules: vanilla2 vs embfix (written 2026-11-02 04:22 PST, before the m50 / l100 results exist)

Sweep: `experiments/plans/vanilla_embfix.md` (arms `vanilla2` = standard transformer of the same shape; `embfix` = AURA, heartbeat on,
token embeddings × √d_model). s25 / m50 / l100 × seeds 1-3, 200M tokens, same data, recipe and seeds.

**Known when this was written** (only these two rows were in `experiments/results/sweep/results.tsv`): s25 seed 1, vanilla2 val 3.4310
(580k tok/s) and embfix val 3.7404 (339k tok/s). For reference, the earlier heartbeat-on s25 arm: 4.055 / 4.058 / 4.086. No m50 or l100
result for either arm existed. The rules below concern m50 and l100 only.

## Definitions

- Per size and arm: mean ± sd of the final val loss over seeds 1-3 (v1 val split, the sweep's `results.tsv`).
- **seed-sd** at a size = the larger of the two arms' seed standard deviations at that size (n = 3 each).
- **Gap** at a size = mean(embfix) − mean(vanilla2). Positive = AURA worse.
- **Paired test** at l100: paired t-test over seeds 1-3 (same seed = same init seed and data order), two-sided, p < 0.05. Also reported:
  Welch t and exact Mann-Whitney (`experiments/sweep_report.py`).

## Rules (applied in this order)

1. **The gap was the embedding setup → adopt embfix for AURA going forward**, if |gap| ≤ seed-sd at **both** m50 and l100.
2. **Oscillator projections cost quality at this scale**, if embfix is behind vanilla2 with paired p < 0.05 at l100 (gap > 0). Then the
   next designs use a **standard backbone**, with oscillator mechanisms (heartbeat, chirality gate, settling, sleep) added as **modules
   that must each beat vanilla2 on their own test**. The existing VISION.md experiments are re-based onto vanilla2 accordingly.
3. **Oscillator projections help**, if embfix is ahead of vanilla2 with paired p < 0.05 at l100 (gap < 0). Then keep the oscillator
   backbone, with embfix as its default.
4. **Otherwise inconclusive** (e.g. a gap > 1 seed-sd at one size but p ≥ 0.05 at l100, or the sizes disagree): run seeds 4-6 of both
   arms at l100 (≈ 2.5 GPU-h) and apply rules 1-3 to n = 6. No design decision before then.

Rules 1 and 2 can both hold (a gap inside one seed-sd at m50 and l100 that's still significant at l100, if the seeds are very consistent).
In that case rule 2 decides, because a consistent quality cost matters more than its size relative to the seed noise.

## Compute report (always produced, whichever rule fires)

Per arm and size:
- **median tok/s** (steps 1000-12207, from each run's log; already in `results.tsv`),
- **GPU-hours per run** = 200M tokens / tok/s (and the measured wall time),
- **val loss mean ± sd**,
- **quality per GPU-hour**: (val loss of the s25 heartbeat-off AURA arm − this arm's val loss) / GPU-hours. That's the loss improvement over a
  fixed weak reference per GPU-hour spent. Higher is better.
- **iso-compute view**: for each embfix size, the largest vanilla2 size whose GPU-hours per run are ≤ embfix's, with both val losses. That
  answers "for the same GPU time, which family gets lower loss?"

Caveat to report with any rule-2 outcome: both families use the AURA-tuned lr (6e-4). If rule 2 fires, run a short lr check for vanilla2
(3e-4 / 1e-3 at s25) and mention whether vanilla2 could be even better. That can only strengthen rule 2, never reverse it.

---

## Addendum 1 (2026-11-02 04:30 PST, before any m50 / l100 result; user decision)

**Confound:** vanilla2 differs from embfix in more than the projections: vanilla2 also has GELU and GPT init (pre-LN is already shared:
both use `x + f(LN(x))` and a final LayerNorm). So vanilla2 vs embfix measures oscillators + recipe, not oscillators alone.

**New arm `aura_std`** = AURA (oscillator projections, heartbeat on) with the exact vanilla2 recipe: `--emb-scale sqrt --ffn-activation gelu
--linear-init gpt`. GPT init applies to nn.Linear layers. Its residual rule (layers that write into the residual stream, attention-out
and ff2, start 1/√(2·n_layers) smaller) is applied to those oscillator layers' amplitudes, the oscillator counterpart of the output weight
scale. **Only the projections differ from vanilla2.** Queued after the current sweep: s25 / m50 / l100 × seeds 1-3.

**What each comparison measures (this replaces "embfix" in rules 1-4 above):**
- **Oscillator verdict = `aura_std` vs `vanilla2`.** Rules 1-4 apply with aura_std in embfix's place. Rule 1 ("adopt for AURA going
  forward") then means: adopt the vanilla2 recipe (embed fix + GELU + init) for AURA.
- **Recipe effect = `embfix` vs `aura_std`** (both oscillator, both embed fix; the difference is GELU + init). Reported with the same
  statistics, no decision attached.
- embfix vs vanilla2 is still reported (oscillators + recipe together), but it no longer decides anything.

**Equal GPU-hours, not only equal tokens:** reported for every pair:
1. **iso-compute across sizes** (free): for each aura_std size, the vanilla2 size with ≤ the same measured GPU time per run, with both
   val losses (e.g. if vanilla2 l100 takes ≤ aura_std m50's time, compare those two).
2. **matched-budget runs** (optional, needs the user's OK): vanilla2 retrained at each size for the token budget it reaches in aura_std's
   GPU time (≈ 1.7-2× the tokens at its measured speed), cosine decayed to that budget, 3 seeds. That's the only clean equal-time comparison:
   comparing a mid-schedule curve point with an annealed endpoint would be biased. Cost ≈ aura_std's GPU time (≈ 4 h).

Known at the time of this addendum: s25 vanilla2 seeds 1-2 (3.4310, 3.4308), s25 embfix seed 1 (3.7404). Nothing at m50 / l100, and
nothing for aura_std.
