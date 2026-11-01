# Plan: deliberation by settling (PLANNED 2026-10-31, NOT RUN; after the current queue)

Question (`experiments/VISION.md` §3b): if each block lets its oscillators settle through iterated Kuramoto dynamics (AKOrN-style),
does more settling help on problems the model can't do in one pass (multi-digit ×, multi-step arithmetic), and do harder problems
settle more slowly on their own?

## Mechanism (new code; nothing like it exists today)

Inside each block, after the oscillator layer computes its driven phases `θ⁰ = Ω·x + Θ` (per token, per oscillator), run K weight-tied
update steps before producing the output:

```
θ^{k+1} = θ^k + γ · [ c · Σ_j J_ij sin(θ^k_j − θ^k_i) + β · sin(θ⁰_i − θ^k_i) ]
output  = A · (cos θ^K + χ sin θ^K) + k · cos θ^K          (the existing read-out)
```

- **Coupling J** between oscillators of the same token: learned, low rank (J = U Vᵀ, rank 64) to keep parameters and compute small.
  Σ_j J_ij sin(θ_j − θ_i) is computed as Im(e^{−iθ_i} · Σ_j J_ij e^{iθ_j}), so it's two rank-64 matmuls, not N².
- **Anchor term** β·sin(θ⁰ − θ): keeps the input in play while settling, like AKOrN's input drive.
- γ, c, β learned scalars per layer. **Weight-tied across the K steps**, so K changes compute, not parameters.
- Applied to ff1 (the widest oscillator layer, d_ff) of every block first. Extending it to the attention projections is a follow-up.
- **Adaptive halting:** per token, stop updating once mean_i |θ^{k+1}_i − θ^k_i| < ε (ε a fixed hyperparameter, two values tried in the
  pilot), up to K_max = 16. Halted tokens are masked, and the batch runs until all have halted or K_max is reached. Steps used are logged
  per token (training: a histogram; eval: per answer token).
- Unit tests before any GPU time: K = 0 equals today's layer bit for bit; with J = 0 and β = 0 the update is a no-op; the gradient through
  K steps matches finite differences; halting masks behave.

## Arms (m50, 200M tokens, 3 seeds, heartbeat on unless noted; same recipe and data as the sweep)

| arm | settling | why |
|---|---|---|
| vanilla | none (`--projection linear`) | from the queued baseline (reused, no new runs) |
| AURA today | none (single pass) | from the sweep (`m50_on_s1..3`, reused) |
| S1 | K = 1 | the new coupling with no deliberation |
| S4 | K = 4 | |
| S8 | K = 8 | |
| SA | adaptive, ε from the pilot, K_max 16 | does compute adapt to difficulty? |
| **R4 (compute control)** | today's block applied **4× weight-tied** (Universal-Transformer style), no Kuramoto coupling | separates "settling dynamics" from "just more compute per token" |

**Why R4 matters:** without it, a win for S4/S8 can't be told apart from "more FLOPs per token help". The claim is about settling, so it
needs a control with the same extra compute and no Kuramoto dynamics.

## Tests

1. **Val loss** on the v1 val split (paired by seed). It's a secondary measure: settling is meant for reasoning, not perplexity.
2. **Arithmetic probe**, the same for every base: fine-tune on 200k plain-format problems (`build_arith_plain.py`, mix × 40 / ÷ 20 /
   multi-step 30 / + − 10; multi-step = 2-4 operations with precedence, e.g. `17 + 6 × 23 − 4`), 1 epoch, then score with
   `arith_md_eval.py`-style cells:
   - × by digit cell (1x1 to 3x3), **fit** (trained problems) and **held** (never generated), so capacity is separated from generalization
     like in the 10-28 test, where 33M plain couldn't even fit 2x2 ×;
   - multi-step by number of operations (1-4).
   **Pass:** S4/S8/SA beat both R4 and AURA-today on held-out multi-digit × or on 3-4-step problems by more than the seed spread
   (paired), and beat vanilla.
3. **Settling at test time:** evaluate S4 at K = 1, 2, 4, 8, 16 (and S8 likewise) on the same problems without retraining. **Pass:**
   accuracy rises with K beyond the trained K, at least up to some point. That's the "thinking longer helps" signature.
4. **Difficulty → steps (SA):** Spearman correlation between the steps used on the answer tokens and difficulty (digits, number of
   operations), and between steps and correctness. **Pass:** harder problems use more steps (ρ > 0, CI excluding 0). Also check whether
   the model is *wrong* more often when it settles slowly, as a candidate confidence signal, to be calibrated, not trusted (synchrony
   isn't truth).
5. **What settled states look like:** global R and cluster structure of the settled phases (number of phase clusters per token, R within
   vs between clusters) on 512 val sequences. If settling just drives R → 1, it's carrying no information, and that counts as a failure
   mode, not success.

## GPU-hour estimate (450 W cap; m50 single pass ≈ 150k tok/s ≈ 22 min per 200M tokens)

Cost per arm is scaled from the oscillator part (≈60% of step time) plus the rank-64 coupling. These are estimates (±30%), to be replaced by a
60-step probe per arm.

| arm | × time | per run | × 3 seeds |
|---|---|---|---|
| S1 | ~1.3× | ~29 min | 1.5 h |
| S4 | ~2.2× | ~48 min | 2.4 h |
| S8 | ~3.5× | ~77 min | 3.9 h |
| SA (K_max 16, assume ~5 steps on average) | ~2.5× | ~55 min | 2.8 h |
| R4 | ~4× | ~88 min | 4.4 h |
| **training subtotal** | | | **≈ 15 h** |
| arithmetic probes: 15 new + 6 reused bases × ~8-15 min | | | ~3.5 h |
| test-time K sweep (S4, S8 × 5 values × 3 seeds) + steps/difficulty + phase structure | | | ~1.5 h |
| pilot (1 seed of S4 and SA, ε ∈ {1e-2, 1e-3}, short runs) | | | ~2 h |
| **total** | | | **≈ 22 GPU-h** (two to three nights) |

- **Memory:** backprop through K steps keeps K copies of the ff1 phases. m50 single pass peaks at 11.3 GiB, and S8 at batch 32 would
  likely exceed 32 GiB, so use activation checkpointing per settling step (recompute in backward, +~30% time for S8/SA, already counted
  above) or batch 16 × gradient accumulation 2. The probe decides.
- **Cheaper first cut** (recommended): the pilot plus 1 seed of every arm (≈ 9 GPU-h). Run the full 3 seeds only if S4/S8/SA show
  anything against R4 at 1 seed. A null at 1 seed is weak evidence, so call it a pilot result, not a conclusion.

## Code needed (~1 day, CPU)

`SettlingOscillatorLayer` (or a `settle_steps` option on `COSFOscillatorLayer`) with low-rank coupling, anchor, halting and step logging;
`--settle-k`, `--settle-adaptive --settle-eps --settle-kmax`, `--block-repeat N` (R4) in `train_overnight.py`; a multi-step generator in
`build_arith_plain.py`; eval scripts for the K sweep, steps vs difficulty and phase clustering; unit tests (above); arms in `sweep_queue.sh`.
