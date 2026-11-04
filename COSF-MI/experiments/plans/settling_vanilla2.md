# Pre-registered: criticality-controlled settling on the vanilla2 backbone (written 2026-11-03, before any run of it)

VISION.md §3b ("deliberation by settling"), rebased on the standard backbone after the oscillator verdict (RESULTS.md 2026-11-03).
Known when this was written: no result of any settling arm; not even a smoke test had run.

## Module (`src/python/cosf_mi/models/settling.py`, unit tests `tests/test_settling.py`: 7 pass)

After each block's FFN, per token (no cross-token terms, so causal by construction):
`φ⁰ = W_in·LN(x)`, then T = 8 weight-tied steps `φ ← φ + γ[K Σ_j J_ij sin(φ_j − φ_i) + β sin(φ⁰ − φ)]`, readout `W_out·[cos φ; sin φ]`
added to the residual stream. N = d_model oscillators (384 at s25), J = U diag(s) Uᵀ symmetric, rank 64, learned. γ = 0.25, β = 0.5.
K is a fixed constant (1.0), not learned. Phases in fp32. W_out starts at N(0, 0.02/√(2·n_layers)), so the module starts near a no-op.
Tests: K = 0 equals a plain single-pass block; one R4 repeat equals K = 0 settling; J symmetric and equal to the Σ definition; the clamp
divides by exactly φ or 2; no future-token leakage (settle and repeat); diagnostics and the controller run.

**Diagnostics**, logged every 200 steps to `settle_diag.jsonl`, per settling step t = 1..8 and per layer: the Kuramoto order R (mean over
tokens), var(R) across the batch's tokens, and a finite-difference Lyapunov estimate λ_t = ln(‖δ_t‖/‖δ_0‖)/t from a 1e-4 perturbation of
φ⁰ (4-sequence sub-batch, no grad). Also K per layer and the clamp fraction.

## Arms (s25 = 23.5M backbone, 200M tokens, the sweep recipe at lr 6e-4 = the l100 check's winner, seq 512, batch 32, warmup 500, cosine to 0)

| arm | flags (on top of vanilla2: `--projection linear --emb-scale sqrt --ffn-activation gelu --linear-init gpt`) |
|---|---|
| a. vanilla2 | none. **Reused**: the existing 3 seeds (3.4310 / 3.4308 / 3.4315, sd 0.0004) |
| b. R4 control | `--settle repeat --settle-repeats N`: the same block applied N times with no dynamics. N is set in the smoke test as the value whose measured tok/s is closest to arm c's (compute-matched by measured speed, recorded below before any 200M-token run) |
| c. settling, fixed K | `--settle settle --settle-K 1.0` |
| d. + clamp ÷ φ | c + `--settle-clamp phi` (any per-step |Δφ_i| > 0.25 is divided by φ) |
| e. + clamp ÷ 2 | c + `--settle-clamp two` |
| f. + criticality controller | c + `--settle-controller auto`: between steps, K ← K·exp(−0.05·λ̄) (λ̄ = EMA of the last-step Lyapunov estimate). If at step 500 the raw estimate is too noisy (sd > 2|mean| and |mean| < sd over steps 250-500), it switches to keeping R in [0.3, 0.6]. The signal used is logged and reported |

## Procedure (exploration mode)

1. **Seed 1 of every arm** (b-f; a exists).
2. **3 seeds** (add seeds 2, 3) only for an arm that beats its comparison by more than the **seed sd, fixed at 0.005** (the largest seed sd
   observed at s25 on the fixed embedding: embfix 0.0049; vanilla2's own 0.0004 is too small to stand for the settling arms). Comparisons:
   b vs a, c vs b, f vs b, f vs c, d vs e.
3. Final claims need 3 seeds per arm involved and a paired t-test (by seed), p < 0.05.

## Decision rules

- **Settling counts only if c or f beats b** (the compute-matched control), not just a.
- **The φ question is d vs e**: "φ is special" only if d beats e (p < 0.05). If they're within noise: no evidence φ matters. If the clamp
  never triggers (clamp fraction ≈ 0), the question is untestable at these settings, and that's reported as such.
- **The controller counts only if f beats c.**
- b vs a is reported (does extra compute in this form help at all), but it decides nothing.
- Also reported per arm: tok/s, GPU-minutes per 200M tokens, final R / var(R) / λ per layer, and the controller's chosen signal and K trajectory.

## Settling inside the new base

The new-base plan (`experiments/plans/vanilla2_base_10b.md`) includes settling **only if** c or f beats b under the rules above.
