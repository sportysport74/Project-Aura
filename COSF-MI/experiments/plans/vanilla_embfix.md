# Plan: fixed vanilla baseline + AURA embedding fix (approved 2026-11-01; runs automatically after 10B stage 2)

## Why

The first vanilla baseline (`--projection linear`, 2026-10-31) **failed to train**: every size stayed on the unigram plateau (~7.4) for
thousands of steps. At init, the positional encoding (std ~0.5) outweighs the std-0.02 token embedding 25:1 in both models, and linear
q/k give attention logits 14× larger than the bounded oscillators. That run is a broken recipe, not a baseline (its rows in
`experiments/results/sweep/results.tsv` are arm `vanilla`: ignore them).

## Arms (s25 / m50 / l100 × seeds 1-3, 200M tokens, the sweep recipe: lr 6e-4, warmup 500, cosine to 0, seq 512, batch 32)

| arm | flags | question |
|---|---|---|
| `vanilla2` | `--projection linear --emb-scale sqrt --ffn-activation gelu --linear-init gpt` | a standard transformer of the same shape: do the oscillators beat it? |
| `embfix` | `--heartbeat live --emb-scale sqrt` | does fixing the 25:1 position/token imbalance help AURA itself? |

Compared with the sweep's existing arms, paired by seed (same init seed and data order): `vanilla2 − on`, `embfix − on`,
`vanilla2 − embfix` (Welch + paired t, `experiments/sweep_report.py`).

- `--emb-scale sqrt`: token embeddings × √d_model before the positional encoding (0.02·√384 = 0.39 at s25; the encoding ≈ 0.5).
- `--linear-init gpt`: N(0, 0.02) for every plain projection, zero bias, the attention-out and ff2 layers scaled by 1/√(2·n_layers).
- `--ffn-activation gelu` replaces the block's tanh. That's a second difference from AURA, kept on purpose: the baseline should be a
  standard transformer, not AURA-minus-oscillators. (`embfix` keeps tanh.)

## Gate (fixed before running)

A 2,000-step s25 `vanilla2` diagnostic must reach loss ≤ 6.0 by step 2000 (heartbeat-on s25: 5.45 on all 3 seeds; broken vanilla:
7.44). If it fails, the vanilla2 arms are skipped and reported, and only `embfix` runs. `scripts/post_stage2.sh`.

## Cost (450 W cap)

vanilla2 ≈ 2 h for 9 runs (the old vanilla's speed), embfix ≈ 3.7 h for 9 runs (the heartbeat arm's speed): ≈ 6 GPU-h.

## Caveat for RESULTS.md

A single learning rate (6e-4) for both families. If vanilla2 loses, a quick lr check (3e-4 / 1e-3 at s25) is needed before calling it,
because the oscillator recipe was the one we tuned.
