# Pre-registered: heartbeat retest on the fixed embedding (written 2026-11-02 09:56 PST, before any result of it)

**Question:** with the embedding fix and the standard recipe (aura_std), does the heartbeat (the live Kuramoto coherence loss, strength
0.1, target R 0.3) still improve the model? Earlier evidence (heartbeat on vs off, 3 seeds per size) was all on the *unfixed* embedding,
where the 25:1 position/token imbalance dominated. That gap (off − on: +0.06 / +0.11 / +0.12) may not carry over.

**Known when this was written:** no aura_std result at all (its first run, s25 seed 1, was still training). No `aura_std_off` result.

## Arms (200M tokens, the sweep recipe: seq 512, batch 32, lr 6e-4, warmup 500, cosine to 0; v1 val split; seeds 1-3)

| arm | flags | runs |
|---|---|---|
| **on** = `aura_std` | `--heartbeat live --emb-scale sqrt --ffn-activation gelu --linear-init gpt` | **reused** from the aura_std sweep (running now, queued before this was written; same code, recipe and seeds) |
| **off** = `aura_std_off` | the same + `--heartbeat-strength 0` | new: s25, m50 × seeds 1-3 |
| vanilla2 + heartbeat | — | **skipped: not possible.** The heartbeat is defined on the oscillators' live phases (Ω·x + Θ). Linear layers have none, and `--projection linear` returns a zero heartbeat loss. A phase for linear layers would be a new mechanism, not this retest. |

Pairs are by seed (same init seed and data order).

## Decision rule

**The heartbeat stays in AURA only if `on` beats `off` at m50 with a paired t-test p < 0.05 (two-sided, seeds 1-3) and mean(on) < mean(off).**
Otherwise it's removed from the default recipe (it stays in the code as an option and in VISION.md as an idea to retest at larger
scale). s25 is reported with the same statistics (Welch, exact Mann-Whitney, paired t) but doesn't decide anything.

Also reported: the true R reached by each arm (`coh=` in the logs) and tok/s (the heartbeat's cost).

## Queue

`scripts/heartbeat_retest_queue.sh` runs after `scripts/next_queue.sh` (aura_std sweep → ft_v2 curve) finishes cleanly, through
`scripts/after_next_queue.sh`. Cost ≈ 1.7 GPU-h (3 × s25 ≈ 12 min, 3 × m50 ≈ 22 min).
