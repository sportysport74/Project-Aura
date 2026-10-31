# l100 control seeds 4-6: batch effect or control failure? (2026-10-30)

Seeds 4-6 of l100 `on` vs `control` widened control − on from +0.029 (n=3) to +0.068 (n=6). Almost all of that came from
`control`: its seeds 4-6 average 0.084 worse than seeds 1-3 (Welch p 0.08), while `on`'s seeds 4-6 are within 0.007 of
seeds 1-3 (p 0.48). Control's sd over 6 seeds is 0.056, against 0.011 for `on`.

## Same setup in both batches

| | seeds 1-3 | seeds 4-6 |
|---|---|---|
| launch | runs 20261028-172126 / 20261029-042725 | run 20261029-204558 |
| commit | 6f290f3 | 28cdd6f (adds `--sleep`, default on, never fires before step 45k; queue loop overrides) |
| training command | `scripts/sweep_queue.sh`, identical flags | same (`SIZES=l100 SEEDS="4 5 6" ARMS="on control"`) |
| log header | identical (md5 of the non-step lines matches, control s1 vs s4) | |
| torch | 2.14.1+cu132 | same |
| NVIDIA driver | 617.42 (32.0.16.1742) | **same**: Windows has been up since the 10/29 01:50 crash and the driver files date from 10/9 |
| other GPU load | Wallpaper Engine running | closed (both arms ~5% faster) |

So no code, config or driver change between the batches. The only known difference is Wallpaper Engine, which changes
speed, not the maths. The driver was **not** reinstalled between the batches.

## Does the control loss reach its target at l100?

Yes, in every seed. `experiments/hb_probe.py` on the final checkpoints, 20 val batches (`hb_probe.tsv`):

| run | R_ψ (control target 0.3) | true R (`on` target 0.3) | drift gate |
|---|---|---|---|
| control s1-s6 | 0.3054-0.3071 | 0.0023-0.0038 | 0.957 |
| on s1-s6 | 0.0024-0.0037 | 0.3064-0.3078 | 0.956 |
| off s1 | 0.0027 | 0.0031 | 0.953 |

Both losses are met and switch off (logged sync loss 0.00000 from step ~2k in all 12 runs). The gate is open, so a
zero sync loss means the target is met, not that the loss is gated off.

## Spikes

The loss "spikes" in the logs (steps 9600, 11000, 11400, 12200; +0.3-0.4) appear at the same steps in every arm and seed.
The data order is fixed, so these are hard batches, not instability. The only seed-specific anomaly is control s6's pre-clip
grad norm of 4.69 at step 800 (other runs peak at 0.65-1.88). Control s6 is also the worst seed (3.923).

## Reading

- No batch effect found in code, config, driver or torch. The control loss works as designed at l100.
- Control is simply more seed-sensitive than `on`. Seeds 4-6 happened to land worse, and s6 had an early gradient
  spike. `on` is tight across all 6 seeds.
- So "+0.068, p 0.023" overstates how cleanly synchrony beats the control. The robust statements are: `on` is
  never worse than control in a paired seed (6/6), and `on` is ~5× more stable across inits.
- Cheapest check not yet run (no more seeds, as asked): rerun control s1 bit-for-bit. If it reproduces 3.7809, the batches
  are numerically identical and seeds 4-6 are plain variance.
