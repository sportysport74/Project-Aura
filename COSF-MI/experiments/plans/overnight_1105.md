# Pre-registered: probe confound check + l100 lr check at 3e-4 (written 2026-11-05, before any run of either)

## 1. Probe confound check

**Question:** on the new base the knowledge probe is flat (2.2B 0.447, 5B 0.434, 10B 0.448). The 2.2B point is the only one annealed from
the 6e-4 peak with a cosine; the 5B/10B points come from the 3e-4 WSD continuation. Is the flat curve the schedule, or saturation?

**Run:** from `checkpoints/v2base/main/stable_step140380.pt` (the 4.5B stable state; WSD constant at 3e-4), continue on the extension corpus
in the same data order (the batches after step 140,380, unseen), with phase A's schedule re-applied: warmup 2,000 steps to **6e-4**, then
cosine to 0 over **134,266 steps (2.2B tokens, phase A's length)**, keeping the optimizer state (`--resume-state ... --branch-cosine-steps
134266 --branch-start-step 140380 --lr 6e-4 --warmup 2000`). Endpoint: 6.7B tokens total. Output `checkpoints/v2base/anneal45_cos`.
Evals: v1 val loss; knowledge probe (`knowledge_pmi.py`) on the anneal point with the 2.2B, 5B and 10B points re-scored in the same run;
paired bootstrap over the 100 questions (`knowledge_compare.py`).

**Rule (the user's, written before running):** **if the anneal point beats the 10B point's probe** (anneal − 10B > 0 with the 95% CI
excluding 0), **the flat curve was the schedule, not saturation.** Otherwise (CI includes 0, or anneal < 10B) the schedule doesn't explain it,
which is consistent with saturation of the probe at this size. Also reported, deciding nothing: anneal vs 2.2B, and the val loss.
Caveat: one run; CIs are over questions, not seeds.

## 2. l100 lr check at 3e-4

Same protocol as 2026-11-03 (`scripts/lr_check_l100.sh`, `LRS=3e-4`): vanilla2 l100, 3,000 steps, warmup 300, cosine to 0, seed 1, full v1 val.
Compared with the existing 6e-4 run (3.7687). **Rule:** 3e-4 counts as better only if its val is lower by more than **0.01** (the noise of
a single compiled 3,000-step run isn't measured; 0.01 is ~3× the largest seed sd seen for l100 vanilla2 at 200M tokens, 0.0013). If it's
better, the base's lr was too high, and that's recorded for the next base; nothing is rerun automatically.

## Queue and safety

`scripts/overnight_1105.sh`: anneal (~3.3 GPU-h) → probe (~5 min) → lr check (~6 min). Every phase smoke-tested first (`SMOKE=1`: 20-step anneal
from the real stable state, the probe on the smoke checkpoint, a 20-step lr check). Through run_train (watchdog, GPU telemetry, pause), 450 W
check, any stop writes `run_control/FAILED`. **No auto-resume** (not authorized for this run). Alerts: Claude watches the chain and sends a
push notification when it ends or fails.
