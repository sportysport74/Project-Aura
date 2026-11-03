# Plan: the new AURA base = vanilla2 l100 on the same 10B tokens, then ft_v2 on top (PLANNED 2026-11-03, NOT RUN; needs approval)

Why: the oscillator verdict (RESULTS.md 2026-11-03, rule 2) says the next designs use the standard backbone. This plan builds that
base with the **exact protocol of the old 10B curve**, so the two curves compare point for point: same corpora, same data order, same
schedule shapes, same evals. Only the model differs: vanilla2 instead of oscillator AURA. (The old curve also had the unfixed embedding,
so the comparison is old AURA vs the new base as a whole, not oscillators alone. The 3-seed sweep already isolates that.)

## Model

`--model-size l100 --projection linear --emb-scale sqrt --ffn-activation gelu --linear-init gpt` (103.5M params; d768, 12 layers, 12 heads,
d_ff 3072, seq 512, 24k tokenizer). No heartbeat (vanilla2 has no phases). Seed 1. Batch 32 × 512 = 16,384 tokens/step.

## Learning rate (from the queued lr check)

The lr check (s25, seed 1: 3e-4 and 1e-3, against 6e-4 with 3 seeds) runs at the end of the current chain. Rule, fixed now:
1. Take the s25 lr with the lowest val loss. If that's 6e-4, use 6e-4 and skip step 2.
2. Otherwise confirm at l100 before trusting it, since the best lr usually drops with width: two 3,000-step l100 runs (cosine to 0 over
   3,000 steps, ≈ 5 min each): the s25 winner vs 6e-4. Use the s25 winner only if it's also lower at l100; else 6e-4.
3. The continuation phase uses half the chosen peak, as the old run did (6e-4 → 3e-4).

## Protocol (identical to the old curve)

| phase | data | schedule | output = eval point |
|---|---|---|---|
| A. base | `data/corpus_v1/tok24k_v2` (2.2B, 1 epoch) | warmup 2000, cosine to 0 | **2.2B** (annealed) |
| B1. main to 4.5B | `data/corpus_v2/tok24k_ext`, `--init-from` A | WSD, peak = ½ lr, warmup 2000, constant | stable state at step 140,380 |
| B2. 5B branch | the same, `--resume-state` B1 | linear decay to 0 over 30,517 steps (0.5B tokens) | **5.0B** (annealed) |
| B3. main to 10B | the same, `--resume auto` | constant, then decay over the last 10% | **10.0B** (annealed) |

Val loss on the v1 val split at every point (`--val-corpus data/corpus_v1/tok24k_v2`). The data order is the same permutation as the old
run's (seeded per epoch), so the two curves see the same tokens in the same order.

**Evals at 2.2B / 5B / 10B:**
1. **knowledge probe** (`experiments/knowledge_pmi.py`): the same 100 questions and method. Reported against the old curve (0.316 / 0.357 /
   0.383) with paired bootstrap CIs, new vs old at each point and new 10B vs new 5B (the same success criterion as before).
2. corpus-format arithmetic battery (`base_arith_eval.py --max-new 160`).
3. v1 val loss (old: 2.877 / 2.774 / 2.680).

**Then ft_v2 on top:** the same fine-tune as the old ft_v2 curve (running now): ft_v2 set, seq 512, batch 32, lr 1e-4, 1 epoch, replay sleep;
seeds 0 and 1 on each of the 3 points → chat-format factuality (both encodings, greedy + 3 sampled, Qwen2.5-7B grader) → pooled paired
comparisons new vs old at each point.

## Safety (the standing rules)

- **Smoke test of every phase before launch** (exact flags): A 20 steps; B1 20 steps with `--init-from`; B2 20 steps from a B1 smoke state;
  B3 20 steps with `--resume auto`; the probe and battery on a smoke checkpoint; one ft_v2 fine-tune for 20 steps; factuality gen + grade on
  one smoke fine-tune (needs a `--limit N` flag on `factuality_gen.py` and `factuality_grade.py` for a dry run; add it). ≈ 20 min of GPU.
- 450 W cap check, crash marker (no auto-restart after a PC crash), stall watchdog, hourly full-state saves, pause/resume, GPU telemetry,
  `run_control/FAILED` on any error. Auto-resume after a watchdog stop only if you approve it for this run, like stage 2.

## Time (450 W cap; vanilla2 l100 measured at 173k tok/s in the sweep)

| item | GPU time |
|---|---|
| lr confirmation at l100 (only if the s25 winner isn't 6e-4) | ~15 min |
| smoke tests | ~20 min |
| A. base, 2.2B tokens | 3.5 h |
| B1. to 4.5B, 2.3B tokens | 3.7 h |
| B2. 5B branch, 0.5B tokens | 0.8 h |
| B3. to 10B, 5.5B tokens | 8.8 h |
| evals at 3 points (probe + battery) | ~0.5 h |
| ft_v2: 6 fine-tunes × ~23 min (half the oscillator model's 48 min at 2× speed) | 2.3 h |
| factuality: generate + grade 6 models | ~0.75 h |
| **total** | **≈ 21 GPU-h** (old oscillator curve: ~27.5 h for the pretrain alone) |

## Disk (C: has 103 GB free on 2026-11-03)

| item | size |
|---|---|
| A: best_model + latest + latest.prev (+ checkpoint_epoch_1) | ~5 GB |
| B main: latest + latest.prev + stable_step140380 + best_model + checkpoint_epoch_1 | ~6.3 GB |
| B2 branch: latest + latest.prev + final_model | ~2.9 GB |
| 6 fine-tunes (weights + config) | ~2.5 GB |
| **total new** | **~17 GB** (data already on disk) |

Optional cleanup before launch, for headroom, needs your OK: the sweep's `latest*.pt` full states (25 GB; `final_model.pt` weights stay) and
the raw parquet shards of corpus_v2 (27 GB; already tokenized).

## Code needed (CPU, ~2 h)

- A queue for the new base, either by parametrizing `scripts/pretrain_10b_queue.sh` (env `MODEL_FLAGS`, `RUN_TAG`, `BASE_LR`; the defaults
  keep the old behaviour) or as a sibling script. Adds phase A (the old curve's base came from `pretrain_queue.sh`).
- `--limit N` for `factuality_gen.py` / `factuality_grade.py` (dry runs).
- A comparison report: the new curve next to the old one (val, probe, battery, chat factuality).
