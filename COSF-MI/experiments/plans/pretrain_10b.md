# Plan: extend the l100 pretrain to 10B tokens (stage 1 DONE 2026-10-31; stage 2 waits for the user)

> **Gate result (2026-10-31): the chat-format factuality gate is invalid for base models.** Leaving the distill transcripts out of the
> extension made "Aura:" mean arithmetic, so the 5B base answers science questions with sums (0.062 → 0.017). A format-neutral PMI probe
> (`experiments/knowledge_pmi.py`) shows knowledge rising: +0.041 [+0.029, +0.054]. Use the PMI probe plus a fine-tuned factuality
> comparison (the same ft_v2 fine-tune on each base) as the gate. Details in CLAUDE.md Setup status.

Question: does knowledge, measured by the factuality eval (`experiments/factuality/`), climb with more pretraining at
104M, or is 104M the ceiling? Eval points: **2.2B** (the existing base, re-scored in every eval run so comparisons are paired),
**5B** and **10B**, each one annealed so they compare fairly.

## Starting point

- `checkpoints/pretrain/l100_on_s1/best_model.pt`: 104M, heartbeat on, 2.2B tokens of `data/corpus_v1/tok24k_v2`, warmup 2000 then
  cosine to 0, v1 val 2.8769. Factuality (joined, sampled_rep, 3 seeds) **0.063**.
- Chinchilla-optimal for 104M is about 2B tokens, so 10B is about 5× past it. Loss will still fall, but slowly. The risk is that facts
  stay flat while loss improves on style; the 5B gate is there for that.

## Design: continue from 2.2B with a WSD schedule (unchanged from the 10-30 plan)

Load the 2.2B weights with a fresh optimizer (`--init-from`), warm up again to a lower peak, then hold the LR constant and decay it at the end
(WSD: warmup-stable-decay). The 5B point is a short decay *branch* off the stable phase; the 10B point is the main run's own decay.
This saves ~7 GPU-h over restarting from scratch, at the price of a small schedule confound: the 2.2B point used a cosine decay, the 5B/10B points WSD.

- Peak LR 3e-4 (half the original 6e-4), linear warmup 2,000 steps, then constant. Main decay: linear to 0 over the last 10% of the
  run (~47.6k steps, 0.78B tokens). Batch 32 × seq 512 = 16,384 tokens/step. Heartbeat on (live), sleep off, seed 1.
- **5B branch:** the main run stops at step **140,380** (2.3B new tokens, 4.5B total) and keeps that full state (`stable_step140380.pt`).
  The branch resumes it and decays linearly to 0 over **30,517 steps (0.5B tokens)**, so it ends at exactly **5.0B total**. The main run
  never sees the branch.
- Loss and forgetting are scored on the **v1 val split** (`--val-corpus data/corpus_v1/tok24k_v2`, 10.96M tokens) at every point.

## Corpus extension: same three sources, new shards only, no new licences

The extension trains only on shards the 2.2B run never opened (v1 used shards FWE 000-001, Cosmopedia 00000-00002, FineMath 00000-00001).
Shard counts below come from v1's measured yield: FineWeb-Edu 0.375B tokens per parquet GB, Cosmopedia 0.244, FineMath 0.609.

| source | licence (as v1) | extension tokens | shards (`scripts/corpus/download_ext.sh`) | raw |
|---|---|---|---|---|
| FineWeb-Edu `sample/10BT` | ODC-By 1.0 (collection; CommonCrawl text keeps its authors' rights) | 4.36B (55%) | 002-008 (7 × 2.0 GB) | 14.3 GB |
| Cosmopedia v2 (`smollm-corpus`) | ODC-By 1.0; text by Mixtral-8x7B-Instruct (Apache-2.0) | 2.38B (30%) | 00003-00012 (10 × 1.2 GB) | 11.8 GB |
| FineMath-4+ | ODC-By 1.0 (collection; CommonCrawl text) | 0.79B (10%) | 00002-00007 (6 × 0.3 GB) | 1.7 GB |
| aura_arith (programmatic) | ours | 0.28B (3.5%) | `build_arith_plain.py --n 6550000 --seed 2` | 1.3 GB jsonl |
| aura_distill | Apache-2.0 teacher outputs | **0, left out** (v1 already has it; keeps fine-tune data unseen) | | |

Each source has about one shard of slack. The builder stops at each budget and logs `SOURCE RAN OUT` if a source runs short.
Not added: Wikipedia (CC BY-SA, share-alike), TinyStories (GPT-generated), anything with GPT/Claude/Gemini outputs. A mix shift toward
Cosmopedia would probably help factuality, but it would confound "more tokens" with "different data": that's a separate experiment.

## Crash safety, pausing, power (new; all tested 2026-10-31)

- **Checkpoint every hour**: `--save-every-minutes 60` writes the full state to `latest.pt`: weights, optimizer, scheduler, data position
  (epoch + batch; the data order is a fixed permutation per epoch) and **RNG states** (torch CPU, CUDA, Python). The write goes to a temp
  file, is fsynced and then renamed, and the previous state is kept as `latest.prev.pt`. A crash loses at most ~1 h of training.
  The full state is also saved at every stop.
- **Resume test** (`scripts/test_resume.sh`, l100 from the 2.2B checkpoint, WSD, SIGKILL at step 70, resumed from the step-50 save):
  - eager fp32 with `--deterministic`: final weights, optimizer state and all 12 logged losses **bit-identical** to the straight run;
  - compiled bf16 at batch 32 (the real config): the resumed run tracks the straight one within 4e-4 loss, the run-to-run noise.
  Bitwise identity needs deterministic kernels: plain eager runs differ by ~1e-6 run to run (atomic adds in the embedding backward;
  bf16 flash attention backward).
- **No automatic restart, ever.** The queue writes `QUEUE_RUNNING` at start and removes it on any normal exit or failure. A PC crash
  (BSOD, power loss) leaves it behind. The next launch then appends what it knows to `checkpoints/pretrain/l100_10b/CRASHES.md` and
  **refuses to start** until it's relaunched with `ACK_CRASH=1` (the user's call). Tested with a simulated marker.
- **450 W check**: the queue and `scripts/resume_train.sh` both stop with exit 3 if `nvidia-smi` doesn't report a 450 W limit. Tested with
  a fake `nvidia-smi` reporting 575 W.
- **Pause for gaming**: `bash scripts/pause_train.sh` saves the full state and exits in ~10 s, without starting the queue's next stage.
  `bash scripts/resume_train.sh` checks the cap and relaunches. Tested: paused at step 220 (11 s, GPU back to 400 MiB), resumed at
  step 220 with RNG restored, finished, and only then ran the next stage. Each resume costs ~1-2 min of compile warmup.
- Stops on the first failure and never retries; a CUDA error is labelled as such.
- Measured at the 450 W cap: **~94k tok/s** (same as the uncapped night pretrain), 17.4 GiB peak.

## Time (at 94k tok/s)

| step | time | GPU |
|---|---|---|
| download ~28 GB of shards | 10-30 min (bandwidth) | no |
| arithmetic data (6.55M problems) + tokenize 7.8B tokens | ~3 + ~15 min | no |
| **Stage 1 (`STAGE=to5b`)**: main run to step 140,380 (2.3B tokens) | **6.8 h** | yes |
| 5B branch (0.5B tokens) + its val | 1.5 h | yes |
| 5B evals: factuality (2.2B base + 5B, both encodings, greedy + 3 sampled seeds, Qwen2.5-7B grader), battery, gate | ~30 min | yes |
| **Stage 1 total** | **≈ 8.8 h**, then it stops for the gate decision | |
| **Stage 2 (`STAGE=to10b`)**: main run from 4.5B to 10B (5.5B tokens, decay included) + v1 val | **16.3 h** | yes |
| 10B evals (2.2B, 5B, 10B) + compare 10B vs 5B | ~40 min | yes |
| **Total GPU** | **≈ 25.8 h** (+ ~2 min per pause/resume) | |

Stage 2 can run continuously (~17 h) or in pieces: pause any time, and the hourly saves cap a crash's loss at ~1 h.

## Disk (C: has 199 GB free on 2026-10-31; WSL vdisk lives on C:)

| item | size |
|---|---|
| raw parquet (`data/corpus_v2/raw`) | ~28 GB, deletable after tokenizing (the shard list is kept in meta.json) |
| arithmetic jsonl (`data/arith_corpus_v2`) | ~1.3 GB |
| tokens (`data/corpus_v2/tok24k_ext`, uint16) | ~15.6 GB |
| main run dir: latest + latest.prev + stable_step140380 + best_model + checkpoint_epoch_1 | 5 × 1.25 = 6.3 GB |
| branch dir: latest + latest.prev + final_model (weights only) | 2.9 GB |
| **peak** | **~54 GB** (C: down to ~145 GB free); ~26 GB after deleting the raw shards |

## 10B success criterion (fixed 2026-11-01, before stage 2 finished; user decision)

**10B succeeds if the format-neutral knowledge probe (`experiments/knowledge_pmi.py`, PMI per answer token, plain "Question/Answer"
text) is higher at 10B than at 5B, with the paired-bootstrap 95% CI over the 100 questions excluding zero.** Run as
`knowledge_pmi.py base=… b5=… b10=…` (the script reports each model against the first, so b10 − b5 is a second call with b5 first).
Caveat: one training run per point, so the CI is over questions, not seeds.

After stage 2 (the real knowledge curve in chat): ft_v2 fine-tune on the 2.2B, 5B and 10B bases, 2 seeds each, the same recipe, then
chat-format factuality (`factuality_gen.py` + grader), paired over questions. The chat-format factuality of **base** models is not used
any more (see the gate note at the top).

## Evaluation points and gate

At 2.2B (existing), 5B (branch) and 10B (final), all annealed and re-scored in the same run:
1. v1 val loss (same 10.96M tokens).
2. **Factuality eval** (headline): joined encoding, greedy + 3 sampled_rep seeds, Qwen2.5-7B grader (validated 38/40 against my hand checks).
3. Corpus-format arithmetic battery (`base_arith_eval.py --max-new 160`).

**Gate at 5B** (`experiments/factuality_compare.py`, paired bootstrap over the 100 questions, 10k draws): continue to 10B only if 5B
beats 2.2B by **≥ +0.05** and the 95% CI's lower bound is above 0. Otherwise stop at 5B: 104M is then capacity-bound on facts, and the next spend should be a bigger model.
Stage 1 ends after the gate either way, and stage 2 starts only on the user's go-ahead.
Optional, +45 min per point: the ft_v2 fine-tune on each base, then factuality again.

## Commands (after approval)

```
bash scripts/corpus/download_ext.sh
.venv/bin/python scripts/distill/build_arith_plain.py --n 6550000 --seed 2 --out data/arith_corpus_v2
.venv/bin/python scripts/corpus/build_corpus.py tokenize --tokenizer models/tokenizer_clean24k.model --tokens 7.92e9 \
  --name tok24k_ext --raw data/corpus_v2/raw --root data/corpus_v2 --shares aura_distill:0 --arith-jsonl data/arith_corpus_v2/train.jsonl
nvidia-smi --query-gpu=power.limit --format=csv,noheader     # must print 450.00 W
TRAIN_CMD="env STAGE=to5b bash scripts/pretrain_10b_queue.sh" ~/.claude/skills/ml-training/scripts/launch_train.sh configs/smoke.yaml
```
(`--tokens 7.92e9` × the shares' sum 0.985 = 7.8B.)

## Code (done 2026-10-31)

- `train_overnight.py`: `--init-from`, `--schedule wsd --decay-frac --decay-start-step`, `--resume-state FILE`, `--save-every-minutes`,
  `--save-at-steps`, `--val-corpus`, `--deterministic`; RNG states in `latest.pt`; atomic save with `latest.prev.pt`; full-state save at
  every stop; pause flag (`run_control/PAUSE` → save, exit 75).
- `scripts/pretrain_10b_queue.sh` (two stages, crash marker, cap check), `scripts/lib_train.sh` (pause-aware `run_train`),
  `scripts/pause_train.sh`, `scripts/resume_train.sh`, `scripts/corpus/download_ext.sh`, `build_corpus.py --raw --root --shares
  --arith-jsonl`, `experiments/factuality_compare.py`; tests `scripts/test_resume.sh`, `scripts/test_pause_queue.sh`.
