# Plan: extend the l100 pretrain to ~10B tokens (PLANNED 2026-10-30, NOT RUN)

Question: does knowledge, measured by the factuality eval (`experiments/factuality/`), climb with more pretraining at
104M, or is 104M the ceiling? The eval points are 2.2B (have it), 5B and 10B, each one annealed so they compare fairly.

## Starting point

- `checkpoints/pretrain/l100_on_s1/best_model.pt`: 104M, heartbeat on, 2.2B tokens of `data/corpus_v1/tok24k_v2`, warmup 2000 then
  cosine to **0** (fully annealed), val 2.8769, 94k tok/s, 6.5 h.
- Chinchilla-optimal for 104M is about 2B tokens, so 10B is about 5× past it. Loss still falls there, but slowly. The risk is that facts
  stay flat while loss improves on style; that's what the 5B gate below is for.

## Recommendation: continue, don't restart

Load the 2.2B weights with a **fresh optimizer**, re-warm to a lower peak, and use a warmup-stable-decay (WSD) schedule. Then:

- the existing checkpoint is the 2.2B point;
- the 5B point is a short decay *branch* off the stable phase;
- the 10B point is the main run's own decay.

| | continue from 2.2B (recommended) | restart from scratch to 10B, WSD |
|---|---|---|
| new tokens trained | 7.8B + 0.5B (5B branch) = **8.3B** | 10B + 0.2B + 0.5B (2B and 5B branches) = 10.7B |
| GPU time at 94k tok/s | **24.5 h** | 31.6 h |
| comparability | the 2.2B point is cosine-annealed, while 5B/10B are WSD; small schedule confound | all three points use one schedule |
| risk | loss bump after re-warming (recovers within a few hundred M tokens, see continual-pretraining work on re-warming) | none new |

The 7 h saved is worth the small confound. If the 5B-vs-2.2B difference is borderline, the from-scratch run is the clean tiebreak.

### Schedule (continue)

- Peak LR 3e-4 (half the original 6e-4: same distribution, already-trained weights), linear warmup over 2,000 steps, then constant.
- Main decay: linear to 0 over the last 10% of the run (from about 9.2B to 10B total tokens).
- 5B branch: from the stable checkpoint saved at 5.0B total tokens, decay linearly to 0 over 0.5B tokens (about 30.5k steps, 1.5 h) into
  a separate dir. The main run never sees the branch.
- Batch 32 × seq 512 = 16,384 tok/step. 7.8B / 16,384 ≈ **476k steps**. Same flags as the night pretrain otherwise: `--heartbeat live
  --no-sleep --seed 1 --log-every 200 --save-every 3000 --resume auto`.

## Corpus expansion: same three sources, new shards, no new licences

The extension trains only on documents the 2.2B run never saw. The builder reads shards in sorted order until each budget is met, so
starting the download one shard past v1's last shard guarantees no overlap. Per source, the shares match v1.

| source | licence (unchanged from v1) | in v1 | extension budget | shards to fetch | raw size |
|---|---|---|---|---|---|
| FineWeb-Edu `sample/10BT` | ODC-By 1.0 (collection; CommonCrawl web text keeps its authors' rights) | 1.21B, shards 000-001 | 4.3B | 002-008 (7 of the 12 left) | ~15 GB |
| Cosmopedia v2 (`smollm-corpus`) | ODC-By 1.0; text by Mixtral-8x7B-Instruct (Apache-2.0) | 0.66B, shards 00000-00002 | 2.35B | 00003-00013 (11 of 104) | ~13 GB |
| FineMath-4+ | ODC-By 1.0 (collection; CommonCrawl web text) | 0.22B, shards 00000-00001 | 0.8B | 00002-00009 (8 of 64) | ~2.3 GB |
| aura_arith (programmatic) | ours | 77M | 0.27B (3.5%) | new `build_arith_plain.py --seed 2`, held-out policy as v1 | <1 GB |
| aura_distill (ft_v1/ft_v2) | Apache-2.0 teacher outputs | 32M | **0, left out** | | |

- Shard counts come from v1's tokens per GB. Fetch one extra shard per source as slack, and check the budget after tokenizing.
- **Leave the distill set out.** v1 already contains the ft_v1 train split, which confounds every fine-tune comparison. Keeping it out of the
  extension means the fine-tune data is not trained on again before the fine-tune.
- **Shift to Cosmopedia? Not in this run.** It's the most "explain"-like source, so shifting the mix toward it would probably help the factuality
  eval, but it would confound "more tokens" with "different data". Keep the v1 shares here. A mix change is a separate experiment.
- **Not added (licence flags for Nodal Systems):**
  - Wikipedia / Simple English Wikipedia: CC BY-SA 4.0, share-alike. Its status for trained weights is unsettled, so I'd keep it out of a
    commercial base.
  - TinyStories: GPT-generated (OpenAI terms) and CDLA-Sharing.
  - Anything with GPT/Claude/Gemini outputs.
- New corpus dir `data/corpus_v2/tok24k_ext` (train.bin + val.bin + meta.json), tokenized with `models/tokenizer_clean24k.model`.
  Forgetting and loss are tracked on **v1 val** (`data/corpus_v1/tok24k_v2/val.bin`) so every point is scored on the same text. The new
  val split is a second check.

## Disk

The WSL vdisk lives on C:, which had 186 GB free (81% used) on 2026-10-30.

| item | size |
|---|---|
| raw parquet (FWE 15 + Cosmo 13 + FineMath 2.3 + slack) | ~33 GB, deletable after tokenizing (keep the shard list in meta.json) |
| tokens, uint16 (8B × 2 bytes) | ~16 GB |
| checkpoints: run dir (best/latest) + 5B stable + 5B branch + 10B | 5-6 × 1.25 GB ≈ 7.5 GB |
| **peak** | **~57 GB** (C: to ~87%) |

If C: gets tight, move the distro to D: (1.6 TB free) first: `wsl --manage Ubuntu-24.04 --move D:\WSL`. Don't do this mid-run.

## Time

| step | time |
|---|---|
| download ~33 GB from the Hub | ~10-30 min (bandwidth-dependent) |
| tokenize 8B tokens (v1 did 2.1B in 3.4 min, all cores) | ~15 min |
| smoke test (`--max-steps 300`, check the resumed loss starts near 2.88 and the warmup bump is bounded) | 5 min |
| main run, 7.8B tokens | **23.0 h** |
| 5B branch, 0.5B tokens | 1.5 h |
| evals per point (val, factuality ×2 encodings + grading, corpus-format battery) | ~25 min |
| **total GPU** | **~25.5 h** |

**Pre-flight: cap GPU power at 450 W** (admin PowerShell: `nvidia-smi -pl 450`; check with `nvidia-smi -q -d POWER`; resets on reboot,
so re-apply each night). The ft_v2 rewrite on 2026-10-30 sat at 575/575 W and 79 °C for over an hour. A ~78% cap usually costs only
5-10% speed and lowers both the 12V-2x6 connector load and the BSOD exposure. If it costs more than 10% on training, re-time the plan.

Run it as ~8 h nightly segments with `--resume auto`, three nights plus the branch. That also limits the damage from another 0x133 BSOD,
since the driver is still 617.42 and the clean reinstall hasn't been done. Same rule as the night queue: stop on the first CUDA error,
no retries.

## Evaluation points

Scored at 2.2B (existing), 5B (branch) and 10B (final), all annealed:

1. v1 val loss (the same 10.96M tokens).
2. **Factuality eval**: base model, `joined` encoding (base models prefer it), greedy + 3 sampled_rep seeds, Qwen2.5-7B grader. This
   is the headline number.
3. Corpus-format arithmetic battery (`base_arith_eval.py --max-new 160`).
4. Optional, +45 min per point: the ft_v2 fine-tune on the 5B and 10B bases, then the factuality eval again, to see whether the
   knowledge survives into the voice.

Cheap trend probes (not annealed, so biased; trend only): every 1B tokens of the stable phase, v1 val loss and factuality greedy-joined only.

**Decision gate at 5B.** The 2.2B base scores **0.063** (joined, sampled_rep, 3 seeds; `experiments/results/factuality/notes.md`),
which is near the floor. Its 3 seeds run 0.045-0.085.

- **Continue to 10B only if** the 5B branch beats 2.2B by at least **+0.05**, with a paired bootstrap over the 100 questions whose
  95% CI excludes 0. Run the same check after the ft_v2 fine-tune on both bases, since the fine-tuned score is less floor-bound.
- Otherwise stop at 5B. 104M is then capacity-bound on facts, and the next spend should be a bigger model, not more tokens.

## Code changes needed before running (none made yet)

1. `train_overnight.py --init-from CKPT`: load model weights only, with a fresh optimizer and schedule. Today `--resume PATH` also loads
   the optimizer and fast-forwards a cosine.
2. `--schedule wsd --decay-frac 0.1` (and `--decay-tokens` for branches): warmup, constant, then linear to 0. The cosine stays the default.
3. `--save-at-tokens 5e9`: keep a stable-phase checkpoint at a token count, besides `latest.pt`.
4. `build_corpus.py`: a `--raw` dir and a per-source shard list, so the extension reads only the new shards; plus a `download`
   subcommand that writes the shard list and licences into meta.json.
5. A queue script (`scripts/pretrain_10b_queue.sh`) in the style of `pretrain_queue.sh`: the main run, then the 5B branch, then the
   eval at each point. It stops on the first failure.
