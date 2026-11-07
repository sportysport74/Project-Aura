# Plan: Aura v3 base (~330M, context 1024, ~15B tokens) + one mixed fine-tune (written 2026-11-07, NOT RUN; needs Jon's approval)

Nothing in this plan has run except a 60-step throughput probe (below). Every number marked *estimate* comes from that probe.

## Model

vanilla2 recipe (`--projection linear --emb-scale sqrt --ffn-activation gelu --linear-init gpt`), new size **`x330`**: d_model 1024, 24 layers,
16 heads (head dim 64), d_ff 4096, **context 1024**, 24k tokenizer (`models/tokenizer_clean24k`), tied embedding. **326,887,424 params**
(measured; 3.2× l100). The size entry is already in `train_overnight.py` (no other code changed yet).

## Measured: throughput and VRAM (probe, 2026-11-07)

60 steps each, compiled bf16, 450 W cap, the current trainer, corpus `data/corpus_v2/tok24k_ext`. tok/s = the steady value at steps 40-60.

| config | tokens/step | tok/s (probe) | peak VRAM (torch) |
|---|---|---|---|
| l100, seq 512, batch 32 (today's base) | 16,384 | 211k | 9.1 GiB |
| l100, seq 1024, batch 16 | 16,384 | 200k | 9.1 GiB |
| x330, seq 1024, batch 8 | 8,192 | 62k | 12.6 GiB |
| **x330, seq 1024, batch 16** | 16,384 | **70k** | **20.9 GiB** |
| x330, seq 1024, batch 24 | 24,576 | 71k | 29.2 GiB (too close to 32 GiB: Sysmem Fallback would hide an OOM) |

**Sustained estimate: ~61k tok/s.** A short probe overstates: l100 probed at 211k but ran at ~184k for hours in the 10B run (−13%, the card
heats up and becomes power-bound). x330 × 0.87 ≈ 61k. **Choice: batch 16 × 1024 with gradient accumulation 4 = 65,536 tokens per optimizer
step** (needs a `--grad-accum` flag; the trainer has none today). Accumulation doesn't change tok/s; it gives a batch size more usual for 330M.

## Data: ~15B tokens, same sources and licences

| piece | tokens | status |
|---|---|---|
| `data/corpus_v1/tok24k_v2` (FineWeb-Edu, Cosmopedia v2, FineMath-4+, distill slice, arithmetic) | 2.20B | on disk |
| `data/corpus_v2/tok24k_ext` (the same sources, later shards) | 7.80B | on disk |
| **new `data/corpus_v3/tok24k_ext2`** | **~5.0B** | to download + tokenize |
| total | **~15.0B** | |

New shards, all unused so far (shard lists checked on the Hub 2026-11-07): FineWeb-Edu `sample/10BT` 009-013 (~2.8B tokens, 9.7 GB parquet),
Cosmopedia v2 `train-00013..00019` (~1.7B, 8.2 GB), FineMath-4+ `train-00008..00011` (~0.5B, 1.2 GB), programmatic arithmetic, new seed,
excluding every problem already in the corpus (~0.17B). Shares stay 55 / 30 / 10 / 3.5%. Licences unchanged: ODC-By 1.0 collections
(FineWeb-Edu, FineMath: CommonCrawl web text; Cosmopedia: Mixtral-8x7B output, Apache-2.0), our arithmetic. No Wikipedia, TinyStories,
TinyChat or closed-model text. Raw shards ~19 GB, deletable after tokenizing (checksums + re-download script, as for corpus_v2).

**Loader change:** the trainer reads one `--corpus`. Rather than concatenating 30 GB into a new file, add a multi-corpus memmap loader
(`--corpus a,b,c`, one epoch over all windows in a seeded permutation). ~1 h of code + tests, no GPU.

## Steps, in order (each needs the previous step's gate; nothing auto-advances past a gate without the stated rule)

### 0. Code + smoke (CPU, then ~30 min GPU)
`--grad-accum`, multi-corpus loader, `build_corpus` for ext2, `finetune_chat.py` at seq 1024 with accumulation; tests; the queue scripts.
Smoke test of **every** phase with the exact flags (~20 steps each, tiny corpus slices so epoch ends and evals run).

### 1. Scaling check: l100 vs x330, 200M tokens, 2 seeds (pre-registered)
Same data (the first 200M tokens of the 15B permutation), seq 1024, 65,536 tokens/step (3,052 steps), warmup 300, cosine to 0, seeds 1 and 2.
lr: l100 6e-4 (its measured best); x330 3e-4 (a guess, half; the lr check in step 2 fixes it). Val on the v1 val split (as every base so far).
**Rule:** go on to step 2 only if x330's val loss is lower than l100's on **both** seed pairs. If not: stop and report (more params aren't
paying at this budget; the fallback is l100 at 1024 context on the same 15B, ≈ 23 GPU-h).
*Estimate:* x330 2 × 55 min, l100 2 × 19 min ≈ **2.5 GPU-h**.

### 2. lr check at x330 (pre-registered)
3,052 steps each (200M tokens), cosine to 0, seed 1: **2e-4, 4.5e-4, 6e-4** (3e-4 seed 1 comes from step 1). Rule: lowest val loss wins; if
the winner is at an edge of the grid, run one more point beyond that edge before choosing. *Estimate:* 3 × 55 min ≈ **2.75 GPU-h** (+ ~1 h if
the edge rule fires).

### 3. Pretrain: 15B tokens, WSD, eval points at 5B / 10B / 15B
One epoch over the 15B corpus, warmup 2,000, constant at the chosen lr, then:
- **5B and 10B branches:** from the stable state, linear decay to 0 over 0.5B tokens (as the v2base 5B branch), → eval points.
- **Main:** decays over the last 10% (1.5B tokens) → the 15B base.
Saves: full state hourly (`--save-every-minutes 60`), `latest.pt` + `latest.prev.pt`, stable states at 5B and 10B, `best_model.pt`.
*Estimate:* 15B / 61k ≈ **68.3 h** + branches 2 × 0.5B ≈ **4.6 h** ⇒ **~73 GPU-h** (~3 days of GPU time, longer with gaming pauses).

**Evals at each point:** v1 val loss (old curve: 2.644 / 2.579 / 2.518 at 2.2B / 5B / 10B for l100 at 512), knowledge PMI probe (l100 10B:
0.448), corpus-format arithmetic battery. ~20 min per point.

### 4. One mixed fine-tune set (`data/distill_pilot/ft_v3`; Qwen2.5-7B-Instruct teacher, Apache-2.0; corpus store only, never Wikipedia)
| part | source | notes |
|---|---|---|
| voice | ft_v2 (fact-first, mood rule, plain maths) | as today |
| RAG | ft_rag (3 corpus sources, 20% random sources) | rebuilt at 1024 context: sources at 75 words, no trimming needed |
| **refusal / unanswerable** | new: Qwen-written questions AURA can't know (personal, real-time, after its data, obscure local) → short honest replies ("I can't know that…", optionally where to look) | **never the gate eval's test messages** (exact + near-dup filter against `data/retrieval/gate`); also ~20% *answerable* look-alikes with normal answers, so it doesn't learn to refuse everything |
| **multi-turn** | new: 2-3 turn conversations with sources at every turn (query = previous + current question), history in the prompt | fits at 1024 (step A ran with 2 sources at 60 words in 512) |
Size ~60-80M tokens; build ~3 h on the GPU (generation + filters), report and a 2,000-sample check with the gate script before the full run.
Fine-tune: 1 epoch, 2 seeds, lr 1e-4 (re-check with a 3-point sweep on 10% of the set if loss looks off), replay sleep as before.
*Estimate:* ~70M tokens / ~61k ≈ 20 min per seed + evals.

### 5. Evals of the fine-tune (pre-registered before step 4 runs, in its own file)
Factuality (100 questions, re-score protocol, with and without the corpus store), the gate eval's off-topic set (**blind hand grades, since no
LLM grader passed the hand check**; or a new grader that passes first), multi-turn (the 38 conversations), refusal rate on answerable questions
(must stay low), chat-quality read-through. Compared with today's best: `rag/s0` + gate G1.

## Totals

| | GPU-h (estimate) |
|---|---|
| smoke tests | 0.5 |
| 1. scaling check | 2.5 |
| 2. lr check | 2.75 (+1) |
| 3. pretrain 15B + 2 branches + evals | ~74 |
| 4. fine-tune data + 2 fine-tunes | ~4 |
| 5. evals | ~2 |
| **total** | **≈ 86 GPU-h** (≈ 3.6 days of GPU; steps 1-2 are ~5 h and decide whether the other ~80 h happen) |

**Disk** (715 GB free in WSL today): raw shards ~19 GB (deleted after checks), ext2 tokens ~10 GB, x330 checkpoints: a full state is ~3.9 GB
(fp32 weights 1.3 GB + AdamW 2.6 GB); latest + prev + 2 stable + branch/main bests ≈ 25 GB; scaling/lr runs ~10 GB (only `final_model.pt`
kept, ~1.3 GB each) ⇒ **peak ~65 GB, ~45 GB after the raw shards go.**

## Running it (standing rules)
- 450 W cap checked first (stop and ask if not 450). Every phase smoke-tested with the exact flags. `run_control/FAILED` + ntfy on any stop,
  ntfy at each gate and at the end.
- **Pause-friendly:** the queues use `scripts/lib_train.sh`, so `bash scripts/pause_train.sh` saves the full state and exits in ~10 s, and
  `bash scripts/resume_train.sh` resumes from `latest.pt` (checks 450 W). The hourly full-state saves cap a crash loss at 1 h.
- Watchdog (15 min without log progress → stop), GPU telemetry, crash marker (no restart after a PC crash). Auto-resume after a watchdog stop
  only if Jon approves it for this run (as for stage 2 / v2base: max 2, two faults within 3 h → stop for good).
- Steps 1 and 2 run as one chain; **step 3 doesn't start until Jon has seen the step 1-2 results** (a ~73 h commitment).
- Compiled runs aren't bit-reproducible, so the scaling and lr checks use the stated seeds and report every run.

## Risks / open points
- At 61k tok/s the pretrain is ~3 days of GPU; the two GPU faults (nvlddmkm 153) happened under exactly this kind of sustained load. Telemetry
  and HWiNFO logging as in stage 2.
- 15B tokens is ~46 tokens/param, past Chinchilla (~20). Fine for a model meant to be run a lot; the 5B / 10B points show the curve.
- The gate's garble signal uses b10_s1's NLL; with a new chat model the thresholds are re-fit on the same tune splits (cheap, ~5 min).
- The repo-level `README.md` still describes the oscillator design; see `PROGRESS.md` for where things stand.
