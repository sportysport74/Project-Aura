# AURA (COSF-MI)

Custom decoder-only transformer with Kuramoto-style oscillators ("COSF"), trained on a
stories/code/math/textbook mix, then chat-fine-tuned and distilled from an open-weights teacher.
GitHub: https://github.com/sportysport74/Project-Aura (**public**, MIT). The original Windows copy
is at `C:\Users\fatba\COSF-MI` and is no longer the working copy.

## Working with Jon's vision

AURA is a long-term research project. Jon's goal is an organic AI that
deliberates by settling oscillator dynamics, has real handedness
(chirality), learns through sleep and consolidation, and weighs ideas
before answering, not a standard transformer with extras. Jon thinks
abstractly and leads with vision; Claude's job is to turn that vision
into things that can be built and tested.

- Text Jon shares about what AURA could become is VISION, not a claim
  about the current code. Don't critique it as if it were a description.
  Instead, map it to experiments: what it would take to build, what
  would show it works, and where the code is today.
- Vision ideas go in experiments/VISION.md as: idea / current state /
  experiment that tests it. Encourage new ideas; extract the testable core.
- Lead with "here's how we could test that", not "that's not true".
  When an idea conflicts with physics or our own data (e.g. full sync
  r≈1 carries no information), say so briefly and propose the
  version that could work.
- Hard rigor applies to RESULTS: anything in RESULTS.md, a paper, a post
  or a public README needs measured evidence (3 seeds, baselines,
  controls). Flag clearly if vision language is about to be presented
  as a result.
- Prefer building toward the vision over arguing about it. A failed
  experiment is information that improves the next design.

## Layout
- **Work happens in `COSF-MI/`.** Run every command from there. The repo root holds an older
  duplicate `src/`, Jarvis CLI files and an empty `pyproject.toml`/`requirements.txt`; don't edit those by mistake.
- `COSF-MI/src/python/cosf_mi/`: the package. Scripts add `src/python` to `sys.path` by hand, and
  pytest gets it from `pythonpath` in `COSF-MI/pyproject.toml`. It is not pip-installed.
  - `models/cosf_oscillator.py`: the model actually trained (`COSFOscillatorTransformer`,
    `COSFOscillatorConfig`): oscillator layers, chiral attention, Kuramoto coherence loss
    ("heartbeat"), Julia phase-velocity bound, homeostatic decay.
  - `models/cosf_transformer.py`, `models/model_configs.py`: older model and its size table, not used by
    `train_overnight.py`, which has its own size table (tiny/mid/big/small/medium).
  - `data/token_dataset_fast.py`: tokenizes text once and caches it under `cache/` (key = path string
    + mtime + tokenizer path, so Windows and Linux caches differ).
  - `training/sleep_cycle.py`, `training/replay.py`: "gentle sleep" and replay.
- `COSF-MI/src/rust/cosf_core/`: Rust crate with φ-shell decomposition, Julia iteration and phase coherence, with optional
  PyO3 bindings (`--features python`). **It is not a data layer** and no Python code imports it.
  The data pipeline is pure Python (`token_dataset_fast.py`).
- `COSF-MI/src/cuda/*.cu`: standalone kernels with a Windows build.bat; not wired into training.
- Data (gitignored), `COSF-MI/data/`:
  - `mix/{stories,code,math,textbook,styled}_{train,val}.txt`: pretraining; `train_overnight.py` reads only `styled_*` (1.27 GB)
  - `tinychat/tinychat_80mb.txt`: chat fine-tune data
  - `cosf_distill_verified.jsonl`: teacher data (qwen2.5:32b, sandbox-verified math), plus `*_rejected.log`
- Tokenizer: `models/tokenizer_mix16k.model` (SentencePiece, 16k, **digits already split one per token**).
  `tokenizer_8k` is the older one.
- Checkpoints (gitignored): `COSF-MI/checkpoints/<run>/best_model.pt`, `latest.pt` (crash recovery).
  On Linux: `aura_chat_{distill,mid,mid2,smart,smart10,tiny}` and **`smart_big_10ep`** (the base for chat fine-tunes; copied
  2026-10-26, `best_model.pt` SHA-256 `828743f0e031cfb96f080c48e7ea662c8723aa5d3fca4e01ce1657b95846d784`, identical to the Windows copy,
  re-checked 2026-10-27). Older runs are still only on `C:\Users\fatba\COSF-MI\COSF-MI\checkpoints`.

## Environment
- Venv: `COSF-MI/.venv` (uv, Python 3.12, torch 2.14.1+cu132). Use `.venv/bin/python`. Not the
  Windows `../venv/Scripts/python.exe` that the `.bat`/`.sh` files reference.
- The `.bat` files and `run_*.sh` scripts are Windows-era (Git Bash, `python.exe`, `ollama stop`).
  They need porting before they'll run on Linux.

## Commands (from `COSF-MI/`)
- Pretrain: `.venv/bin/python train_overnight.py --model-size big --epochs 10 --batch-size 32 --lr 5e-4 --warmup 200 --data-dir data/mix --tokenizer models/tokenizer_mix16k.model --checkpoint-dir checkpoints/<run> [--resume auto]`
  - Also takes `--config <yaml>` (defaults; CLI overrides), `--max-steps N` and `--log-every N` (plain `step= loss= tok/s=` lines).
- `.train_cmd` = `.venv/bin/python train_overnight.py --config`. Smoke test:
  `~/.claude/skills/ml-training/scripts/smoke_test.sh configs/smoke.yaml 200`
- Chat fine-tune and distill: `.venv/bin/python finetune_chat.py --checkpoint checkpoints/smart_big_10ep/best_model.pt --distill data/cosf_distill_verified.jsonl --out checkpoints/<run>`
  - Chat format: `You: <turn> Aura: <reply> ...`. Story replay "sleep" keeps the story skill. The best model is picked on (chat val + distill val)/2.
- Teacher data: `.venv/bin/python scripts/generate_cosf_distill.py --n 10000` (Ollama `qwen2.5:32b` at localhost:11434, now the
  WSL Ollama, which has only qwen2.5:14b pulled so far: `ollama pull qwen2.5:32b` first. Stop it before training: it needs about 20 GB of VRAM).
- Math eval: `.venv/bin/python experiments/math_eval.py <ckpt> [<ckpt>...]` (120 questions; the last number in the reply is scored).
- Chat: `.venv/bin/python chat_with_aura.py [checkpoint]` (interactive). With no argument it takes the first existing
  checkpoint from its list. **Default chat checkpoint: `aura_chat_distill`** (user decision 2026-10-28; added first in the list, before
  `aura_chat_smart10`). The distill A/B's ft_v1 models are not a replacement at 33M (they loop in chat).
- Tests: `.venv/bin/python -m pytest tests`
- **Pause training to game:** `bash scripts/pause_train.sh` (the trainer saves its full state and exits in ~10 s; the queue stops
  without starting its next stage; it prints the GPU memory so you can see it's free). **Resume:** `bash scripts/resume_train.sh` (checks
  the 450 W cap first and refuses otherwise, then relaunches the queue recorded in `run_control/LAST_QUEUE` from `latest.pt`). Works for
  queues that use `scripts/lib_train.sh` (`pretrain_10b_queue.sh`). If it says no training is running, an eval phase is on; wait or kill it.
- **After a PC crash:** nothing restarts by itself. The 10B queue records the crash in `checkpoints/pretrain/l100_10b/CRASHES.md` and
  refuses to start until relaunched with `ACK_CRASH=1 bash scripts/resume_train.sh` (the user's call).
- Resume test: `bash scripts/test_resume.sh` (SIGKILL + `--resume auto`; bit-identical in `--deterministic` fp32 eager).

## Known state (2026-10-25)
- Production size "big": d_model 512, 8 layers, 8 heads, d_ff 2048, seq 256, vocab 16k, **41.9M params**.
- Last run, `aura_chat_distill`: best (chat+distill)/2 val 1.686; story val 2.79 (was 2.66).
  Math exam 5/120, against 2/120 for smart10. It learned the answer *template* from the teacher but not the arithmetic
  ("The sum of 64 and 88 is 73"; subtraction gets answered as a sum).
- Smoke run on Linux (tiny, 11.5M params, batch 32×256), before SDPA/bf16/compile: about 100 ms/step, ~80k tok/s, 5.2 GiB peak VRAM.
  With the current defaults (bf16 + SDPA + compile): ~30 ms/step, ~260k tok/s, 2.15 GiB.
  That's only ~5.6 TFLOPS: attention was a hand-written softmax in fp32. Branch `sdpa-bf16` replaces it with SDPA + bf16
  (+18% tok/s on `big`; see Setup status). The oscillators must stay fp32, which caps what bf16 can do.
- Tests: 66 pass. Failing: `tests/test_constants.py` (imports `E_8_6`, removed from `core/constants.py`)
  and `test_phi_decomposition.py::test_assign_shells` (expects -1 below base, gets 0).
- `cosf_core` Rust test `test_convergence` fails: it asserts φ¹⁷ ≈ e^8.6, but φ¹⁷ = 3571 and e^8.6 = 5432
  (34% apart). `core/constants.py` docstring line 10 repeats the wrong value. Its `CONVERGENCE_RATIO`
  is 1.0 by construction (φ^(ln C / ln φ) = C), so it doesn't check anything.

## Setup status
Linux/WSL migration checklist. **Read this first; tick items off (with date) as they get done, add new ones as they appear.**

**2026-11-01 05:47: second GPU driver error (nvlddmkm 153)**, stage 2 at step 182,735; the stall watchdog stopped the queue at 06:03
(worked as designed). Not resumed; waiting for the user. Resume point latest.pt step 171,501 (05:04). Two 153s in ~9 h of training on driver
621.01 at 450 W: the fault recurs under sustained compute load and isn't caused by gaming. Details in `checkpoints/pretrain/l100_10b/CRASHES.md`.

**2026-11-01 03:04: stage 2 launched** (`runs/20261101-030419`, user: option 1, extension data unchanged). 10B success criterion fixed
in `COSF-MI/experiments/plans/pretrain_10b.md`: knowledge-probe gain over 5B, 95% CI excluding 0. The post-hoc gate switch is recorded in
**`COSF-MI/RESULTS.md`** (new: evidence-only results file). After stage 2: run `knowledge_pmi.py` on base/b5/b10 (the running queue doesn't
call it: don't edit a running bash script), then ft_v2 × {2.2B, 5B, 10B} × 2 seeds → chat factuality.
**Vanilla baseline (`--projection linear`) FAILED TO TRAIN; not a result.** All 9 runs sat on the unigram plateau (loss ~7.4) for
thousands of steps, with gnorm spikes to 13.7 and a final val of 4.5-7.5 with huge seed spread (oscillator arms: ~3.1). Cause found at init:
token embedding std 0.02 vs positional encoding std 0.5 (position outweighs token 25:1 in the residual stream; true for both models), and
linear q/k give attention logits 14× larger than the bounded oscillators (std 0.53 vs 0.038). The baseline needs a standard recipe
(embedding scale, init, GELU, an lr check) before it means anything. Rows are in `experiments/results/sweep/results.tsv` with arm `vanilla`: ignore them.

**10B pretrain, stage 1 DONE (2026-10-31 20:19; results `COSF-MI/experiments/results/pretrain_10b/`).** 5B branch val (v1) **2.7735**
(2.2B base 2.8769). The planned gate (chat-format factuality) **FAILED and is invalid as a knowledge test**: 0.062 → 0.017. Cause: the
extension left out the distill transcripts, so the only "You: … Aura: …" text in the new 2.8B tokens was arithmetic, and the 5B base now
answers "Why does bread rise?" with "1372 - 22 = 1350 / Answer: 1350" (battery 53% → 69%). The base's own score was simile soup from the
distill slice. **Format-neutral probe** (`experiments/knowledge_pmi.py`: log P(answer | its question) − log P(answer | 10 other questions),
plain "Question/Answer" text, same answer on both sides so fluency cancels): PMI 0.316 → **0.357, +0.041 [95% CI +0.029, +0.054]**. So
knowledge (question→answer association) does rise from 2.2B to 5B. **Stage 2 waits for the user.** Next fair gate: the same ft_v2
fine-tune on both bases, then factuality. Vanilla baseline launched 21:31 by hand (`runs/20261031-213151`; the chain's launch at 20:20 was
refused by launch_train's GPU-busy check while the grader was still exiting).

**10B pretrain, stage 1 (2026-10-31):** approved and launched 06:11 (run `runs/20261031-061128`) on `data/corpus_v2/tok24k_ext` (7.80B
tokens; raw shards 27.6 GB in `data/corpus_v2/raw`, deletable). At the 450 W cap the card is power-bound: **~86k tok/s** (not 94k),
so stage 1 ≈ 9.6 h, stage 2 ≈ 17.8 h. **12:40:07: GPU driver error** (Windows nvlddmkm Event 153, "Error occurred on GPUID 2e00",
driver 621.01): the GPU went idle and the trainer hung at step 120,179 (CUDA context dead, main thread spinning). Found at 14:15 via a pause
request, killed 16:07, not retried; details in `checkpoints/pretrain/l100_10b/CRASHES.md`. Resumed from latest.pt (step 111,979) on the
user's go-ahead. **New: stall watchdog** in `scripts/lib_train.sh` (no log progress for 15 min → kill the trainer, STOP, no retry; tested).
A driver error 153 under the new driver means the 10-29 BSOD fix isn't a full fix: the GPU still faults under long load.

**Resume point (2026-10-31):** the three tasks are done: factuality eval (10-30), fact-first ft_v2 data (10-30, mood rule + plain maths;
**not trained yet**), and the **10B pretrain plan, now with code and tests: `COSF-MI/experiments/plans/pretrain_10b.md`, waiting for the
user's approval** (≈25.8 GPU-h in two stages with a 5B gate, peak ~54 GB disk). New in `train_overnight.py`: hourly full-state saves with RNG,
`--init-from`, WSD schedule, `--resume-state`, `--val-corpus`, `--deterministic`, pause flag. Resume after SIGKILL is bit-identical
(`scripts/test_resume.sh`); pause/resume for gaming tested (`scripts/pause_train.sh`, `scripts/resume_train.sh`). l100 at the 450 W cap:
~94k tok/s (no slowdown). ⏳ **Sysmem Fallback is still allowing fallback** after the driver reinstall: eager l100 at batch 32 peaked at
34.7 GiB on the 32 GiB card and ran ~25× slow instead of raising OOM. Check NVIDIA Control Panel → Manage 3D settings → *CUDA - Sysmem
Fallback Policy* → *Prefer No Sysmem Fallback* (and whether it applies to WSL processes).

**Resume point (2026-10-30 21:30):** factuality eval built (`COSF-MI/experiments/factuality/`, `factuality_{gen,grade}.py`; results +
grader validation in `experiments/results/factuality/notes.md`): base 0.063, ft_l100 ~0.145; 104M knows almost none of the facts.
**ft_v2** built (`data/distill_pilot/ft_v2`, fact-first, `scripts/distill/build_factfirst.py`, report in `experiments/results/ft_v2_build/`),
**not trained yet**: next is ft_v2 on the l100 base (2 seeds) + factuality/readable eval vs ft_v1. 10B extended-pretrain plan (NOT RUN, needs
code changes + approval): `experiments/plans/pretrain_10b.md`. GPU power cap 450 W is now a standing rule (see Rules; done 2026-10-31).

**2026-10-31:** NVIDIA driver clean-installed: **621.01 (CUDA 13.5 UMD)**, 450 W cap applied, Sysmem Fallback policy re-set, `nvidia-smi` works
in WSL after `wsl --shutdown`. Torch sanity check passed on 621.01 (torch 2.14.1+cu132, sm_120): bf16 8192³ matmul 213 TFLOPS at the 450 W cap
(rel. err 2.9e-3 vs fp32), SDPA flash bf16 fwd+bwd OK, `torch.compile` fwd+bwd OK (fp32 max diff vs eager 3e-7).

**Resume point (2026-10-30 19:10):** branch `distill-investigation`, not pushed. Night queue done: full l100 pretrain
`COSF-MI/checkpoints/pretrain/l100_on_s1` (val 2.8769). Seeds 4-6: n=6 control−on +0.068, but control is seed-sensitive and there is no batch
effect (`experiments/results/sweep/control_check.md`; proposed check, not run: rerun control s1 bit-for-bit). Distill FT on the l100 base
(`experiments/results/ft_l100/notes.md`): forgetting halved, arithmetic held (corpus-format 53%→50%), but the voice is still simile soup
over wrong facts. Batteries depend on prompt encoding (piece-by-piece vs one encode); use `base_arith_eval.py --max-new 160` to compare
base vs FT. Next: decide between a factuality-graded explain set, a fact-first rewrite of the pilot explain slice, or more pretraining.

**Resume point (2026-10-29 21:00):** scaling sweep **DONE, 27/27**. Results in `COSF-MI/experiments/results/sweep/{results.tsv,summary.md}`.
Reading (user-corrected): **most of the gain is any phase-coherence loss, with synchrony possibly adding a small part.** At l100, off→control is
+0.091 and control→on +0.029 (paired p 0.029, n=3, thin); s25 +0.054/+0.006, m50 +0.068/+0.039. Next: the night queue
`scripts/pretrain_queue.sh` = full l100 pretrain (heartbeat on, 2.2B tokens, `--no-sleep`, ~7 h) then seeds 4-6 of l100 on vs control
(~4 h) to firm up the synchrony question. Stops on the first failure, no retries. The older resume point follows.

**Resume point (2026-10-27 evening):** `main` = `3b7ca8e` (heartbeat-ablation, heartbeat-control and sync-cleanup merged and pushed;
worktree `~/projects/aura-syncfix` removed). The control result is **corrected below**: control vs heartbeat isn't significant.
**New goal: distillation.** The investigation is done (see "Distillation investigation" below); **the pilot plan waits for user approval.
Don't start generation before that.**

### Windows BSOD during the sweep (2026-10-29) ✅ fix applied 2026-10-31
- 01:49: bugcheck 0x133 DPC_WATCHDOG_VIOLATION (DPC_QUEUE_EXECUTION_TIMEOUT_EXCEEDED). Same as 10/26 06:49. `cdb !analyze -v` puts both in the
  NVIDIA driver's DPC path (10/26: `nvlddmkm`; 10/29: `dxgmms2!VidSchiUnwaitMonitoredFences` called from `nvlddmkm`'s DPC). Driver 617.42
  (32.0.16.1742, 2026-09-29). Wallpaper Engine (`wallpaper32`) was the active process in the 10/26 crash and is running now. Older dumps
  (0x13A heap 5/26 and 6/10, 0x3B opera 9/2) look unrelated. WinDbg installed via winget; analyses in `%LOCALAPPDATA%\Temp\dumpan\` (outside the repo).
- The sweep lost only `l100_control_s2`. Relaunched as run `20261029-042725` at 04:27, finished 07:12 with no new crash.
- [x] Mitigation (2026-10-31, user): **driver clean-installed, 617.42 → 621.01**, GPU power capped at 450 W, Sysmem Fallback policy re-set.
  Applied, not yet proven: both 0x133 crashes were hours apart, so only a long run without a BSOD counts as confirmation. If it recurs,
  record the bugcheck and the driver version here. (Wallpaper Engine during training and HAGS weren't part of this fix.)

### Distillation investigation (2026-10-27; nothing changed, no generation run)
**1. The old `aura_chat_distill` run.**
- Teacher: Ollama `qwen2.5:32b` (Qwen2.5-32B-Instruct, Apache-2.0), temperature 0.8, `scripts/generate_cosf_distill.py`.
- Method: **sequence-level** (synthetic text, no logits). The teacher writes whole samples from templated prompts. 60% are "numeric"
  (question, short answer, plus a Python block with asserts run in a sandbox; kept only if every assert passes; the block isn't trained on)
  and 40% are "dialogue" about AURA's own world (kept if there's no assistant filler). Every sample fits AURA's 240-token budget.
- Data: **10,000 samples (6k numeric, 4k dialogue) = 597,586 AURA tokens, of which 287,936 are trained-on reply tokens.**
  Rejected 2,848 (acceptance 78%), **mostly duplicates (2,100)**, so template diversity was the limit. 711 numeric failed the sandbox.
- Training: `finetune_chat.py` from `checkpoints/smart_big_10ep/best_model.pt` (on Linux, SHA-256 verified against
  the Windows copy). TinyChat in the `You: … Aura: …` format with loss only on Aura's replies,
  distill blended to 30% (each sample repeated up to 4×), and story replay "sleep" every 300 steps.
- **What 1.686 is:** best checkpoint picked on (chat val + distill val)/2 = (1.8810 + 1.4910)/2. Both are reply-only losses.
  Chat val = the first 2,252 TinyChat lines after `random.Random(1234).shuffle` (minus the last, possibly cut, line). Distill val = the first
  500 of the 10k distill lines after `random.Random(seed).shuffle`, seed 0 (the default; not saved in the checkpoint).
- **Reusable? Yes.** Re-scored on CPU with today's code: **distill val 1.4910 exactly** (seed 0 confirmed), chat val 1.8779 vs 1.8810
  stored (0.003 apart, probably the original Windows-GPU numerics). Both files are unchanged since the run. **Rule for fair comparison: re-score old
  checkpoints with the current eval code instead of comparing against logged numbers.** A new teacher's data gets its own held-out set,
  but these two stay as fixed benchmarks.
- That checkpoint has **33,727,690 unique params** (+241,866 vs today's `big`). Counting the tied 16,000×512 embedding twice gives exactly
  **41.9M**, which settles the Known-state figure (an older config plus double-counting).

**2. AURA's tokenizer** (`models/tokenizer_mix16k.model`): SentencePiece, **16,000 tokens**, byte fallback (256 byte pieces, so any
Unicode round-trips), digits split one per token, newlines and indentation preserved, specials `<unk>`=0 `<s>`=1 `</s>`=2, no pad.
**Not compatible with any Qwen teacher** (byte-level BPE, ~151k vocab), so token-level logit distillation is out. Use sequence-level, or
later a cross-tokenizer method (e.g. Universal Logit Distillation), which is a project of its own. Measured: **1.10 AURA tokens per Qwen
token** on teacher replies.

**3. Teacher licences** (read from the HF model cards and LICENSE files): **Qwen2.5-7B-Instruct and Qwen2.5-14B-Instruct are Apache-2.0**
(plain Apache 2.0 text, "Copyright 2024 Alibaba Cloud", no added terms, no NOTICE file). 32B-Instruct is also Apache-2.0.
**Qwen2.5-3B-Instruct is `qwen-research`, so excluded.** Apache 2.0 puts no limit on using outputs to train other models. Attribution:
there's no NOTICE file to carry, so the obligations attach to redistributing the *weights*. For our data and model, credit
"Qwen2.5-7B-Instruct (Apache-2.0, © Alibaba Cloud)" in the run config and any dataset or model card (good practice, and our own rule).
Size: 7B is 15.2 GB in bf16 and fits with room for the KV cache. 14B is 29.6 GB in bf16, too big with KV cache on 32 GB, so it needs FP8 or AWQ weights.

**4. vLLM on the 5090: works.** Separate venv `~/venvs/vllm` (uv, Python 3.12): **vLLM 0.32.0 with its own torch 2.13.0+cu130 (sm_120 in
the arch list)**, 7.9 GB. It doesn't touch `~/venvs/ml` or the project venv. **Needs `VLLM_USE_FLASHINFER_SAMPLER=0`**: without nvcc (no CUDA
toolkit in WSL) FlashInfer's top-p/top-k sampler can't JIT-compile and engine warm-up fails, even though vLLM logs that FlashInfer kernels are disabled.
Attention backend FLASH_ATTN (FA2), vLLM's torch.compile and CUDA graphs all work. Model load + warm-up 189 s. KV cache 10.9 GiB (205k tokens).
Model in `~/.cache/huggingface/hub` (15 GB). Benchmark (Qwen2.5-7B-Instruct bf16, `max_model_len` 2048, temperature 0.7, top-p 0.9,
max 256 new tokens; 200 prompts = 80 OASST2 + 60 GSM8K + 60 templates):

| batch | sequences | time | generated tokens | throughput |
|---|---|---|---|---|
| 200 prompts × n=1 | 200 | 12.5 s | 20,497 | 1,642 tok/s (small batch, long-tail bound) |
| 200 prompts × n=4 | 800 | 11.5 s | 81,613 | **7,119 tok/s** (69.8 samples/s) |

Replies average ~100 Qwen tokens = ~110 AURA tokens. Prompt + reply averages **150 AURA tokens**, and 90-100% fit AURA's 256 window.
**Teacher GSM8K accuracy 218/240 = 91%**, and 58/60 questions have at least 1 correct of 4 samples, so gold-answer filtering is cheap and strong.
Quality issue found: a single system prompt with a maths rule ("end with Answer: N") leaked "Answer:" into non-maths replies, so use a
per-category system prompt. Scripts and outputs: scratchpad `scripts/vllm_bench.py`, `vllm_out_*.jsonl` (not in the repo).

**Licence of AURA's *existing* data (decided 2026-10-27: keep for now, hobby/research):** `data/tinychat` is **TinyChat (starhopp3r),
CC-BY-NC-4.0, generated by GPT-4o mini**, and the pretraining "styled" data is TinyStories-based (**GPT-3.5/4 generated, CDLA-Sharing-1.0**).
Both are closed-model outputs, and TinyChat is non-commercial. **A commercial AURA would need a clean base retrain:** new pretraining and
chat data without TinyChat/TinyStories (e.g. open-weights-teacher data like the pilot's), not just a new fine-tune on top of
`smart_big_10ep`, because the base already learned from them. The new pilot data avoids both.

**5. Proposed pilot plan (awaiting approval)**
- *Teacher:* Qwen2.5-7B-Instruct bf16 via vLLM (`~/venvs/vllm`), temperature 0.7-0.8, per-category system prompts, max 256 new tokens,
  and drop anything whose prompt + reply exceeds 240 AURA tokens. Record teacher + licence in every run config. A 14B-AWQ teacher is a later
  quality A/B on a subset, not the pilot.
- *Prompt sources (all licence-clean):*

  | source | licence | share | notes |
  |---|---|---|---|
  | Programmatic arithmetic and word problems | ours | ~25% | exact answers, unlimited and diverse. Targets the 64+88=73 failure. Keep only if the teacher's `Answer: N` matches |
  | Qwen-generated prompts ("Magpie": the model writes user prompts from a topic list) | Apache-2.0 (Qwen outputs) | ~45% | everyday questions, simple science, how-to, short chats |
  | Story prompts (templated characters × settings × morals) | ours | ~15% | keeps the story skill; replaces GPT-made TinyStories-style data |
  | GSM8K train, 7,473 questions × 4 samples, gold-filtered | MIT | ~8% | harder multi-step maths |
  | OASST2 English root prompts, 4,617 (15-300 chars) × 2 | Apache-2.0 | ~3% | human-written prompts |

  Excluded: TinyChat (CC-BY-NC, GPT-4o mini), UltraChat (ChatGPT), SmolTalk Magpie-Ultra and everyday-conversations
  (Llama-3.1 outputs, whose licence requires "Llama" in derived model names), Dolly (CC-BY-SA share-alike), TinyStories (GPT).
- *Size:* **50M AURA tokens** first (~330k samples at ~150 tokens). Generate once, dedupe (exact + near-duplicate), filter, and hold out 2k
  samples per category as the new val set. Scale to 100M only if the 50M pilot beats the old checkpoint on the evals below.
- *Training:* sequence-level SFT with `finetune_chat.py` (reply-only loss, story replay sleep) from the same base as before
  (`smart_big_10ep`, already on Linux; decided 2026-10-27). A/B against the old `aura_chat_distill`.
- *Evaluation beyond val loss* (all on fixed sets, re-scored for every checkpoint with the same code):
  1. Frozen val sets: TinyChat chat val (2,252), old distill val (500), story val (512 styled; measures forgetting), new pilot val (by category).
  2. **A fixed 40-prompt readable set** (10 chat, 10 explain-simply, 10 story, 10 maths). Greedy + 1 sampled reply per checkpoint, written
     side by side into a markdown file to read.
  3. **Arithmetic battery** (programmatic, held-out numbers): exact-match accuracy by operation (+ − ×) × digits (1-3). That's more informative at
     33M than GSM8K. Also the existing 120-question `experiments/math_eval.py`.
  4. Format and degeneration: % replies that end their turn (`You:`), % `Answer: N` where required, repetition (distinct-2/3), length.
  5. Optional: Qwen-7B as judge on the 40-prompt set (relevance/grammar 1-5), labelled as a weak signal.
- *Time estimate (GPU-exclusive, mostly unattended):* prompt generation and templates ~0.5 h; **teacher generation for 50M AURA tokens ≈
  33M Qwen output tokens at ~7k tok/s ≈ 1.3 h, ~1.7 h with rejection and 4× sampling for maths**; dedupe/filter/tokenize ~15 min (CPU);
  fine-tune ~1 h (`finetune_chat.py` is still eager fp32, ~70k tok/s; 50M tokens ≈ 12 min/epoch × 3 plus evals); evals ~20 min.
  **Total ≈ 4-5 h** for the 50M pilot, ~7 h for 100M. Teacher generation dominates. Training on 50M tokens is cheap.

**Pilot progress (plan approved 2026-10-27 with additions; branch `distill-investigation`):**
- [x] `finetune_chat.py` ported to compile + bf16 (`ccbf09a`). A/B, 500 steps from `smart_big_10ep` with the old distill blend: compile+bf16
  vs eager fp32 same seed: loss gap 0.005 mean / 0.013 max, val gaps chat 0.0013 / story 0.0019 / distill 0.0026. Noise floor (fp32
  seed 1 vs 0): 0.025 / 0.101 and 0.0027 / 0.0054 / 0.0039. R identical (0.32). **53k → 186k tok/s (3.5×), 11.9 → 8.0 GB.**
  New flags: `--precision`, `--compile/--no-compile`, `--max-steps`, `--log-every`, and **`--split-seed`** (the distill val split no longer
  changes with `--seed`; 0 = the split `aura_chat_distill` used). Needed for the 2-seed fine-tune A/B.
- [x] `guard_bash.py` false positive fixed (shlex punctuation mode) + `~/.claude/hooks/test_guard_bash.py` (39 cases, all pass).
- [x] Generator `scripts/distill/generate_pilot.py` and `prepare_sources.py`. Sources in `data/distill_sources/` (gitignored, licence
  in each `SOURCE.txt`); prompt JSONL in `data/distill_pilot/`. AURA-token counting matches `finetune_chat.encode_chat` exactly
  (0 mismatches on 800 real replies). Run: `VLLM_USE_FLASHINFER_SAMPLER=0 ~/venvs/vllm/bin/python -I scripts/distill/generate_pilot.py`.
- [x] **Sample run (2,000 candidates, 0.5 min generation + ~3 min model load):** kept 1,817 (0.17M AURA tokens).
  **Identity filter rejection rate: 0.0% overall and in every category** (the Aura persona system prompt is enough).
  Near-dup rate: arith 0.6% (3 exact), every other category 0.0%. Keep rates: arith 95%, explain 99.5%, chat 100%, **story 62% (120/320
  too long for the 240-token budget)**, gsm8k 78% (24 wrong answers at n=1), oasst2 89%. 20 random samples:
  `data/distill_pilot/sample/review_20.md`. Issues found: stories run long, GSM8K replies use LaTeX `\( p \)`, OASST2 brings in
  technical/code prompts and one invented fact ("My system time is 2:45 PM"), and arithmetic sometimes skips the working.
  Proposed fixes are waiting for the user's call before the full run.
- [x] **Fixes applied (2026-10-27, user-approved):** one persona for every category, now with "short sentences, everyday words, no jargon",
  plain text only (no LaTeX/code/markdown headings) and "can't see a clock, the internet or the news". Voice: neutral (warm and friendly;
  the user's voice placeholder was left unfilled). Story ≤100 words. Arithmetic: "write the calculation as an equation". GSM8K: "a few
  short steps, plain equations". New reply filters `latex` and `code`. OASST2 prompts needing code/terminal/real-time facts are dropped up
  front (4,021 of 4,603 kept). **Second 2,000-sample check** (`data/distill_pilot/sample2/`, 20 samples in `review_20.md`): kept 1,941.
  Identity rejects 0.0% again.

  | category | keep (run 1 → run 2) | dup | rejects in run 2 |
  |---|---|---|---|
  | story | 62.2% → **99.1%** | 0.3% | too_long 2, near_dup 1 (avg 207 → 141 AURA tokens) |
  | arith | 95.4% → **97.6%** | 1.4% | wrong 5 (was 20), exact_dup 7 |
  | oasst2 | 88.8% → **93.8%** | 0% | too_long 3, code 1, truncated 1 |
  | explain | 99.6% → 99.2% | 0.1% | too_long 4, exact_dup 1 |
  | chat | 100% → 100% | 0% | none |
  | gsm8k | 77.5% → **71.7%** | 0% | wrong 14 (was 24), **too_long 15 (was 3)**, latex 5 |

  Open before the full run: GSM8K replies got longer with "few short steps". Proposed: 3 samples per question, keep the shortest
  correct one within 240 tokens. Some stories stop mid-plot under the 100-word cap and some still overrun it. Small teacher factual errors exist
  (unfilterable). **Full 50M run not started; waiting for approval.**
- [x] **Third 2,000-sample check, `data/distill_pilot/sample3/`** (2026-10-27, `47de4eb`). Voice per the user: "Aura is curious and
  playful: enthusiastic about ideas, light gentle humor, notices interesting details, sometimes a short follow-up question (at most about 1 in 4),
  humor never at the expense of being correct", in the persona for every category, and math gets "at most one short playful line". GSM8K: 3 replies per
  question, keep the shortest correct one that fits. Stories: the teacher judges ENDED/UNFINISHED (temperature 0), and UNFINISHED is rejected
  (`no_ending`). New stat: % of kept replies ending in a question. Full-run sizing now uses token shares calibrated from a sample's
  stats (`--calibrate-from`), because sample 2 averaged ~80 AURA tokens/sample, not the 150 assumed (50M would have yielded only ~30M).

  | category | keep s2 → s3 | Δ pts | dup | identity | **ends in "?"** | rejects (s3) |
  |---|---|---|---|---|---|---|
  | arith | 97.6% → 96.8% | −0.8 | 1.4% | 0% | 0% | wrong 9, exact_dup 7 |
  | chat | 100% → 100% | 0 | 0% | 0% | **80.6%** | none |
  | explain | 99.2% → 99.7% | +0.5 | 0% | 0% | **61.1%** | too_long 2 |
  | gsm8k | 71.7% → 73.3% | +1.7 | 0% | 0% | 0% | too_long 13, latex 11, wrong 8 |
  | oasst2 | 93.8% → 90.0% | −3.7 | 0% | 0% | **59.7%** | too_long 5, truncated 2, code 1 |
  | story | 99.1% → **81.2%** | **−17.8** | 0% | 0% | **35.0%** | too_long 43, no_ending 17 |

  **Gate: FAIL, so the full run was NOT started** (rule: identity 0%, dup < 10%, keep within 10 points, "?"-endings < 35% in every
  category). Failed: question endings in chat/explain/oasst2/story, and the story keep rate −17.8. Causes found:
  1. "At most about 1 in 4" can't be followed: each reply is generated independently and never sees the others, so the teacher asks in most replies.
     **Proposed fix: decide in code per request.** ~25% of non-math requests get "end with one short follow-up question", the rest get
     "don't end with a question"; stories never end with a question.
  2. The `no_ending` rejects are mostly **false positives caused by (1)**: the stories end ("From then on, they weren't afraid of the dark
     anymore.") and then add "What do you think happened next?", which the judge reads as unfinished.
  3. The playful voice makes stories longer: kept median 109 words (asked ≤100); too_long rejects median 175 words. **Proposed: "70-90 words"**.
  4. GSM8K: best-of-3 barely helped (73.3%): too_long 13 and LaTeX 11 (up from 5; the voice?). Proposed: "write × or *, never LaTeX".
  Tokens/sample in s3: story 174, gsm8k 169, oasst2 110, explain 104, chat 81, arith 45. Waiting for the user.
- [x] **Fourth 2,000-sample check, `data/distill_pilot/sample4/`** (2026-10-27, `e1e3abb`; user-approved fixes). The follow-up question is
  decided **in code per request**: 25% of chat/explain/oasst2 requests get "end with one short, natural follow-up question", the rest "do not
  ask any question; end with a statement"; never for stories or maths. Playful is defined **without questions** (vivid comparisons, a fun or
  surprising detail, a light joke, enthusiasm), with three tone examples in the persona (volcano/soup, owl parliament, yeast burps). "Don't make up
  memory tricks or mnemonics unless they clearly make sense." Stories "70 to 90 words, with a clear ending, do not end with a question".
  GSM8K "never use LaTeX: write × or *".

  | category | keep s3 → s4 | Δ vs s3 | Δ vs s2 | dup | identity | ends in "?" | rejects (s4) |
  |---|---|---|---|---|---|---|---|
  | arith | 96.8% → 94.0% | −2.8 | −3.6 | 0.6% | 0% | 0.2% | wrong_answer 27, exact_dup 3 |
  | chat | 100% → 99.7% | −0.3 | −0.3 | 0% | 0% | **27.0%** | truncated 1 |
  | explain | 99.7% → 99.7% | 0 | +0.5 | 0% | 0% | **26.3%** | too_long 2 |
  | gsm8k | 73.3% → 69.2% | −4.2 | −2.5 | 0% | 0% | 0% | too_long 21, wrong 11, latex 4, truncated 1 |
  | oasst2 | 90.0% → 92.5% | +2.5 | −1.2 | 0% | 0% | **31.1%** | too_long 4, truncated 1, code 1 |
  | story | 81.2% → **97.5%** | **+16.2** | −1.6 | 0.3% | 0% | 0% | too_long 6, no_ending 1, exact_dup 1 |

  The in-code question decision works: replies end in "?" 96-99% of the time when assigned a question and 2-3% when told not to; ~25-31% overall.
  The `no_ending` false positives are gone (17 → 1). Story median dropped (162 tokens/sample).
  **Gate (literal, vs the last run = sample3): FAIL on one item only: story keep moved +16.2 points.** That's an *improvement*
  (back to sample2's 99.1% level, −1.6 vs s2), which the "within 10 points" rule counts as a failure. As stated beforehand and per the rule,
  **the full run was NOT started.** Every other condition passes: identity 0%, dup ≤0.6%, "?" ≤31.1%, other keep rates within 4.2.
  New finding: **arith wrong_answer 9 → 27, but 23 of those 27 are correct answers missing the `Answer: N` line** (the playful line replaced it:
  "…Each friend would get 54 candies! Yum yum!"). Only 4 are truly wrong (e.g. 71 × 565 = 40015). The filter is right to drop them (format).
  A fix is optional (keep ≥94% anyway): "the playful line goes before the Answer line". Tokens/sample s4: story 162, gsm8k 171, oasst2 104,
  explain 101, chat 85, arith 59. Full run, once allowed: `--target-tokens 50e6 --calibrate-from data/distill_pilot/sample4/stats.json`.
- [x] **Gate fixed (2026-10-27, user decision):** the story +16.2 was a recovery, not a failure. `scripts/distill/gate.py <run stats> --ref <last
  accepted stats>` now fails only on keep-rate **drops** > 10 points (plus identity > 0, dup ≥ 10%, "?"-endings ≥ 35%). Checked:
  s4 vs s3 PASS, s4 vs s2 PASS, s3 vs s2 FAIL (questions, story −17.8). Use it for every future sample run.
- [ ] **Full run attempt 1: STOPPED by the near-dup guard** (2026-10-27, `c1e1aa8`; log `data/distill_pilot/full_run.log`, partial output
  `data/distill_pilot/full/`: 13,662 samples / 1.17M AURA tokens, **not to be used**). Planned: 619,092 requests (arith 255,706, explain 187,208,
  chat 106,854, story 57,830, gsm8k 7,473 ×3, oasst2 4,021) → ≈52.6M tokens. It stopped after the first 16k-request chunk (19.4 min incl.
  stage 1): **arith dup rate 11.0%** (≥10%). The process exited cleanly and the GPU is idle. Two problems in *my prompt generation*, found at scale only:
  1. **Arith duplicates are repeated problems:** 1,202 of 1,219 dup rejects are *exact* duplicates (17 near). `arith_prompt` draws
     (op, a, b) with replacement from a small space (e.g. division: divisor 2-12 × quotient 2-99 ≈ 1,078 problems), so "240 ÷ 8" appeared 4× in 12k.
     Also arith wrong_answer 900/12,031 (7.5%), mostly the sample4 format issue (the playful line displaces `Answer:`).
     Fix: each (op, a, b) at most once, wider ranges (divisor up to 99, larger quotients, some 4-digit +/−), and "the Answer line comes last;
     any playful line goes before it".
  2. **Stage-1 prompt pools collapsed:** explain **5,532 unique of 215,299** written prompts (2.6%), chat **7,475 of 122,892** (6%). The
     teacher repeats itself with only ~140 topics × 6 askers × 5 styles, so the run shrank to 338,037 requests before starting. Fix: a stage 0
     where the teacher expands each topic into ~20 subtopics (~2,800); more askers/styles plus a random "angle" per request; near-dup dedupe of prompts;
     and a **unique-prompt yield gate (≥80%) measured in the sample run** before any full run (the 2,000-sample checks were too small to show this).
  Waiting for the user's approval of these fixes. **Don't start the fine-tune until the user has read sample4.**

Earlier resume point (2026-10-26): branch `sdpa-bf16` (since merged) had SDPA + bf16 + torch.compile as defaults in `train_overnight.py`,
measured over 500-step runs only. Next steps, in order:
1. [x] **Heartbeat: kept as is** (live, strength 0.1, target 0.3; user decision 2026-10-26). It works as a floor on R:
   without it R stays ~0.01, it lifts R to ~0.3 by step 60, is 0 while R > 0.3, and switches back on when R drifts down
   (the old tiny 10k-step A/B ended at R 0.246). Val cost is nil (3.613 vs 3.592-3.613 without), but there's no measured
   benefit either. Open idea: an A/B with and without it on a capability (math eval or forgetting), not val loss.
   **Ablation, 1 epoch `big` (2026-10-26):** `--heartbeat-strength 0` (`runs/20261026-213533`) against arm 1 (`runs/20261026-182317`),
   same seed and config, both compiled. Without the heartbeat, R settles at **0.004** (0.002-0.006 from step 100 on) instead of
   ~0.30. Val **3.1530 vs 3.1094 (+0.044 worse)**, avg train loss 3.6024 vs 3.5404, and the train curve is higher at every point
   from ~step 500 (smoothed gap 0.06 mean, 0.08 max). So at `big` scale the floor helps, unlike the old tiny 10k-step A/B.
   **Confirmed over 3 seeds (2026-10-27; seeds 2/3 ran back to back, interleaved on/off):**

   | | seed 1234 | seed 2 | seed 3 | mean ± sd |
   |---|---|---|---|---|
   | val, heartbeat ON | 3.1094 | 3.1213 | 3.1184 | **3.1164 ± 0.0062** |
   | val, heartbeat OFF | 3.1530 | 3.1444 | 3.1451 | **3.1475 ± 0.0048** |
   | paired gap (off − on) | +0.0436 | +0.0231 | +0.0267 | +0.0311 |

   The gap of means (+0.031) is 5.0× the ON-arm seed-to-seed sd and 2.6× its range. All 3 seed pairs point the same way, and the ranges
   don't overlap (worst ON 3.1213 < best OFF 3.1444). **The heartbeat's R floor improves val by ~0.03 at `big` scale, 1 epoch.**
   R with the heartbeat off: 0.004-0.006 on every seed. Runs: on s2 `runs/20261027-052255`, off s2 `runs/20261027-054935`,
   on s3 `runs/20261027-062532`, off s3 `runs/20261027-065319`.
   **Heartbeat-off is consistently slower:** 181.5k / 186.3k / 186.8k tok/s against 259.7k / 247.4k for heartbeat-on on the same
   sync-free code (−27%). Not GPU sharing, since the runs were interleaved. Unexplained; the leading guess is that strength 0.0 changes
   the compiled graph or causes recompiles. Check with `TORCH_LOGS=recompiles,graph_breaks` and a short profile. It only matters for ablations.
   With the sync-free Julia code, heartbeat-on runs at **~254k tok/s** over a full epoch.
   **Control experiment (2026-10-27): is it synchronization, or just an extra loss term?** New mode `--heartbeat live_control`
   (`heartbeat-control` branch): the identical loss (strength 0.1, drift gate, target 0.3) on R_ψ = |mean_j e^{i(θ_j+ψ_j)}| with
   fixed random offsets ψ (seed 271828, not saved). It's met by a random phase pattern, so true R stays ~0.
   Tiny-model check: optimizing only that loss gives R_ψ 0.997 with true R 0.017 (the heartbeat gives true R 0.997). `live` mode is bit-identical to `main`.
   1 epoch `big`, same config (`configs/val_compile_bf16.yaml --heartbeat live_control`). **Final: 6 control seeds, 4 ON, 3 OFF**
   (seeds 4-6 control and seed 4 ON added 2026-10-27 evening on `main` `3b7ca8e`; faster code, same numerics):

   | arm | val per seed (1234, 2, 3, 4, 5, 6) | n | mean ± sd | median | true R |
   |---|---|---|---|---|---|
   | heartbeat ON | 3.1094, 3.1213, 3.1184, 3.1084 | 4 | **3.1144 ± 0.0064** | 3.1139 | 0.30 |
   | CONTROL | 3.1336, 3.1273, 3.1791, 3.1145, 3.1038, 3.1165 | 6 | **3.1291 ± 0.0266** | 3.1219 | 0.004-0.005 |
   | heartbeat OFF | 3.1530, 3.1444, 3.1451 | 3 | 3.1475 ± 0.0048 | 3.1451 | 0.004 |

   - Control vs ON: +0.015, **Welch t = 1.30, df 5.8, p = 0.24** (one-sided 0.12); **Mann-Whitney U = 16/24, exact p = 0.48**.
     Pooled ranking of ON + control (1 = best): ctrl5, on4, on1234, ctrl4, ctrl6, on3, on2, ctrl2, ctrl1234, ctrl3. The **best run of all is a control**.
   - OFF vs ON: +0.033, Welch t = 7.81, df 5.0, **p = 0.0006**; Mann-Whitney 12/12, exact p = 0.057 (the floor for 3 vs 4).
   - Control vs OFF: −0.018, Welch p = 0.16, Mann-Whitney p = 0.17.
   **Conclusion (corrected 2026-10-27):** the heartbeat clearly beats no heartbeat. The control is **not distinguishable** from the heartbeat
   (p = 0.24 / 0.48) and sits between ON and OFF with a large spread, so this experiment **can't say whether synchronization itself
   helps or any phase-coherence loss of this shape does.** The earlier "synchronization itself is what helps" (3 control seeds) was overstated:
   "4.9×" used only the ON arm's sd and ignored the control's own spread (0.028, driven by seed 3 at 3.179). To separate the two you'd
   need many more seeds (the control's sd is ~4× the ON sd) or a longer horizon where the gap might grow.
   Notes: the control's loss term averaged 1.65× the heartbeat's (logged `sync=`), and its seed 3 fell behind by step 500 with no
   instability. The control still pushes each token's phases toward a coherent (random) pattern. Runs: control s1234/2/3
   `runs/20261027-074042`, `-081106`, `-084020`; s4/5/6 `runs/20261027-171812`, `-181204`, `-183912`; ON s4 `runs/20261027-174718`.
   **Speed clue:** the control (strength 0.1) runs at full speed (~238k), so the slow heartbeat-OFF runs are specific to
   `--heartbeat-strength 0`. That supports the guess that strength 0.0 changes the compiled graph (still unverified).
2. [x] **Validation done (2026-10-26): recommend merging.** 1 epoch of `big`, seed 1234:

   | | compile+bf16 (`runs/20261026-182317`) | eager fp32 (`runs/20261026-190304`) |
   |---|---|---|
   | val loss (all 200 val batches) | **3.1094** | 3.1166 |
   | avg train loss | 3.5404 | 3.5541 |
   | median tok/s | **219.2k** | 69.9k |
   | R after step 1000 | 0.297-0.305 (mean 0.302) | same |

   Smoothed loss gap 0.014 mean / 0.051 max; R gap 0.0002 / 0.0011. Both arms show the same two loss jumps at the same
   steps (15,900: 2.94→3.51 / 2.94→3.52; 20,300: 2.85→3.36 / 2.86→3.36). They're hard batches, not instability: arm 1's final
   model scores those exact batches +0.56 and +0.52 above the batch before, the same size as the jumps. The replay with grad norm
   logged (`runs/20261026-203208`, every 10 steps): gnorm at the jump is 1.12× / 1.02× the median before (0.46 / 0.40, not clipped),
   loss is back to normal at the next 10-step point, LR is smooth (cosine 74% / 94% through, 7.9e-5 / 3.9e-6), there's no epoch or shard
   boundary, and no sleep. Grad norm over the run: median 0.44, max 1.40, 0.5% of steps clipped.
   **Compiled runs aren't bit-reproducible:** the same seed and config diverged from step 200 (eager runs were identical).
   `train_overnight.py` now logs `gnorm=` (pre-clip) on each log line.
   Planning notes kept from before the run: 1 epoch of `big` (21,504 steps, batch 64, 1 epoch of `big` (21,504 steps, batch 64,
   seed 1234, full cosine schedule), two arms through `launch_train.sh`, one after the other:
   `configs/val_compile_bf16.yaml` (run `runs/20261026-182317`, ~27 min) then `configs/val_eager_fp32.yaml` (~75 min, launch
   it by hand after arm 1 exits). Compare the end-of-epoch val loss on all 200 val batches, R (`coh=`) and loss spikes. One compile-fp32
   500-step run spiked at step 500 (1 in 12), so watch for spikes. Merge into `main` only if it holds.
3. [x] **Sync cleanup**: Julia check (2026-10-26) and the remaining syncs (2026-10-27), see below.
Updated 2026-10-26.

Done:
- [x] WSL RAM limit: `free -h` shows 23 GiB (`.wslconfig` memory=24GB, mirrored networking).
- [x] `.claude/settings.local.json`: 48 Windows-era permissions cut to 14 Linux-relevant ones (gitignored, not in the repo).
  Dropped `timeout:*` (it let any command through unprompted). `git push:*` and `pkill:*` are still allowed; consider removing them.
- [x] `.wslconfig` has `[experimental] hostAddressLoopback=true` (2026-10-26), in effect since the reboot.
- [x] Smoke test passes. Baseline (tiny, batch 32×256, 200 steps, fp32, hand-written attention):
  loss 8.48 → 6.72, ~77k tok/s, ~106 ms/step, **5.24 GiB peak VRAM**.

- [x] Docker GPU test (2026-10-26): `docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi` sees the
  RTX 5090 (32607 MiB, driver 617.42 at the time; 621.01 since 2026-10-31). Docker Desktop doesn't autostart after a reboot; while it's down, `docker` is missing
  in WSL (it's a symlink into `/mnt/wsl/docker-desktop`). Start it from Windows first.

- [x] Second opinion via Ollama (2026-10-26): **Ollama now runs inside WSL** (0.40.11, systemd service `ollama`,
  `127.0.0.1:11434`), with `qwen2.5:14b` pulled. `second_opinion_status()` reports it up, and a real `ask_second_opinion` critic call works.
  The MCP config (`~/.claude.json`, `OLLAMA_URL=http://localhost:11434`) didn't need changing.
  Windows Ollama is quit and disabled at startup: WSL→Windows loopback never worked (SYNs stuck in `SYN_RECEIVED`,
  even with `hostAddressLoopback`), and in mirrored mode the two would fight over port 11434. Don't re-enable it.
  - **Ollama VRAM:** systemd override has `OLLAMA_CONTEXT_LENGTH=8192`, `OLLAMA_KEEP_ALIVE=2m`, `OLLAMA_MAX_LOADED_MODELS=1`.
    qwen2.5:14b takes **~10 GB** (100% GPU) while loaded and unloads 2 min after the last call. Training has priority:
    `launch_train.sh` (ml-training skill) runs `ollama stop` on every loaded model and waits for the VRAM to free before
    its GPU-busy check (tested: 11.9 → 1.4 GB in ~3 s). A second-opinion call *during* a run reloads ~10 GB, so either
    size batches for ~20 GB or don't use it mid-run. Smoke tests and ad-hoc runs don't go through `launch_train.sh`; for those, run `ollama ps` first. Running the 32b teacher (~20 GB) means no training at the same time.

Open:
- [x] `gh auth login` (2026-10-26): logged in as sportysport74 over https (scopes: repo, workflow, gist, read:org).
  Token files are off-limits to Claude: `~/.claude/settings.json` denies Read/Edit on `~/.config/gh/**` and the HF token files.
- [x] Batch-size probe (2026-10-26), `big` (41.9M), seq 256, fp32, 30 steps; tok/s is the steady value at step 30:

  | batch | torch peak | device peak | % of 32607 MiB | tok/s |
  |---|---|---|---|---|
  | 32 | 9.71 GiB | 12.6 GB | 39% | ~51k |
  | 64 | 18.88 GiB | 21.7 GB | 67% | ~68.5k |
  | 72 | 21.18 GiB | 23.3 GB | 72% | ~67-69k |
  | **80** | **23.47 GiB** | **25.7 GB** | **79%** | **~68.6k** |
  | 128 | did not fit: spilled to system RAM, 53 s/step | 32.1 GB | | |

  About 0.29 GiB per sample. Throughput plateaus at batch 64, so batch 80 buys memory use and no speed.
  The 77k smoke baseline is the *tiny* model, so it isn't comparable.
- [ ] **Sysmem fallback hides OOM.** Past the VRAM limit, the Windows driver pages into system RAM instead of raising
  `OutOfMemoryError`, so the smoke test's OOM check can't catch it. A run just gets roughly 250× slower. The fix is on Windows (user):
  NVIDIA Control Panel → Manage 3D settings → *CUDA - Sysmem Fallback Policy* → *Prefer No Sysmem Fallback*; whether it applies to WSL
  is unverified. Until then, treat step_ms jumping by 100× as an OOM.
- [x] SDPA + bf16 (2026-10-26, branch `sdpa-bf16`). `KuramotoAttention` (the only attention `big` uses) now calls
  `F.scaled_dot_product_attention(is_causal=True, scale=φ/√d_k)`; the model passes `causal=True` instead of a tril mask.
  There's no custom bias, and the only mask is causal. bf16 runs the **flash** kernels for forward and backward (verified with the profiler);
  fp32 uses the mem-efficient kernel. `--precision bf16` is now the default in `train_overnight.py`.
  **Oscillator layers always run in fp32** (autocast is disabled inside `COSFOscillatorLayer.forward`): Θ reaches ~2400 rad, where
  the bf16 step size is 16 rad. CE and the heartbeat loss are fp32 too. So bf16 only covers the attention core and `lm_head`.
  The attention-coherence diagnostic is recomputed for batch element 0 only (SDPA returns no weights).
  fp32 equivalence against the old code on CPU: logits within 3e-7, grads within 4e-9.
  A/B (`big`, batch 64, 500 steps, seed 1234; tok/s = median of steps 100-500):

  | arm | tok/s | torch peak | val loss (50 batches) |
  |---|---|---|---|
  | A old attention, fp32 | 67.4k | 18.88 GiB | 5.717 |
  | B SDPA, fp32 | 74.7k (+11%) | 16.63 GiB | 5.831 |
  | C SDPA, bf16 | **79.4k (+18%)** | **15.90 GiB** | 5.826 |

  A and B are the same maths and still differ by a mean of 0.050 on the smoothed train curve and 0.11 on val, from float
  noise compounding after about step 150. That's the noise floor at 500 steps. C vs B is 0.046 / 0.005, inside it, so bf16 is kept.
- [ ] **Next speedups, from a torch.profiler run** (2026-10-26; `big`, batch 64, bf16, 3 steps after 5 warmup steps; ~215 ms/step):
  the GPU is busy 80.6% and idle 19.4% (41.6 ms/step). ~3,000 kernel launches per step. The mostly elementwise oscillator
  chain (mul/add/cos/sin/sum/neg in fp32) is ~60% of GPU time. `aten::mm` is 26% and runs as fp32 SIMT sgemm, with no tensor cores.
  Flash attention doesn't make the top 15. There are 55 real syncs per step: **48 in `apply_julia_correction`** (one `.item()` per oscillator,
  33 ms CPU blocked), 2 in forward (`drift.item()` and the heartbeat `coherence.item()`, 12 ms), and 3 in bookkeeping. The data loader costs 2.4 ms/step of GPU idle.
  All 144 fp32 `aten::mm` calls are the oscillators' Ω·x (`F.linear(x, handed_freq)` in `COSFOscillatorLayer.forward`): 48 forward
  plus 96 backward. The other 3 are `lm_head` (already bf16).
- [x] **torch.compile, now the default** (`--compile`, `--no-compile` for eager; `model.compile()` in place, so checkpoint keys are unchanged
  and checkpoints load into an eager model, verified). 3-seed matrix (`big`, batch 64, 500 steps, seeds 1234/2/3), each run compared with eager fp32 of the same seed:

  | config | tok/s (mean of 3) | torch peak | R gap 100-500 | loss gap 100-500 | val (mean) |
  |---|---|---|---|---|---|
  | eager fp32 | 74.8k | 16.63 GiB | - | - | 5.789 |
  | eager bf16 | 78.6k | 15.90 GiB | 0.0021 | 0.045 | 5.790 |
  | **compile bf16** | **215.9k (2.75×)** | **7.60 GiB** | 0.0025 | 0.031 | **5.787** |
  | compile fp32 | 148.6k | 11.16 GiB | 0.0013 | 0.040 | 5.969* |

  The seed-to-seed spread of eager fp32 (from init alone) is R 0.0024 and loss 0.044, so every gap is inside it. *compile fp32 seed 3
  spiked at step 500 (train loss 5.65 → 6.51, val 6.29). It's 1 event in 12 runs, cause unknown; that config isn't used. On seed 1234, eager bf16 had
  shifted R early (0.0043 vs the 0.0011 floor), but seeds 2 and 3 didn't (0.0005, 0.0007), so it wasn't systematic. Note: the
  heartbeat sync loss is exactly 0 whenever R > target (0.3), which is most of the time after step ~50, so watch R (`coh=`), not `sync=`.
  The smoke test (tiny) with the defaults runs at ~260k tok/s, 2.15 GiB peak (was 77k and 5.24 GiB).
- [x] TF32 for Ω·x (`--matmul-precision high`): **rejected, default stays `highest`.** +4% tok/s only (the step is bound by syncs and launches), and its loss gap
  0.064 / val 0.078 vs the same-seed baseline is above the seed spread on a single run. Not worth the phase-precision risk.
- [x] Julia correction sync-free (2026-10-26, branch `julia-nosync`, `ab9a070`): bit-identical, syncs 55 → 7 per step,
  `big` batch 64 compiled **216k → 240k tok/s** (2 seeds). Compiled profile: 104 → 84 ms/step, GPU busy 56% → 68.5% (26 ms/step idle left).
- [x] Remaining syncs removed (2026-10-27, branch `sync-cleanup`, `02a7e1a`). In the `live`/`live_control` heartbeat, drift stays a
  tensor and coherence comes back as a detached 0-dim tensor (callers read it with `float()`/f-strings; `ab_run.py` fixed for JSON).
  Static mode still takes a float. In the trainer, epoch loss is summed on the GPU in float64; the progress bar and sleep check share one `.item()`
  every 10 steps; batches come from pinned memory with `non_blocking`. **Bit-identical** to the previous code (loss, sync loss, coherence,
  every grad) in live, live_control and static modes, with the drift gate open, partial, near the edge (0.98) and clamped (1.5).
  Real syncs/step 7 → ~1 (that one is the every-10-steps read; the other ~455 `_local_scalar_dense` are AdamW's CPU step counters, no GPU wait).
  Compiled profile: 83.6 → 73.6 ms/step, GPU busy 68.5% → **83.4%**. **tok/s 255k/257k → 295k/294k (+15%, seeds 1234/2)**, same
  7.60 GiB peak. Remaining idle (12 ms/step) is mostly launch gaps in backward. Next levers: `torch.compile(mode="reduce-overhead")` (CUDA
  graphs) and fused AdamW (`fused=True`). Both need an A/B.
- [x] Data loader pinned memory: done as part of the sync cleanup (the H2D copy no longer stalls). Workers aren't needed (loader idle ~0.5 ms/step).
- [ ] Spend the freed memory: seq_len > 256 or a bigger model. Re-run the batch-size probe on `sdpa-bf16` first.
- [x] Param count mismatch explained (2026-10-27): `lm_head` is **tied** to the token embedding, so today's `big` has 33,485,824
  unique params. The `aura_chat_distill` checkpoint has 33,727,690 (an older config, +241,866), and counting its tied 16,000×512 matrix
  twice gives exactly 41.9M, the Known-state figure. Known state should say "33.5M unique (41.9M counted the tied embedding twice)".

## Rules
- Distillation teachers must be open-weights with a license that allows training on outputs; record teacher + license per run.
- **Plan for possible commercial use** (decided 2026-10-28): every new dataset, teacher and model choice must allow commercial use.
  Flag non-commercial (CC-BY-NC), share-alike (CC BY-SA), "research only" terms and closed-model outputs before using them.
  Today's base and chat data (TinyStories, TinyChat) already fail this, so the next pretrain must use a clean corpus.
- Never commit checkpoints, data, logs, caches, `memory/jarvis/` or `.claude/settings.local.json` (see `.gitignore`).
  The repo is public: scan new files for secrets before pushing.
- **Committed results use repo-relative paths** (decided 2026-10-28): no `/home/...` or `C:\Users\...` in anything under
  `experiments/results/` or other committed output. Eval scripts write paths through `ab_eval.rel()`. Check before committing:
  `git grep -n /home/ -- COSF-MI/experiments/results` must print nothing.
- One GPU: don't run the 32b teacher and training at the same time.
- **GPU power cap 450 W for every long run** (user rule, 2026-10-31). The cap (`nvidia-smi -pl 450`, admin PowerShell on Windows) **resets on
  reboot**, so check it before any long run: `nvidia-smi --query-gpu=power.limit --format=csv,noheader` must print `450.00 W`.
  **If it doesn't, stop and ask the user; don't launch** (and don't try to set it from WSL).
- **Compiled runs aren't bit-reproducible** (`--compile` is the default): the same seed and config diverge after ~100-200 steps.
  Eager runs with the same seed are bit-identical. Published or compared results must use `--no-compile`, or report
  several seeds (mean ± spread), never a single compiled run.

### Distill full run attempt 2 (2026-10-27, two-stage pipeline, `d92b932`)
- [x] Stage 1, full-scale prompt pool (`scripts/distill/build_pool.py`, `data/distill_pilot/pool/`): **POOL GATE PASS**. Survival: arith
  100%, explain 99.7% (186,613/187,208), chat 99.9% (106,786/106,854), story, gsm8k and oasst2 100%. 7,117 subtopics from 250 topics; top
  subtopic ≤0.015%, top topic 0.49%. Expected ≈52.4M AURA tokens. Root cause of attempt 1's collapse: one shared vLLM seed, so identical
  requests gave identical outputs; fixed with per-request seeds. Note: some subtopics are adult or off-scope ("Solar energy policy").
- [x] Stage 2, 2,000-sample reply check (`data/distill_pilot/sample5/`): **gate PASS** vs sample4 (drops only). Keep: arith 90.4 (−3.6),
  chat 99.4, explain 99.7, gsm8k 69.2, oasst2 90.0, story 93.4 (−4.1); dup ≤0.2%; identity 0%; "?" chat 28.9 / explain 24.3 / oasst2 29.2%.
- [x] **Full reply generation DONE** (2026-10-27 21:40 → 2026-10-28 00:13, 152 min, no early stop; `data/distill_pilot/full/`:
  `kept.jsonl` 590 MB, `rejected.jsonl`, `stats.json`, `run_config.json`): 618,429 requests (every prompt unique) → **584,706 samples,
  59.25M AURA tokens**. Identity rejects 0%. Dup ≤0.3% everywhere.

  | category | kept | AURA tokens | share | keep | dup | ends in "?" |
  |---|---|---|---|---|---|---|
  | arith | 228,166 | 19.19M | 32% | 89.2% | 0.3% | 0% |
  | explain | 186,131 | 19.35M | 33% | 99.7% | 0.1% | 26.7% |
  | chat | 106,302 | 10.31M | 17% | 99.6% | 0.3% | 27.5% |
  | story | 54,906 | 9.05M | 15% | 94.9% | 0% | 0% |
  | gsm8k | 5,403 | 0.94M | 2% | 72.3% | 0% | 0% |
  | oasst2 | 3,798 | 0.42M | 1% | 94.5% | 0.1% | 26.9% |

  More than the planned 52.4M because arith replies grew to 84 tokens/sample (calibrated: 59) with the wider ranges and the playful line,
  so arith is 32% of tokens, not ~25%. To get the planned mix, subsample arith to ~13M tokens when building the fine-tune set.
  **Next (needs the user): read sample4/sample5, then the fine-tune A/B** (≥2 seeds per arm vs the old `aura_chat_distill` re-scored with current code).

### Distill fine-tune A/B (2026-10-28; built in a second session `fatba-bf`, handed over to this one)
- **Fine-tune set `data/distill_pilot/ft_v1`** (`scripts/distill/build_finetune_set.py`; report copied to
  `experiments/results/ab_distill/ft_v1_build_report.md`): from the 584,706-sample run, simile openers capped at 30% per category (they were in
  42-64% of explain/chat/arith replies), repeated arith/gsm8k quips ("Sure thing", candies, piles…) thinned to ≤1% each, and arith problems used
  by any eval dropped (133). Val: 2,000 per category (gsm8k 300, oasst2 200). **Train: 316,502 samples / 34.22M AURA tokens.** Shares: explain
  34.4, story 25.5, **arith 22.0 (user accepted, target was 25)**, chat 14.8, gsm8k 2.5, oasst2 0.9%. The quip thinning removed 51% of arith, so the op
  mix is skewed (÷ 31%, + 25%, × 13%).
- **New eval tools:** `experiments/arith_battery.py` (410 held-out problems in 11 cells), `experiments/readable_40.json`, `experiments/ab_eval.py`
  (frozen val sets, battery, the 120-question `math_eval`, readable side-by-side, repetition/turn-ending stats; `--base` scores the base model),
  `scripts/ft_ab_queue.sh` + `configs/ft_ab/*.args`. `finetune_chat.py --pilot DIR` trains on DIR/train.jsonl only (no TinyChat) and picks the best on
  pilot val.
- **Design:** both arms start from `smart_big_10ep`, with the same recipe (batch 32, lr 1e-4, 1 epoch, replay sleep, compile + bf16, `--split-seed 0`), seeds 0 and 1.
  NEW = ft_v1 only (licence-clean, 9,890 steps). OLD = the exact `aura_chat_distill` recipe retrained (TinyChat + old distill at 30%, 3,949 steps;
  it reproduces 1.686 (chat+distill)/2, the original's value). Reference = the original `aura_chat_distill` re-scored (chat 1.878,
  distill 1.491, story 2.789). Results: `experiments/results/ab_distill/summary.md`, `readable.md`. Run `runs/20261028-051938`.

  | | NEW (s0, s1) | OLD (s0, s1) | ref |
  |---|---|---|---|
  | **arith battery, all 410** | **13% (12, 14)** | 1% (1, 1) | 1% |
  | + 2-digit / − 2-digit | 26% / 45% | 1% / 0% | 2% / 0% |
  | × and ÷ (all cells) | 0-2% | 0-3% | 0-3% |
  | readable maths correct (of 10, greedy) | 2 (3, 2) | 0 | 0 |
  | math_eval exam (of 120, legacy format) | 2 (2, 3) | 2 (1, 4) | 5 |
  | **story val (base 2.6602)** | 2.7814 (**+0.121**) | 2.7890 (+0.129) | 2.7893 |
  | replies with a repeated 4-gram (greedy / sampled) | **56% / 21%** | 5% / 2% | 4% / 1% |
  | mean words per reply | 47 | 14 | 13 |

  TinyChat val (new 3.39 vs old 1.88) and old-distill val are **not comparable** (OLD trains on them). Pilot val favours NEW the same way.
  **Reading:** (1) **arithmetic improved for real** (13% vs 1%, both seeds), but NEW learned the *format* (equation + `Answer:`) and adds
  without carrying ("47 + 38 = 65 / 63"); × and ÷ are still ~0. The legacy exam didn't move (different prompt format). (2) **Story forgetting is equal or
  slightly better for NEW, so no story-mix follow-up** (user's rule). (3) **Chat quality is worse for NEW:** longer, playful replies degenerate at 33M
  (greedy loops like "I'm so sorry, I didn't mean to get mad at you" ×6, similes that make no sense). OLD is short and bland but coherent. So ft_v1 at 33M is not
  a drop-in replacement for `aura_chat_distill`. It's evidence for both the format test and the scaling plan below.

### Scaling plan (PLANNED 2026-10-28, NOT RUN; needs user approval)
**0. Arithmetic-format test first (before any size decision), ~1 GPU-h.** Programmatic data, no teacher: A = plain equation
("47 + 38 = 85" + Answer line), B = scratchpad with the carries written digit by digit, right to left ("ones: 7 + 8 = 15, write 5 carry 1; tens: 4 + 3 + 1 = 8 → 85").
Same problem count (~150k), same ranges, battery problems excluded. Fine-tune from `ab_new_s0` (and `smart_big_10ep` as a check),
2 seeds per arm, ~10 min each, then the battery. **Decision rule:** if B reaches ≥50% on the +/− 2-3 digit cells while A stays ≤30%, the
bottleneck at 33M was *format*: put scratchpad arithmetic into the distill/pretrain mix. If both stay low, it's capacity, and the sweep decides.
**1. Clean pretraining corpus (commercial-safe; no GPT- or closed-model-generated text):**

| source | licence | available | planned share of ~2.2B tokens |
|---|---|---|---|
| FineWeb-Edu (HuggingFaceFW/fineweb-edu) | ODC-By (attribution) | 1.3T | ~55% |
| Cosmopedia v2 (in HuggingFaceTB/smollm-corpus): textbooks, stories, incl. young-children stories | ODC-By; text by Mixtral-8x7B (Apache-2.0) | ~28B | ~30% (replaces TinyStories) |
| FineMath-4+ (HuggingFaceTB/finemath) | ODC-By | 34B (3+) | ~10% |
| Our licence-clean distill data (Qwen2.5-7B, Apache-2.0) + programmatic arithmetic (scratchpad if test 0 wins) | Apache-2.0 / ours | 59M + unlimited | ~5% |

**ODC-By covers the dataset collection, not the underlying web text** (user note 2026-10-28): it licenses the compilation with an
attribution requirement. The pages themselves keep their original authors' rights (FineWeb/FineWeb-Edu/FineMath are CommonCrawl
extracts, also subject to CommonCrawl's terms of use). So ODC-By makes the *collection* usable commercially with attribution, but it
isn't a guarantee for every page in it. Attribute each dataset in the model/corpus card. Cosmopedia's text is model output (Mixtral, Apache-2.0).
Excluded or flagged: **Wikipedia (CC BY-SA 3.0 + GFDL, share-alike)**, OpenWebMath (no licence on its card), Project Gutenberg (PD in the US only, strip
the PG trademark headers), Python-Edu (per-file code licences), TinyStories/TinyChat (GPT-generated; TinyChat is NC). Download ~6-8 GB of parquet shards,
not whole datasets. Record every source + licence in the run config and a corpus card.
**Tokenizer:** train a new SentencePiece BPE on a ~2 GB sample of the new corpus: **24k** vocab, byte fallback, digits split, same specials. The
current 16k was trained on the TinyStories-style mix and fits web text poorly. 32k makes the embedding swamp the small sweep model (32k×384 =
12.3M of a ~26M model); 24k is the compromise. The same tokenizer is used for every size, so the sweep is fair. ~30-60 min CPU.
**2. Scaling sweep (seq_len 512, same corpus and token budget, 16k tokens/step = batch 32 × 512):** sizes (measured param counts, 24k vocab):
S d384 L8 H6 = **23.5M**, M d512 L12 H8 = **50.2M**, L d768 L12 H12 = **103.7M**. Arms: heartbeat **on / off / control** (`live_control`), **3 seeds**
each (the lessons from the heartbeat experiment: one compiled run is noise, and the control is needed to attribute the effect). Estimates scaled from `big`'s measured
295k tok/s and 7.6 GiB, ±25%:

| size | est. tok/s | est. peak VRAM | 200M tokens/run | 9 runs (3 arms × 3 seeds) |
|---|---|---|---|---|
| S 23.5M | ~400k | ~6 GiB | ~8 min | ~1.3 h |
| M 50.2M | ~185k | ~11 GiB | ~18 min | ~2.7 h |
| L 103.7M | ~90k | ~17-18 GiB | ~37 min | ~5.6 h |
| **total** | | | | **~9.5 GPU-h** (two nights, or one long one) |

Measure the real tok/s and VRAM with a 30-step probe per size before the sweep. 200M tokens under-trains L (Chinchilla is ~2B), so compare sizes at
equal tokens and look at the curves, not just the end values.
**3. Research question: does the heartbeat effect hold, grow or vanish with size?** Per size: on vs off vs control val loss on a clean held-out set,
Welch t + exact Mann-Whitney per pair, and the gain (off − on) vs log(params). With 3 seeds per cell, effects below ~2-3× the seed sd won't resolve; report that
honestly. The control arm separates "synchrony helps" from "any phase-coherence loss helps", which is what the 33M result couldn't tell apart.
**4. Full pretrain of the winning size, then distill on top:** e.g. L ~104M on ~2.1B tokens (~20 tokens/param) at ~90k tok/s ≈ **6.5 h** (one night),
through `launch_train.sh` with `latest.pt` crash recovery. Then the distill fine-tune: ft_v1 (34M tokens) at ~90k tok/s ≈ **7 min/epoch**, 2 seeds, plus
`ab_eval` (~3 min/checkpoint), against the 33M results above. **Watch the repetition rate** (the 33M failure) and the battery.
**Code changes needed first (~half a day, no GPU):** `--seq-len` in `train_overnight.py`/`finetune_chat.py` (256 is hard-coded in the dataset, config and
`SEQ`), new size-table entries, a **uint16 memmap token loader** (the current loader keeps the whole token cache in RAM as int64: 2.2B tokens = 17.6 GB, which is
too much for 24 GB WSL), a tokenizer-training script, a corpus download/convert script, and `--max-aura-tokens` 240 → ~480 for future distill data.
**Timeline:** day 1: commit, arithmetic-format test (1 GPU-h), code changes + corpus download + tokenizer (CPU). Nights 1-2: sweep (~9.5 GPU-h).
Night 3: full pretrain (~6.5 h). Day 4: distill fine-tune + evals (~1 h).

### Step 0 and scaling prep: progress (2026-10-28, user-approved)
- [x] Clean corpus **`data/corpus_v1/tok24k`**: **2,122,809,844 train + 10,571,015 val tokens** (uint16, 4.2 GB; built in 3.4 min). FineWeb-Edu
  1210M (57.0%), Cosmopedia v2 660M (31.1%), FineMath-4+ 220M (10.4%), ft_v1 *train* split 32.5M (1.5%: **ran out**, target 5%; the slot can be filled
  with programmatic (scratchpad) arithmetic after step 0). Raw shards in `data/corpus_v1/raw/` (8.41 GB: FineWeb-Edu sample/10BT 000+001,
  Cosmopedia v2 00000-00002, FineMath-4+ 00000-00001).
- [x] Tokenizer **`models/tokenizer_clean24k`**: SentencePiece BPE 24k, byte fallback, digits split, "\n" one token, specials 0/1/2. 48 s, 2.5 GB RAM.
  **4.55 chars/token vs 4.29** for the old 16k on FineWeb-Edu + Cosmopedia (~6% fewer tokens).
- [x] Code: `cosf_mi/data/memmap_dataset.py`, `scripts/corpus/{build_corpus,train_tokenizer}.py`, `train_overnight.py --seq-len --corpus` and sizes
  `s25/m50/l100` (`23a58d6`). `finetune_chat.py --seq-len` added after step 0 (default 256; refuses a window above the checkpoint's
  `max_seq_len`, which is 256 for every 33M checkpoint; 20-step test at 128 OK).
- [x] Step 0 queue (`scripts/step0_queue.sh`, run `runs/20261028-064058`): 4 arith fine-tunes from `ab_new_s0` (plain/scratch × seeds 0,1) → `ab_eval`
  → decoding check (`--bs 16`). Data: `data/arith_format/{plain,scratch}` (same 150k problems; ops + 71k, − 64k, ÷ 8k, × 7k; 63k 3-digit ×
  scratchpads dropped as too long, so the battery's × 3-digit cell can't be learned in this test).
  **Done 08:07, no CUDA error (peak 8,771 MiB with the logger on).** Results: `experiments/results/arith_format/summary.md`, `readable.md`.

  | battery cell (2 seeds) | plain | scratch |
  |---|---|---|
  | + 1 / 2 / 3-digit | 85 / 99 / 96% | 37 / 99 / 85% |
  | − 1 / 2 / 3-digit | 57 / 96 / 79% | 37 / 96 / 81% |
  | × (all cells) | 0-1% | 0-3% |
  | ÷ 1-digit / 2-digit divisor | 14 / 15% | 5 / 0% |
  | ALL 410 | **49%** (51, 48) | 42% (41, 42) |
  | math_eval exam (of 120) | **14** (14, 15) | 1 |

  **Decision rule: neither branch fires.** Plain alone reaches 96-99% on +/− 2-digit and 79-96% on 3-digit (rule: plain ≤ 30%), so it's **not
  format and not capacity at 33M for +/−: it was data.** In the distill A/B, ft_v1 had arith at 22% of tokens, thinned unevenly by the quip filter,
  wrapped in playful text; ~135k clean +/− problems fix carrying. Scratchpad costs 3× the tokens and is worse on 1-digit, ÷ and the exam.
  Why 1-digit is low: **the training set has no 1-digit +/− at all** (+ is 2×2 7k / 3×3 64k; − 2×2 3k / 3×3 61k). Scratch runs its column template on
  "4 + 7" and invents a tens column; plain just guesses. × (7k, 2×2 only) and ÷ (8k, mostly 3-4 ÷ 2 digits) were too thin to test. Scratch
  failures there are **single-digit fact errors** ("5 × 3 + 2 = 13", "48 ÷ 6 = 7"), so the model doesn't know its times tables.
  **Recommendation:** plain format, not scratchpad, for the corpus's arithmetic slot, with full coverage: 1-digit facts and times tables, balanced
  + − × ÷, 1-4 digits. × and ÷ need their own test with ~50k each (≈20 min GPU) before calling them capacity.
  Side effect: the arith-only fine-tune breaks chat (greedy: turn ended 57% / 42%, down from 96%; repeated 4-gram 68% / 77%). Expected for a narrow
  fine-tune; these checkpoints are test probes, not for use. Story val didn't get worse (2.78 → 2.75).
  **Decoding check** (`experiments/results/ab_distill/decoding/summary.md`; ~400 prompts × 2 samples, T 0.7): replies with a repeated 4-gram:

  | | greedy | top-k 40 | top-p 0.9 | top-p 0.9 + rep 1.15 |
  |---|---|---|---|---|
  | NEW (s0, s1) | 57 / 55% | 31 / 34% | 37 / 39% | **6 / 7%** |
  | OLD (s0, s1) | 5 / 5% | 3 / 2% | 3 / 3% | 1 / 1% |
  | ref aura_chat_distill | 4% | 2% | 3% | 1% |

  **Most of the looping is decoding:** the repetition penalty takes NEW from 57% to 6%, and plain sampling halves it. **The content is still
  the model's problem:** the loop-free replies are simile soup ("Volcanoes are like tiny, invisible friends who can't do it!"). NEW stays ~10×
  OLD under every setting, and its replies are ~4× longer (59 vs 15 words), so per-word it's closer. **Side effect of rep 1.15:** NEW's "?"
  endings on chat/explain jump from 6% to 64%. The penalty (over prompt + reply) pushes away from "." after a few sentences, so it changes the
  reply's shape and isn't a free fix.
- [x] **× / ÷ data-vs-capacity test** (2026-10-28, run `runs/20261028-164812`; `scripts/arith_md_queue.sh`, results
  `experiments/results/arith_md/summary.md`). Data: `scripts/distill/build_arith_plain.py`, plain format, mix × 40 / ÷ 40 / + 10 / − 10%, each
  operand 1-4 digits (÷ = (a × b) ÷ b: quotient and divisor 1-4 digits). Same steps in both arms: **25k unique × 4 epochs vs 100k × 1 epoch**,
  seeds 0/1, from `arith_plain_s0`. Eval: 15 problems per digit cell, **fit** (problems both arms trained on) vs **held** (never generated).
  Rule decided beforehand: capacity if it can't fit its own training problems and 4× unique data doesn't help; data if held-out rises with 100k.
  **Result: capacity (for plain-format, one-pass answers at 33M).**
  - × with both operands ≥ 2 digits: fit 0-20%, held 0-7%, in every arm. It can't reproduce problems it saw 4×. 25k vs 100k: no difference.
  - × 1 × n-digit: held 2x1 20-40%, 1x2 27-40%, 1x3 7-20%, 1x4 0%.
  - **Times tables learned:** × 1x1 80-87%, battery × 1-digit 90-100% (was 0%). These are memorized facts (always trained).
  - ÷: 1-digit quotient ~100% (any divisor, it's estimation); 2-digit quotient fit 13-47% / held 20-60%; ≥ 3-digit quotient ~0. Fit ≈ held, 25k ≈ 100k.
  - Battery ALL 56% → 61-65%. Cost: with + − at only 20% of the mix, + 3-digit fell 98 → 85-95% and − 3-digit 90 → 72-80%.
  - Open: **scratchpad × now that the facts are known.** Step 0's scratch × failures were single-digit fact errors, and the facts are now
    learned, so scratch + facts could work where plain can't. Untested. The sweep's final models (`final_model.pt`) can also take the battery
    per size (the corpus has the same "You: … Aura: …" arithmetic), which tests capacity vs size directly.
  - Caveat: `arith_md_eval.py` and `ab_eval.py` score the same checkpoint differently (`arith_plain_s0`: battery 56% vs 51%, + 1-digit 100% vs 85%;
    max_new 40 vs 120, different batch composition). Compare numbers only within one script's table. Whether padding/batch composition changes
    greedy answers is unchecked and worth a look.
- [x] **Corpus arithmetic slot** (2026-10-28): `data/arith_corpus_v1` = `build_arith_plain.py --n 1830000 --seed 1`: + 472k, − 424k, × 467k,
  ÷ 467k, 1-4 digits; 199k are 1-digit facts (~11%, each fact repeated hundreds of times); non-facts unique; every multi-digit eval triple and every
  `heldout()` triple excluded. **1-digit facts are NOT held out** (they're needed to teach the tables; the battery reserves 30 of the 64 × facts and 30
  of the 45 one-digit − facts), so from now on the battery's 1-digit cells measure fact recall, not generalization. `--exclude-facts` restores the
  strict rule. New corpus **`data/corpus_v1/tok24k_v2`**: 2,199,809,974 train + 10,955,119 val tokens. FineWeb-Edu 1210M (55.0%), Cosmopedia 660M (30.0%),
  FineMath 220M (10.0%), aura_distill 32.5M (1.5%, all of it), **aura_arith 77.0M (3.5%)**. `tok24k` (without arithmetic) kept.
- [ ] **Scaling sweep, launched unattended 2026-10-28 evening** (`scripts/sweep_queue.sh`, user-approved; results
  `experiments/results/sweep/{results.tsv,summary.md}` via `experiments/sweep_report.py`). s25 / m50 / l100 × on / off / control × seeds 1-3 =
  27 runs, 200M tokens each (12,207 steps, batch 32 × seq 512), lr 6e-4, warmup 500, **cosine to 0 at the budget** (new `--decay-steps`), val on the
  whole val split, `--save-final` keeps `checkpoints/sweep/<size>_<arm>_s<seed>/final_model.pt`. Data order is fixed per epoch, so a seed fixes
  the init and the arms are paired by seed. Probe (60 steps, seq 512): **s25 23.47M, 6.8 GiB, ~310k tok/s; m50 50.23M, 11.3 GiB, ~165k; l100
  103.65M, 16.2 GiB, ~88k.** Heartbeat-off is not slower at these sizes. Estimate ≈ 11-12 h. Stops on the first failure (a CUDA error is
  labelled) and never retries. Relaunching skips finished runs.
- **CUDA driver error in the first step-0 queue (run `runs/20261028-062703`, crashed ~06:34:56):** `torch.AcceleratorError: CUDA error: unknown
  error` / `CUDA_ERROR_UNKNOWN from cuMemcpyDtoHAsync_v2` in `ab_eval.generate` (decoding check, `new_s0`, top-p + repetition-penalty setup, batch 64).
  - **Peak VRAM right before:** not continuously sampled (that queue had no logger). Spot readings: **32,035 MiB at ~06:29 and 32,028 MiB at ~06:33
  of 32,607 MiB (98.2%)**, so the card was full for at least the ~5 min before the crash. At launch (06:27) it was 1,512 MiB. After the crash it was 589 MiB.
  - **Did memory exceed 32 GB?** No. It hit the ceiling: the PyTorch allocator logged 2 failed allocations (06:27:45: 644 MB, 06:28:15: 889 MB,
  both "free: 0, total: 34,162,016,256 B"), freed its cache and retried. "Failed with OOM" means the driver refused, not that it spilled to system RAM.
  - **Other GPU processes:** none in WSL: no Ollama model loaded (`ollama ps` empty), no other training or vLLM job. Running alongside were
  CPU-only jobs: the corpus download, the tokenizer sample/training (ended 06:34) and a CPU tokenization test. On Windows, ~22 desktop processes held
  GPU contexts (dwm, Wallpaper Engine, Discord, VS Code, Steam webhelper, Docker Desktop UI…) but no game. Windows reports their VRAM as N/A,
  so their share can't be measured. No `nvlddmkm`/display-reset event in the Windows System log for the 30 min around it.
  - **Cause: likely, not proven,** memory exhaustion/fragmentation in `generate` (full-context fp32 logits, 64 × 256 × 16k ≈ 1 GB per step, a new
  size every step), ending in a driver fault. Puzzle: the same code at batch 64 ran the A/B eval earlier with 0 allocator warnings. GPU healthy afterwards.
  - **Mitigation:** the decoding check now runs last, at batch 16, with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (also on the arith eval). A
  2-second VRAM/util/temp/power sampler now logs to `runs/20261028-064058/vram.csv`.
  - **Rule (user, 2026-10-28): if a CUDA driver error recurs, stop the queue and tell the user; don't retry.** (The queue already stops on any
  error via `set -e`; no automatic relaunch.)
