# AURA (COSF-MI)

Custom decoder-only transformer with Kuramoto-style oscillators ("COSF"), trained on a
stories/code/math/textbook mix, then chat-fine-tuned and distilled from an open-weights teacher.
GitHub: https://github.com/sportysport74/Project-Aura (**public**, MIT). The original Windows copy
is at `C:\Users\fatba\COSF-MI` and is no longer the working copy.

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
  Only `aura_chat_{distill,mid,mid2,smart,smart10,tiny}` were copied to Linux. `smart_big_10ep`
  (the base for chat fine-tunes) and older runs are still only on `C:\Users\fatba\COSF-MI\COSF-MI\checkpoints`.

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
  checkpoint from its list, starting with `aura_chat_smart10`. **`aura_chat_distill` isn't in that list**, so pass it explicitly.
- Tests: `.venv/bin/python -m pytest tests`

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

**Resume point (paused 2026-10-26, user away; GPU left idle, no jobs, no Ollama model loaded):**
Branch `sdpa-bf16` (not merged, not pushed) has SDPA + bf16 + torch.compile as defaults in `train_overnight.py`,
measured over 500-step runs only. Next steps, in order:
1. [x] **Heartbeat: kept as is** (live, strength 0.1, target 0.3; user decision 2026-10-26). It works as a floor on R:
   without it R stays ~0.01, it lifts R to ~0.3 by step 60, is 0 while R > 0.3, and switches back on when R drifts down
   (the old tiny 10k-step A/B ended at R 0.246). Val cost is nil (3.613 vs 3.592-3.613 without), but there's no measured
   benefit either. Open idea: an A/B with and without it on a capability (math eval or forgetting), not val loss.
   **Ablation, 1 epoch `big` (2026-10-26):** `--heartbeat-strength 0` (`runs/20261026-213533`) against arm 1 (`runs/20261026-182317`),
   same seed and config, both compiled. Without the heartbeat, R settles at **0.004** (0.002-0.006 from step 100 on) instead of
   ~0.30. Val **3.1530 vs 3.1094 (+0.044 worse)**, avg train loss 3.6024 vs 3.5404, and the train curve is higher at every point
   from ~step 500 (smoothed gap 0.06 mean, 0.08 max). So at `big` scale the floor helps, unlike the old tiny 10k-step A/B.
   Caveat: one compiled run per arm (not bit-reproducible). The 1-epoch compile-vs-eager val gap was 0.007, so +0.044 is ~6× that
   and consistent across the whole curve, but confirm with 2 more seeds per arm before publishing.
   Unexplained: the no-heartbeat run was slower (181.5k vs 219k tok/s, with the faster sync-free Julia code). Check
   whether the GPU was shared at the time, or whether strength 0 changes the compiled graph.
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
3. **Sync cleanup** (open item below): vectorize the Julia check and drop the per-step `.item()` calls. Re-profile first.
Updated 2026-10-26.

Done:
- [x] WSL RAM limit: `free -h` shows 23 GiB (`.wslconfig` memory=24GB, mirrored networking).
- [x] `.claude/settings.local.json`: 48 Windows-era permissions cut to 14 Linux-relevant ones (gitignored, not in the repo).
  Dropped `timeout:*` (it let any command through unprompted). `git push:*` and `pkill:*` are still allowed; consider removing them.
- [x] `.wslconfig` has `[experimental] hostAddressLoopback=true` (2026-10-26), in effect since the reboot.
- [x] Smoke test passes. Baseline (tiny, batch 32×256, 200 steps, fp32, hand-written attention):
  loss 8.48 → 6.72, ~77k tok/s, ~106 ms/step, **5.24 GiB peak VRAM**.

- [x] Docker GPU test (2026-10-26): `docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi` sees the
  RTX 5090 (32607 MiB, driver 617.42). Docker Desktop doesn't autostart after a reboot; while it's down, `docker` is missing
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
- [ ] Remaining syncs (7 per step): `drift.item()` and the
  heartbeat `coherence.item()` in forward, and the bookkeeping `.item()` calls. Original note: 48 `.item()` in `apply_julia_correction` (done), `drift.item()` and the
  heartbeat `coherence.item()` in forward (also graph breaks under compile at `cosf_oscillator.py:1293`, `:1383`), and the bookkeeping
  `.item()` calls every step. Re-profile after compile first: the sync share of the ~75 ms step is now larger.
- [ ] Data loader workers/pinned memory: ≤1-2%, last.
- [ ] Spend the freed memory: seq_len > 256 or a bigger model. Re-run the batch-size probe on `sdpa-bf16` first.
- [ ] Param count mismatch: `big` reports **33.5M** params at start-up, while Known state says 41.9M. Check against a checkpoint.

## Rules
- Distillation teachers must be open-weights with a license that allows training on outputs; record teacher + license per run.
- Never commit checkpoints, data, logs, caches, `memory/jarvis/` or `.claude/settings.local.json` (see `.gitignore`).
  The repo is public: scan new files for secrets before pushing.
- One GPU: don't run the 32b teacher and training at the same time.
- **Compiled runs aren't bit-reproducible** (`--compile` is the default): the same seed and config diverge after ~100-200 steps.
  Eager runs with the same seed are bit-identical. Published or compared results must use `--no-compile`, or report
  several seeds (mean ± spread), never a single compiled run.
