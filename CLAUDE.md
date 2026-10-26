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
- Smoke run on Linux (tiny, 11.5M params, batch 32×256): about 100 ms/step, ~80k tok/s, 5.2 GiB peak VRAM.
  That's only ~5.6 TFLOPS: attention is a hand-written softmax (`cosf_oscillator.py:475`, `:1008`) in fp32, with no
  SDPA and no bf16 autocast. Flash SDPA works on this GPU (sm_120), so this is the main speed lever.
- Tests: 66 pass. Failing: `tests/test_constants.py` (imports `E_8_6`, removed from `core/constants.py`)
  and `test_phi_decomposition.py::test_assign_shells` (expects -1 below base, gets 0).
- `cosf_core` Rust test `test_convergence` fails: it asserts φ¹⁷ ≈ e^8.6, but φ¹⁷ = 3571 and e^8.6 = 5432
  (34% apart). `core/constants.py` docstring line 10 repeats the wrong value. Its `CONVERGENCE_RATIO`
  is 1.0 by construction (φ^(ln C / ln φ) = C), so it doesn't check anything.

## Setup status
Linux/WSL migration checklist. **Read this first; tick items off (with date) as they get done, add new ones as they appear.**
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
- [ ] **Real speed lever:** bf16 autocast + `F.scaled_dot_product_attention` (`is_causal=True`, `scale=coupling/√d_k` for the
  φ block) in both attention blocks (`cosf_oscillator.py:468`, `:999`). Keep the phase/coherence maths in fp32. Then spend
  freed memory on seq_len > 256 or a bigger model. Re-run the probe afterwards.

## Rules
- Distillation teachers must be open-weights with a license that allows training on outputs; record teacher + license per run.
- Never commit checkpoints, data, logs, caches, `memory/jarvis/` or `.claude/settings.local.json` (see `.gitignore`).
  The repo is public: scan new files for secrets before pushing.
- One GPU: don't run the 32b teacher and training at the same time.
