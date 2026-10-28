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

**Resume point (2026-10-27 evening):** `main` = `3b7ca8e` (heartbeat-ablation, heartbeat-control and sync-cleanup merged and pushed;
worktree `~/projects/aura-syncfix` removed). The control result is **corrected below**: control vs heartbeat isn't significant.
**New goal: distillation.** The investigation is done (see "Distillation investigation" below); **the pilot plan waits for user approval.
Don't start generation before that.**

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
- Never commit checkpoints, data, logs, caches, `memory/jarvis/` or `.claude/settings.local.json` (see `.gitignore`).
  The repo is public: scan new files for secrets before pushing.
- One GPU: don't run the 32b teacher and training at the same time.
- **Compiled runs aren't bit-reproducible** (`--compile` is the default): the same seed and config diverge after ~100-200 steps.
  Eager runs with the same seed are bit-identical. Published or compared results must use `--no-compile`, or report
  several seeds (mean ± spread), never a single compiled run.
