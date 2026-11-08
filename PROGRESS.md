# AURA: where things stand (2026-11-07)

One page. Measured results, with their evidence, are in [`COSF-MI/RESULTS.md`](COSF-MI/RESULTS.md); ideas are in
[`COSF-MI/experiments/VISION.md`](COSF-MI/experiments/VISION.md); plans (each written and committed before it ran) are in
[`COSF-MI/experiments/plans/`](COSF-MI/experiments/plans/). The top-level [`README.md`](README.md) is the overview; [`HISTORY.md`](HISTORY.md) tells how the project got here.

## Right now

- **Running (since 2026-11-07 13:29):** Aura v3 steps 1-2, a scaling check (104M vs 330M at context 1024, 2 seeds) and an lr check at 330M,
  ~5-6 h. Results land in `COSF-MI/experiments/results/v3_checks/report.md`.
- **Next:** the Aura v3 base, ~330M params, context 1024, ~15B tokens
  ([`aura_v3_base.md`](COSF-MI/experiments/plans/aura_v3_base.md)). The plan is ≈ 86 GPU-h. The 15B-token pretrain
  (step 3, ~74 GPU-h) waits for Jon to see the step 1-2 results.
- **Best chat setup:** `checkpoints/rag/s0` (104M) with the corpus retrieval store, the gate and one turn of memory:
  ```
  cd COSF-MI && .venv/bin/python chat_with_aura.py checkpoints/rag/s0/best_model.pt --store corpus [--show-sources] [--no-gate]
  ```
  The script's default checkpoint (no argument) is still `aura_chat_distill`. Switching it is Jon's decision.

## What AURA is today

A **standard decoder-only transformer** ("vanilla2": linear projections, scaled embedding, GELU, GPT init). It has 104M params, 512 context,
and a 24k SentencePiece tokenizer, and was pretrained on 10B licence-clean tokens:

| source | share | licence |
|---|---|---|
| FineWeb-Edu | 55% | ODC-By |
| Cosmopedia v2 | 30% | ODC-By, Mixtral output |
| FineMath-4+ | 10% | ODC-By |
| programmatic arithmetic | 3.5% | ours |

It was fine-tuned on Qwen2.5-7B-Instruct teacher data (Apache-2.0) and adds retrieval over a 6M-passage corpus store at chat time.

The original **oscillator architecture** (Kuramoto phase projections, chirality, heartbeat) was tested against this baseline and lost. The
oscillator ideas now live on as optional modules that must each beat vanilla2 (VISION.md §0).

## Main results (all pre-registered unless noted; details in RESULTS.md)

| date | experiment | verdict |
|---|---|---|
| 10-26/27 | heartbeat (coherence loss) on oscillator AURA, 33M | helps vs off (+0.031 val); a random-phase control ties it, so it's unclear whether synchrony itself helps |
| 10-28 | arithmetic format test (plain vs scratchpad) | plain wins; + and − were limited by data. × and ÷ are capacity-limited at 33M |
| 10-29 | scaling sweep 25M/50M/100M × heartbeat on/off/control | most of the gain comes from any phase-coherence loss |
| 10-31 → 11-02 | 10B-token pretrain of oscillator l100 | knowledge probe rises 2.2B → 5B → 10B (CIs exclude 0) |
| 11-02/03 | **oscillators vs a standard transformer** (3 sizes × 3 seeds) | **the standard transformer wins by 0.27-0.33 val loss at ~2× the speed: the backbone switched to vanilla2** |
| 11-03 | heartbeat retest on the fixed embedding | the heartbeat stays (inside oscillator AURA only) |
| 11-04 | criticality-controlled settling on vanilla2 | makes it worse (+0.035) at 3× the cost; not used |
| 11-05 | **new base: vanilla2 l100 on the same 10B tokens** | better than the old curve everywhere: val 2.518 vs 2.680 at 10B; chat factuality 0.302 vs 0.206 |
| 11-06 | probe confound check; lr 3e-4 vs 6e-4 | the flat knowledge probe isn't the schedule; 6e-4 stays |
| 11-06 | **retrieval** (corpus store, RAG fine-tune) | works: factuality 0.310 → 0.376, false claims 32% → 25%. The Wikipedia store is worse than the corpus store |
| 11-07 | retrieval re-score (sources not penalized for repetition) | the same verdicts, larger: **0.308 → 0.512, false claims 33% → 13%** |
| 11-07 | **retrieval gate, multi-turn, answer entropy** | gate G1 cuts bad off-topic answers 42% → 27% at no factuality cost (on by default in chat). One turn of history lifts follow-ups 0.090 → 0.353. Entropy predicts false claims only weakly (AUROC 0.68) |

## Known open problems

- **The gate replies "I'm not sure what you mean" to "hi" and "lol":** the garble check runs before the chit-chat check. A reorder needs its own eval.
- **Unanswerable questions** ("population of a small village", "my dog's name") still get made-up answers about 63% of the time. The v3
  fine-tune adds refusal data for this.
- **No LLM grader** (Qwen 7B or 32B) passed the hand check on off-topic answers, so that metric is hand-graded (blind).
- **Context is 512**, which is too small for sources plus history plus an answer. v3 moves to 1024 (next-base checklist in `CLAUDE.md`).
- **Arithmetic:** × and ÷ beyond one-digit facts are near 0 at 104M.
- **Hardware:** two Windows GPU driver faults (nvlddmkm 153) happened under sustained load in October/November, with none since. HWiNFO and
  telemetry logging run during long jobs.

## Where things live (`COSF-MI/`)

| what | path |
|---|---|
| trainer (pretrain) | `train_overnight.py` |
| fine-tune | `finetune_chat.py` |
| chat | `chat_with_aura.py` |
| model | `src/python/cosf_mi/models/cosf_oscillator.py` (vanilla2 = `--projection linear …`) |
| retrieval and gate | `src/python/cosf_mi/retrieval.py`, `src/python/cosf_mi/chat_gate.py` |
| evals | `experiments/` (`factuality_*`, `knowledge_pmi.py`, `gate_*`, `retrieval_*`, `ab_eval.py`) |
| results | `experiments/results/<experiment>/` (committed). Checkpoints and data are local only (gitignored) |
| run infrastructure | `scripts/lib_train.sh` (watchdog, telemetry, pause), `pause_train.sh` / `resume_train.sh`, `supervise.sh`, `notify.sh` (phone alerts) |
| working rules and setup log | `CLAUDE.md` (repo root) |

Working rules: pre-register decision rules before running, smoke-test every phase of an unattended chain, run at a 450 W GPU cap, commit
results with repo-relative paths, and use only licence-clean data and teachers (commercial use is possible under Nodal Systems LLC).
