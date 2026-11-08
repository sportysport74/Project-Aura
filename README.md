# AURA (COSF-MI): a small language model, tested in the open

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

AURA is an experimental decoder-only language model built from scratch in PyTorch on a single RTX 5090. It began as
an **oscillator model**: every projection inside the transformer was a phase-oscillator layer, and training added a
**Kuramoto-style synchrony loss** (the "heartbeat"). In a pre-registered comparison in November 2026 the oscillator
backbone lost to a standard transformer, so AURA now runs on a standard backbone. Oscillator ideas come back only
as optional modules, each of which has to beat that backbone on its own test.

This is a one-person research project. Its goal is to find out, with controlled experiments, which ideas actually
help a language model and which don't, and to report both.

**Measured results live in [`COSF-MI/RESULTS.md`](COSF-MI/RESULTS.md); current status in one page in
[`PROGRESS.md`](PROGRESS.md); how the project got here in [`HISTORY.md`](HISTORY.md); the long-term vision in
[`COSF-MI/experiments/VISION.md`](COSF-MI/experiments/VISION.md).** This README keeps results and vision apart on purpose.

## What AURA is today

- **Model ("vanilla2"):** a standard decoder-only transformer: linear projections, token embeddings scaled by √d,
  GELU, GPT-style init. The current base has ~104M parameters, context 512, and a 24k SentencePiece tokenizer. A
  ~327M model with context 1024 is in short test runs (see *In progress*).
- **Pretraining data:** about 10B tokens of licence-clean text: FineWeb-Edu (55%), Cosmopedia v2 (30%, synthetic
  text written by Mixtral, an open-weights model), FineMath-4+ (10%) and programmatic arithmetic (3.5%). No text from
  closed models.
- **Chat:** fine-tuned on data written by an open-weights teacher (Qwen2.5-7B-Instruct, Apache-2.0). At chat time it
  retrieves passages from a 6M-passage store built from the same corpus, passes messages through a small gate
  (garbled input, chit-chat) and keeps one previous turn of conversation.
- **What it can do:** fluent short text and stories, simple addition and subtraction, and answers to factual
  questions that are right more often with retrieval than without. It still makes up facts, especially for questions
  it can't answer. It is a research model, not an assistant.

## Results so far

Full details, seeds, controls and caveats are in [`COSF-MI/RESULTS.md`](COSF-MI/RESULTS.md). In short:

**1. Oscillator projections vs a standard transformer: the standard transformer wins.** Pre-registered; same model,
data, schedule and seeds, only the projections differ (params within 0.2%); 3 seeds per size, 200M tokens, val loss
(cross-entropy, lower is better):

| size | standard (vanilla2) | oscillator AURA | gap |
|---|---|---|---|
| 23.5M | 3.431 | 3.760 | +0.329 |
| 50.2M | 3.255 | 3.554 | +0.299 |
| 103.7M | 3.141 | 3.412 | +0.271 |

The gap is about 100× the seed spread at every size and shrinks only slowly with size, and the standard model trains
1.9–2.3× faster. Per the rule written down before the runs, the backbone switched to vanilla2.

**2. The new base beats the old oscillator curve everywhere.** vanilla2 at 104M on the same 10B tokens, same protocol
and data order (one training run per point; confidence intervals are over the 100 questions, not seeds):

| tokens | val loss, oscillator → **vanilla2** | chat factuality after the same fine-tune |
|---|---|---|
| 2.2B | 2.877 → **2.644** | 0.156 → **0.246** |
| 5B | 2.774 → **2.579** | 0.180 → **0.260** |
| 10B | 2.680 → **2.518** | 0.206 → **0.302** |

It got there in about half the GPU time (17 h vs ~27.5 h for the 10B pretrain).

**3. Retrieval works.** On the same 100 factual questions, a retrieval fine-tune with the corpus store raised chat
factuality from 0.308 to **0.512** and cut false claims from 33% to **13%**. A Simple English Wikipedia store did
worse than the corpus store, so the corpus store is the default.

**4. A gate and one turn of memory help.** The gate cut inappropriate answers to off-topic messages from 42% to
**27%** (garbled input: 52% → 12%) with no loss of factuality. Keeping the previous turn raised the score on
follow-up questions from 0.090 to **0.353**.

**5. Inside the oscillator model, a phase-coherence loss helped.** 3 sizes × 3 seeds, 200M tokens, val loss, on the
old embedding:

| size | heartbeat on | off | control |
|---|---|---|---|
| ~25M | 4.066 | 4.127 | 4.073 |
| ~50M | 3.903 | 4.009 | 3.942 |
| ~104M | 3.757 | 3.877 | 3.786 |

The *control* is an identical loss that targets a random phase pattern instead of synchrony. It recovers most of the
gain, so the defensible claim is: **a phase-coherence loss of this shape helped the oscillator model, and synchrony
specifically may add a smaller extra on top.** With 6 seeds at ~104M the heartbeat beats the control, but the control
arm was unstable across seeds, so that part is not settled. The heartbeat also helped on the fixed embedding
(November 3). It is defined on oscillator phases, so it doesn't apply to the vanilla2 backbone as it stands.

## In progress

**Aura v3:** ~327M parameters, context 1024, ~15B tokens. A scaling check (327M beat 104M on both seeds at 200M
tokens) and a learning-rate check run first; the full pretrain follows. The plan, with its rules written before the
runs, is in [`COSF-MI/experiments/plans/aura_v3_base.md`](COSF-MI/experiments/plans/aura_v3_base.md).

## What is not established

These ideas appear in older documents in this repository. None of them is a measured capability today:

- **Oscillators as a better backbone.** Tested and lost (result 1). Oscillators as *modules* on a standard backbone
  are still open. One such test, within-token settling, made the model worse (+0.035 val) at 3× the compute.
- **Chirality as learned handedness.** The ±1 chirality parameters keep the random sign they were initialized with
  (99.95% unchanged after training), and without an input-dependent gate they can be folded exactly into the other
  oscillator parameters. A per-token gate showed no gain at tiny scale.
- **Sleep and consolidation without forgetting.** Sleeps that change the weights failed. Replay sleep, which
  rehearses old data, cut forgetting by about 76% in a small test and is still used during chat fine-tuning, but it
  is rehearsal, a known technique, not consolidation. Older figures such as "68.75% consolidation success" came from
  small early runs and are not current results.
- **Thinking by settling, self-diagnosis, memory between conversations, golden-ratio dynamics as a source of
  capability.** These are research directions. Each one has an idea, its current state in the code, and the experiment
  that would test it in [`COSF-MI/experiments/VISION.md`](COSF-MI/experiments/VISION.md).

## Getting started

Work happens in `COSF-MI/`. The repository root holds older files from earlier versions (Windows `.bat` scripts, the
Jarvis CLI, an older duplicate `src/`) that are kept for history. The commands below are for Linux (development and
training run on Ubuntu under WSL2); native Windows is untested.

```bash
cd COSF-MI
uv venv && source .venv/bin/activate      # Python 3.12, PyTorch with CUDA
uv pip install -r requirements.txt
python -m pytest tests                    # unit tests
python train_overnight.py --help          # training options (model size, projection type, heartbeat, corpus, precision)
python chat_with_aura.py <checkpoint> --store corpus   # chat with retrieval (add --show-sources to see the passages)
```

The vanilla2 backbone is `--projection linear --emb-scale sqrt --ffn-activation gelu --linear-init gpt`. Data,
retrieval stores and checkpoints are not included (they are large and gitignored). `CLAUDE.md` describes the current
layout, commands and experiment history in more detail.

## Repository map

- `COSF-MI/src/python/cosf_mi/models/cosf_oscillator.py`: the model that is trained, both the oscillator and the
  vanilla2 variants (attention, oscillator layers, heartbeat loss).
- `COSF-MI/src/python/cosf_mi/retrieval.py`, `chat_gate.py`: retrieval and the chat gate.
- `COSF-MI/train_overnight.py`: pretraining. `COSF-MI/finetune_chat.py`: chat fine-tuning and distillation.
  `COSF-MI/chat_with_aura.py`: chat.
- `COSF-MI/experiments/`: evaluations (knowledge probe, factuality, arithmetic, gate, retrieval) and sweep scripts;
  `plans/` holds pre-registered experiment plans, written before the runs; `results/` holds their outputs.
- `COSF-MI/src/rust/cosf_core/`: a standalone Rust crate (φ-shell decomposition, phase coherence). It is not used by
  training.

## How results are reported

- A result needs several seeds, a baseline and, where possible, a control. Single runs are reported as single runs.
- Success criteria are written down before the data comes in. When a criterion changes afterwards, the change and the
  reason are recorded.
- Negative results stay in. The plain transformer won, and that is in `RESULTS.md` too.

## License and credits

MIT, see [LICENSE](LICENSE). Training data and teachers are chosen so that commercial use stays possible; teacher
data comes from open-weights models whose licenses allow training on their outputs, and the teacher and license are
recorded per run. The Wikipedia retrieval store (CC BY-SA) is used only at chat time, never as training data. Built
by Jon (sportysport74, Nodal Systems LLC), with AI coding assistance.

Background: the Kuramoto model of coupled oscillators; AKOrN (Miyato et al., *Artificial Kuramoto Oscillatory
Neurons*, ICLR 2025) for oscillator neurons in deep networks.
