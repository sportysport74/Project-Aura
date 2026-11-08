# How AURA got here

AURA didn't start as one idea. It went through several names (Jarvis, COSF, COSF-MI, AURA), several goals and
many small experiments. Some approaches worked, some didn't, and some looked like they worked until a later audit
showed why. This file keeps that path in order, because the current design only makes sense in light of it.

Measured results that still hold for the current model are in [`COSF-MI/RESULTS.md`](COSF-MI/RESULTS.md). The
original documents from each phase are kept in the repository; the links point to them.

## Phase 0: COSF Jarvis (by January 2026)

**Goal:** a personal AI assistant built on COSF math.

- A character-level COSF transformer (7 layers, 384 dims) trained on Shakespeare.
- Around it, an agent design: an intent router (chat, system task, knowledge search, math), a ChromaDB memory with
  φ-weighted retrieval, and a "Julia stability governor" that watched activations for blow-ups.
- A Rust crate, `cosf_core`, for φ-shell decomposition and phase coherence.

**What came out of it:** the assistant was parked; the model underneath became the focus. The ideas that survived
are the stability bounds (still in the oscillator model as the Julia phase-velocity bound) and the goal of an AI with
memory, which is now a research direction in `VISION.md`.
Docs: [`JARVIS_CLI_READY.md`](JARVIS_CLI_READY.md), [`COSF-MI/JARVIS_ARCHITECTURE.md`](COSF-MI/JARVIS_ARCHITECTURE.md).

## Phase 1: COSF-MI, oscillators, chirality and sleep (January 2026)

**Goal:** replace linear layers with phase oscillators, y = A·sin(Ω·x + Θ), and add biology-inspired mechanisms.

- Oscillator layers with φ-scaled frequencies and golden-angle phases; homeostatic decay pulling frequencies toward
  φ; learnable ±1 chirality; multi-intent "bridge word" data ([STORY] vs [CODE] meanings of words like *branch*).
- Tiny model (9.4M parameters) on TinyStories.
- **Sleep:** the first sleep reset wiped learned phases and doubled the loss (4.03 → 8.08). "Gentle sleep" kept
  phases and chirality and only reset fatigue; it measured 68.75% "consolidation success" over 16 sleeps.
- **January 8: AURA Level 1, "the heartbeat"**: a Kuramoto coherence loss to synchronize the oscillators.

Docs: [`STATUS_2026_01_05_AURA_LEVEL1.md`](STATUS_2026_01_05_AURA_LEVEL1.md),
[`COSF-MI/GENTLE_SLEEP_GUIDE.md`](COSF-MI/GENTLE_SLEEP_GUIDE.md),
[`COSF-MI/TRAINING_ANALYSIS_2026_01_05.md`](COSF-MI/TRAINING_ANALYSIS_2026_01_05.md),
[`RESONANT_ARCHITECTURE.md`](RESONANT_ARCHITECTURE.md), [`HOMEOSTATIC_DECAY.md`](HOMEOSTATIC_DECAY.md).

## Phase 2: the October audit (October 3–8, 2026)

After bug fixes, a 100-epoch tiny run reached **val loss 1.648** (the January runs: 6.30). Then the January
features were checked one by one against the code ([`COSF-MI/STATUS_2026_10_08.md`](COSF-MI/STATUS_2026_10_08.md)):

- **The heartbeat was synchronizing the wrong thing:** the fixed phase-offset parameters, not the live per-token
  phases. That's why it reached R = 0.975.
- **The January "drift reduction" was weight collapse:** sleep pulled every frequency halfway to φ, and the weight
  spread fell from 1.618 to 0.013 after seven sleeps. "Consolidation success" was measured in a way that is mostly
  noise.
- **The model code had never been committed** (a `.gitignore` rule matched it). It was rebuilt and tracked.

This was the turning point: from here on, every idea gets an A/B test with a fixed seed before it is kept.

## Phase 3: A/B rounds on the tiny model (October 8–9, 2026)

Tiny model, 10,000 steps, fixed seeds ([`COSF-MI/STATUS_2026_10_08_EVENING.md`](COSF-MI/STATUS_2026_10_08_EVENING.md)):

- **The biggest single gain came out of the sleep investigation.** The golden-angle phases started at up to
  ~2,400 rad, and weight decay kept rotating them. Wrapping them into [0, 2π) improved val loss by **0.61**.
- **The live heartbeat** (R on the real per-token phases, held at a floor of 0.3): **−0.043 val over 3 seeds**, and
  every heartbeat seed beat every baseline seed. Pushing R toward 1 made things worse: moderate sync helps, full
  sync hurts.
- **Replay sleep works.** Train on stories, then on Shakespeare. Replaying old story batches during sleep cut
  forgetting by **about 76%** (+0.33 vs +1.41, 2 seeds), and a control that replayed recent data forgot as much as no
  sleep. It's rehearsal, a known technique, but the first sleep that did a measurable job. It is still used to keep
  story skills during chat fine-tuning.
- **Sleeps that move the weights** (toward φ, toward an average, decay) did not beat gentle sleep; the φ sleep was
  destructive (+2.6 val).
- **Context-dependent chirality** (a per-token gate that can reverse handedness): no gain, and the bridge-word probe
  found no meaning separation (rounds 3 and 5). Parked.
- **Scale contest:** batch size 32 and a 23M "mid" model beat the old recipe at equal GPU time.

## Phase 4: from a story model to a chat model (October 9–25, 2026)

- Larger models (23M "mid", then 41.9M "big"), a mixed corpus (web, wiki, code, math, stories) and a 16k tokenizer.
- Chat fine-tuning with replay sleep to keep storytelling; a repetition fix in the chat loop.
- **Teacher distillation:** Qwen2.5 (open weights) wrote math and dialogue data, with math answers checked in a
  sandbox. The math exam went from 2/120 to 5/120: the model learned the answer template, not the arithmetic.

Docs: [`COSF-MI/STATUS_2026_10_11.md`](COSF-MI/STATUS_2026_10_11.md),
[`COSF-MI/STATUS_2026_10_25.md`](COSF-MI/STATUS_2026_10_25.md).

## Phase 5: Linux, speed and controls (October 25–27, 2026)

- Moved to WSL2 on an RTX 5090. SDPA attention, bf16, `torch.compile` and removing GPU syncs took training from
  ~75k to **~295k tokens/s**.
- **Heartbeat ablation at 41.9M, 3 seeds:** off was +0.031 val worse.
- **First control experiment:** the same loss aimed at a random phase pattern instead of synchrony. With the first
  seeds it looked as if the control did *not* reproduce the gain, pointing at synchrony itself. With 6 control seeds
  the control landed between heartbeat and off, and its gap to the heartbeat (+0.015) was not significant (p = 0.24).
  So this size could not tell "synchrony helps" from "any phase-coherence loss helps", which is what the scaling
  sweep in Phase 6 was designed to separate.

Details: [`CLAUDE.md`](CLAUDE.md), Setup status.

## Phase 6: scaling up (October 28 – November 2, 2026)

- **Distillation at scale:** 59M tokens of teacher data with quality gates (identity leaks, duplicates, repetitive
  "playful" lines, made-up facts).
- **Arithmetic format test:** a 33M model reached 96–99% on two-digit addition and subtraction with plain-format
  problems. The earlier failure was the data, not the model size.
- **A clean corpus:** 2.1B tokens (FineWeb-Edu, Cosmopedia, FineMath) with a new 24k tokenizer, no GPT-generated text.
- **Scaling sweep** (25M / 50M / 104M × heartbeat on / off / control × 3 seeds): a phase-coherence loss helps more as
  the model grows, but at these sizes the control recovered most of the gain. The synchrony-specific part is smaller
  than the 41.9M result first suggested.
- **104M model pretrained to 10B tokens:** a format-neutral knowledge probe kept rising (0.316 → 0.357 → 0.383).
- **Baselines:** the first plain-transformer baseline failed to train, which exposed a 25:1 position-over-token
  imbalance in the embeddings, present in AURA too. Fixing it helped AURA by about 0.33 val at 25M. The comparison
  against a properly tuned plain transformer followed (Phase 7).

Results: [`COSF-MI/RESULTS.md`](COSF-MI/RESULTS.md). Vision and planned tests:
[`COSF-MI/experiments/VISION.md`](COSF-MI/experiments/VISION.md).

## Phase 7: the verdict and the switch (November 3–4, 2026)

- **Same-recipe test** (pre-registered): oscillator projections lost to plain linear layers by 0.27–0.33 val loss at
  25M / 50M / 104M, at about half the speed. Per the pre-registered rule, AURA moved to a standard backbone
  (vanilla2).
- The heartbeat still helped oscillator AURA on the fixed embedding.
- **Within-token settling on vanilla2:** worse (+0.035) at 3× the compute; holding it at the edge of chaos was worse
  still; φ in the clamp made no difference.

## Phase 8: the new Aura (November 4–7, 2026)

- **vanilla2 104M on the same 10B tokens** beat the old curve at every point, in about half the GPU time (chat
  factuality 0.206 → 0.302).
- **Retrieval:** factuality 0.308 → 0.512, false claims 33% → 13%.
- **Confidence gate and one turn of memory:** bad answers to garbled input 52% → 12%; follow-up answers 0.090 → 0.353.
- **Next: Aura v3**, a ~327M model with 1024 context on ~15B tokens.

## Where the oscillators stand

Oscillators were not abandoned; their role changed. As the backbone (replacing every linear layer) they lost, and
that question is closed. As modules on a strong backbone they are still open: settling (retest with a zero-init
readout on multi-step reasoning), cooperative attention, and oscillator-based regulation of the confidence signal.
Each must beat vanilla2 on its own test. Replay sleep, from the oscillator phase, is still used in fine-tuning.

## What carried through

| idea | first appeared | where it stands now |
|---|---|---|
| oscillator layers | Phase 1 | lost to linear layers as the backbone (Phase 7); open only as modules on vanilla2 |
| heartbeat (synchrony) | Phase 1, fixed in Phase 3 | a coherence loss helped the oscillator model; the synchrony-specific share is unsettled; not used on vanilla2 |
| sleep | Phase 1 | weight-changing sleeps failed; replay sleep works and is used in fine-tuning |
| chirality | Phase 1 | fixed per-feature signs; a per-token gate showed nothing at tiny scale; parked with the oscillator backbone |
| settling | Phase 7 | worse on vanilla2 at 3× the compute; a retest with a zero-init readout is open |
| stability bounds | Phase 0 | in the oscillator model (Julia phase-velocity bound); not part of vanilla2 |
| assistant with memory | Phase 0 | one turn of chat memory and retrieval are in (Phase 8); memory between conversations is a research direction in `VISION.md` |
