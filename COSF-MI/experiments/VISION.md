# AURA: vision, current state, and how to test each idea

This file holds **where AURA is meant to go** (Jon's vision, 2026-10-31), set against **what the code does today** and **the
experiment that would show whether each idea holds**. It is not a results file: measured numbers live in `experiments/results/`
and in the Setup status of `CLAUDE.md`, and this file only points to them. Update the "current state" lines when the code changes,
and move an idea to "tested" only when its experiment has run.

Ground rule for every experiment below: compare against **both** the matched vanilla baseline (`--projection linear`: the same model
with every oscillator projection replaced by `nn.Linear`) and the heartbeat-off/control arms, with ≥3 seeds, because one compiled run
is noise (see the heartbeat control story in `CLAUDE.md`).

---

## 0. The prerequisite: do oscillator layers beat (or match) a standard transformer at all?

- **Vision:** an architecture outside the transformer monoculture that's worth building on.
- **Current state:** never measured. Every comparison so far is oscillators vs oscillators (heartbeat on/off/control). The model
  docstring says it computes "via phase synchronization instead of matrix multiplication", but each oscillator layer is a matmul
  `Ω·x` followed by `A·(cos + χ·sin)(Ω·x + Θ)` (`cosf_oscillator.py`, `COSFOscillatorLayer.forward`).
- **Experiment (queued 2026-10-31, after 10B stage 1):** sweep arm `vanilla`: s25 / m50 / l100 × seeds 1-3, 200M tokens, identical
  recipe and data to the sweep (`scripts/sweep_queue.sh`, `ARMS=vanilla`). Parameter counts match within 0.2%.
  - Read: val loss vanilla vs on/off/control per size (Welch + paired by seed, `experiments/sweep_report.py`), and tok/s, which gives
    the compute price of the oscillators.
  - If vanilla ≥ heartbeat-on at every size, the oscillators are a cost with no benefit at this scale, and the ideas below need
    a different mechanism to pay their way.
  - Caveat: this baseline keeps AURA's tanh FFN activation and resonant positional encoding, so it isolates the oscillators. A
    GPT-style baseline (GELU, learned positions) would be a second arm.

## 1. Weights as dynamical systems: state and memory between calls

- **Vision:** connection weights are active dynamical systems. Memory lives in phase relationships, frequencies and resonance,
  and the substrate changes with use instead of being "dead stone between calls".
- **Current state:** after training, Ω, Θ and amplitudes are frozen parameters, so the same input gives the same output.
  `sync_state` and `phase_velocity` buffers do update on every forward pass, but they only feed the Julia velocity monitor (applied
  after optimizer steps, in training). The refractory path, the one input-history-dependent mechanism, is disabled
  (`enable_refractory=False` in `train_overnight.py`). Nothing carries state from one call to the next.
- **Experiment:** add a persistent phase state that evolves at inference (a Kuramoto update of per-oscillator phase offsets driven by
  the tokens seen, carried across chunks), then test memory **beyond the context window**: facts stated in chunk 1, questions in
  chunk 5 (seq 512, so recall needs the carried state). Arms: stateful oscillators, the same model stateless, vanilla with the same
  chunked setup, and vanilla + a recurrent memory token (the cheap engineering answer). Metric: recall accuracy vs distance in chunks.
  Pass: the stateful arm recalls above the stateless/vanilla floor and isn't beaten by the memory-token baseline.

## 2. Phase synchronization (Kuramoto order) as a working mechanism

- **Vision:** synchrony between oscillators carries computation and binds representations.
- **Current state:** the "heartbeat" is an auxiliary loss that keeps the Kuramoto order R of the live phases near 0.3 (`--heartbeat
  live`). It helps val loss vs no heartbeat. But the **control** arm, the same loss toward a random phase pattern (true R ≈ 0), gets
  most of the gain: at l100, off→control is larger than control→on, and the latter is thin (n=3, seed-sensitive). See
  `experiments/results/sweep/` and `control_check.md`. So the measured effect is mainly "a phase-coherence regularizer helps"; that
  synchrony itself matters is not established.
- **Experiments:**
  1. Settle control vs on at l100 with more seeds (the control's seed spread is ~4× the on arm's, so 8-10 seeds per arm), or a longer
     horizon (the 10B run is heartbeat-on only, so a control at 2B tokens would be needed).
  2. Does synchrony carry information? Probe whether R (or per-head phase clusters) predicts something about the input (e.g. topic,
     syntax depth) better than chance, and whether clamping R (fixing phases) hurts more than clamping a random equivalent.
  3. Against vanilla (section 0): if vanilla matches heartbeat-on, synchrony isn't buying anything at this scale.

## 3. Chirality and directional dynamics

- **Vision:** non-commutative, direction-dependent dynamics, where the history and orientation of the signal path shape the
  representation.
- **Current state:** chirality is a per-oscillator scalar χ that multiplies the frequencies and the sine term (`COSFOscillatorLayer`).
  It's an `nn.Parameter` initialized at +1 or −1 at random (`chirality_init_prob=0.5`) and then trained by ordinary gradients like any
  weight; `experiments/chirality_history.py` tracks its sign changes. The explicit `flip_chirality()` is called only from tests, never in
  training. An optional per-token gate (`chirality_context_groups`) exists but isn't used in the current runs. Nothing is
  non-commutative: it's a learned per-feature scale on a sinusoid, trained by standard symmetric backprop.
  - **What training did with it** (l100, 2.2B tokens, `checkpoints/pretrain/l100_on_s1`; 2026-10-31, `experiments/chirality_drift.py`,
    full tables in `experiments/results/chirality/l100_on_s1.md`). The ±1 init was rebuilt exactly from seed 1 (check: the 200M-token
    sweep run with the same seed keeps 100.00% of those signs).
    - **Signs: frozen.** 99.95% kept their init sign (42 flips of 82,944, mostly attention-out in layers 0-1). Crossing zero makes
      the oscillator's phase vanish, so a sign change is a barrier gradients almost never cross. In practice the handedness is the
      **random pattern drawn at init**, not something learned.
    - **Magnitudes: almost all shrank, none really grew.** Mean |χ| 0.62 (median 0.66, p5 0.30, p95 0.83). Only 0.3% grew above 1.1.
      Per type: k shrank most (mean 0.50, 47% below 0.5), then q 0.58, ff2 0.60, ff1 0.64, out 0.68, v 0.69. Layers 0-2 shrank more
      (≈0.52, ~39% below 0.5) than layers 5-11 (≈0.67); attention-out grows with depth (0.49 in layer 0 → 0.89 in layer 11).
    - **Most of that shrink is weight decay, not learning.** χ isn't excluded from AdamW's weight decay (0.01; only `.phases` is), and
      decay alone over this schedule (Σ lr ≈ 39.7) would leave every |χ| at **0.672**, close to the observed median 0.655. The
      gradient signal is in the deviations from 0.67: k pushed further down, v/out (deep layers) held up.
    - **The magnitude is partly unidentifiable.** χ multiplies Ω inside the phase (Ω·χ·x), so there only the product matters, and decay
      can shrink χ while Ω grows to compensate. χ's own role is the weight of the sine term in cos θ + χ·sin θ.
    - **Per-token gate:** off in this run (`chirality_context_groups = 0`), so there's nothing to report.
    - **Reading:** today's chirality is a fixed random ±1 pattern plus a decayed sine/cosine mix weight. That's not yet the
      direction-dependent dynamics in the vision. Before the ablation below, exclude χ from weight decay (like `.phases`), so its
      magnitude reflects the gradient only.
  - **Summary (Jon, 2026-10-31): current chirality is a fixed random sign (it can't cross zero) with decay-driven shrinkage, so learned
    handedness is untested, not refuted.**
  - **And without the per-token gate it can't be tested:** per-feature χ is redundant with Ω, Θ and the amplitude. A chirality layer
    equals a plain oscillator layer with Ω' = χΩ and adjusted Θ and A (checked numerically, max difference 1e-6; derivation in
    `experiments/plans/chirality_ab.md`). It changes training dynamics (the zero barrier, decay, and the heartbeat's view of the phases),
    not what the model can compute. **Learned handedness needs the per-token gate**, where direction depends on the input.
- **Experiments:**
  1. **Planned A/B at m50 (after the current queue): `experiments/plans/chirality_ab.md`.** A today's χ; B χ without weight decay;
     C angle parameterization (moves smoothly between left and right, no dead zero); D C + per-token gate; E (recommended) today's χ +
     gate. 200M tokens, 3 seeds, val loss + a new direction-sensitive benchmark + whether χ / β / the gate actually move. Expectation from
     the redundancy result: A ≈ B ≈ C on val loss, and D/E are the real test.
  2. Direction-sensitivity test: tasks where order or direction is the whole signal (reversal, A→B vs B→A relations, "the reversal
     curse"). Compare context-gated chirality vs off vs vanilla. Pass: a gap on the directional tasks that doesn't appear on matched
     undirected tasks.

## 3b. Deliberation by settling

The source doc for this section didn't come through (2026-10-31): this is written from Jon's notes plus AKOrN as published. Add the
doc's own claims under "Claims" when it's available.

- **Claims (the vision, not evidence):**
  1. Thinking is **settling**: the oscillators iterate their coupled dynamics until the phases stop changing, and the settled
     pattern is the answer.
  2. **More settling helps on harder problems** (deliberation), and harder problems **take more steps** to settle on their own, so
     compute adapts to difficulty.
  3. Meaning lives in **structured, partial synchrony**: which oscillators lock with which.
- **Constraints, stated up front:**
  - **Full synchrony (R ≈ 1) carries no information**: every phase is the same, so nothing distinguishes one input from another. If
    meaning is in the phases, it has to be in partial, structured sync (clusters, relative phases), and "settled" can't mean "R → 1".
    Today's heartbeat holds a global R near 0.3, which is a level of sync, not a structure.
  - **Synchrony is not truth.** A settled state is self-consistent, not correct. Answers still need external verification (for
    arithmetic: check the result programmatically), and "how settled" must be calibrated against correctness before it can be used as
    confidence.
- **Prior work:** AKOrN (Miyato, Löwe, Geiger, Welling, *Artificial Kuramoto Oscillatory Neurons*, ICLR 2025). Its neurons are oscillators
  (unit vectors) updated by several weight-tied, generalized Kuramoto steps per layer, coupled through learned connectivity and driven
  by the input. The paper reports gains on object discovery, robustness, calibration and Sudoku-style reasoning. Not reproduced here, and
  not tested on language or arithmetic.
- **Evidence in AURA today: none.** Each oscillator layer is a single pass, `A·(cos + χ·sin)(Ω·x + Θ)`: there's no iteration, no
  coupling between oscillators beyond the heartbeat loss, and no notion of settled or halted.
- **Test:** `experiments/plans/settling.md` (planned 2026-10-31, after the current queue). AKOrN-style settling at K = 1/4/8 weight-tied
  iterations per block, plus adaptive halting (stop when the phase change < ε, log the steps). Test it on multiplication (~0% today) and
  multi-step arithmetic: does more settling help, and do harder problems use more steps? m50, 3 seeds, against the vanilla baseline,
  today's AURA, and a compute-matched control (the same extra compute without the Kuramoto dynamics).

## 4. Consolidation: sleep and decay turning short-term patterns into structure without forgetting

- **Vision:** like synaptic consolidation, oscillation phases plus sleep let new learning settle into structural memory without
  overwriting old skills, unlike production LLMs, which forget catastrophically under continual learning.
- **Current state:** sleep (`training/sleep_cycle.py`, homeostatic reset/decay) is **off** in the clean-corpus pretrain and the sweep
  (`--no-sleep`). In the chat fine-tunes, "sleep" means replaying stored story batches, which is standard rehearsal, and story val
  still got worse (forgetting; see the distill A/B in `CLAUDE.md`). No experiment has tested sleep against forgetting.
- **Experiment (continual-learning benchmark):** start from one base, then train sequentially on domain A (e.g. arithmetic) and then
  domain B (e.g. stories or a new fact set), with A held out afterwards. Arms (3 seeds each): plain sequential fine-tune; replay;
  AURA sleep (gentle reset/decay); sleep + replay; vanilla + replay; vanilla + EWC (a standard anti-forgetting baseline). Metrics:
  retention on A after B (accuracy and loss), learning of B, and the trade-off curve. Pass: sleep beats replay / EWC on retention at
  equal B performance.

## 5. Continuous online learning

- **Vision:** the model keeps learning while in use, without blowing out its baseline.
- **Current state:** none. There's no update path at inference, and no fast/slow weight split.
- **Experiment:** define fast weights (e.g. phase offsets Θ) updated at inference from user-provided facts, and slow weights (Ω)
  updated only during "sleep" consolidation. Test: teach N new facts in conversation, then measure recall now, after K unrelated turns
  and after a sleep step, plus drift on a fixed general benchmark (the factuality set, v1 val loss). Baselines: vanilla + test-time
  fine-tuning of a LoRA, and RAG over the same facts. Pass: retention comparable to RAG with less drift than the LoRA baseline.

## 6. A different computational mode (hardware)

- **Vision:** oscillator dynamics suit non-matmul hardware (neuromorphic, spintronic / terahertz oscillators) that GPUs don't favour.
- **Current state:** AURA runs on the standard GPU stack. Ω·x is a matmul, and the oscillator part (elementwise cos/sin in fp32, which
  must stay fp32 because phases reach ~2400 rad) takes ~60% of GPU time, a cost on GPUs rather than an advantage. An
  `enable_terahertz_scaling` flag exists but has never been used or tested.
- **Experiment:** first quantify the GPU price (vanilla vs oscillator tok/s from section 0). Then a hardware argument needs a
  simulation of the target substrate: noise, limited phase precision, quantized frequencies. Measure how the trained model degrades
  under those constraints compared with vanilla under equivalent analog noise. Pass: oscillators degrade more gracefully, which is the
  property that would justify the hardware.

---

Order that makes sense: **0 first** (queued), then **4** (it's the most distinctive claim and testable at s25/m50 in a day), then 3.1
(cheap ablation), with 1 and 5 needing new mechanisms before they can be tested.
