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
- **Experiments:**
  1. Ablation (cheap, s25/m50, 3 seeds, 200M tokens): chirality off (`enable_chirality=False`), fixed random (today), context-gated.
     If off = fixed, the sign pattern does nothing.
  2. Direction-sensitivity test: tasks where order or direction is the whole signal (reversal, A→B vs B→A relations, "the reversal
     curse"). Compare context-gated chirality vs off vs vanilla. Pass: a gap on the directional tasks that doesn't appear on matched
     undirected tasks.

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
