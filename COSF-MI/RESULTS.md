# AURA results

Measured results only. Every entry states its evidence: seeds, baselines and controls, and the exact file it comes from. Vision and
plans live in `experiments/VISION.md` and `experiments/plans/`; work in progress lives in the Setup status of `CLAUDE.md`. If an
entry's design changed after the data came in, it says so here.

## 2026-11-05 — New AURA base: vanilla2 l100 on the same 10B tokens (vs the old oscillator curve)

- **Design** (`experiments/plans/vanilla2_base_10b.md`, approved 2026-11-04): vanilla2 l100 (103.5M; linear projections, embedding × √d, GELU,
  GPT init), lr 6e-4, the old curve's exact protocol and data order (2.2B cosine; WSD continuation at 3e-4 with a 5B decay branch and the 10B
  final decay). Same evals at every point. One training run per point (CIs are over the 100 questions). Factuality: ft_v2 × 2 seeds per point,
  pooled. `experiments/results/v2base/`; the old curve in `experiments/results/pretrain_10b/` and `ft_v2_curve/`.

| point | v1 val, old → **new** | knowledge PMI, old → **new** (new − old, 95% CI) | battery ALL, old → **new** | chat factuality after ft_v2 (joined), old → **new** (new − old) |
|---|---|---|---|---|
| 2.2B | 2.8769 → **2.6442** | 0.316 → **0.447** (+0.131 [+0.106, +0.156]) | 53% → **67%** | 0.156 → **0.246** (+0.090 [+0.063, +0.118]) |
| 5B | 2.7735 → **2.5791** | 0.357 → **0.434** (+0.077 [+0.056, +0.098]) | 69% → **74%** | 0.180 → **0.260** (+0.080 [+0.045, +0.115]) |
| 10B | 2.6802 → **2.5180** | 0.383 → **0.448** (+0.065 [+0.045, +0.086]) | 74% → **76%** | 0.206 → **0.302** (+0.096 [+0.062, +0.132]) |

(Factuality, pieces encoding: new − old +0.090 / +0.102 / +0.081, all CIs above 0. Fully correct answers: new 3-10% vs old 1-4%.)

- **The new base is better at every point on every measure**: val −0.16 to −0.23, knowledge probe +0.07 to +0.13, chat factuality +0.08 to
  +0.10, battery +2 to +14 points, at ~2× the training speed (17 h for the 10B pretrain vs ~27.5 h).
- **Pre-set criterion (10B vs 5B on the probe, CI excluding 0): PASS, narrowly**: +0.014 [+0.002, +0.027].
- **But on the new base the probe barely moves with more tokens**: 2.2B 0.447 → 5B 0.434 → 10B 0.448 (10B − 2.2B: +0.002 [−0.017, +0.020]).
  Chat factuality after fine-tuning does rise: 2.2B → 10B **+0.056 [+0.023, +0.089]** (joined), +0.060 [+0.029, +0.093] (pieces); val loss falls
  steadily. One possible confound: the 2.2B point is the only one annealed from the 6e-4 peak (cosine), the others from the 3e-4 WSD
  continuation, and the probe may reward the sharper anneal. So the probe looks saturated or schedule-sensitive for this model; **chat factuality
  after the same fine-tune is the better knowledge curve** here.
- Run notes: no GPU faults (0/2 auto-resumes). Phase A crashed once at its epoch end on a code bug (oscillator-only metrics); fixed, resumed
  from the step-118,760 state, and the re-run reached the identical val loss (2.6442).

## 2026-11-04 — Criticality-controlled settling on vanilla2, s25 (pre-registered; settling doesn't help the backbone)

- **Design** (`experiments/plans/settling_vanilla2.md`, committed 2026-11-03 19:49, addendum with smoke tests and compute matching 20:36, both
  before any 200M-token run): a within-token Kuramoto settling module after every block of vanilla2 s25 (T = 8 steps, symmetric low-rank J,
  K = 1), lr 6e-4, 200M tokens. Seed 1 for every arm; seeds 2-3 for arms beating their comparison by > 0.005 (selection by script:
  `experiments/results/settling/stage2_selection.md`). Paired by seed. `experiments/results/sweep/summary.md`.

| arm | val (s25) | n | tok/s | GPU min / run |
|---|---|---|---|---|
| a. vanilla2 | **3.4311 ± 0.0004** | 3 | 588k | 6 |
| b. R4 control (plain block × 9 repeats, compute-matched to c by measured speed) | 3.8461 ± 0.0085 | 3 | 157-170k | 20-21 |
| c. settling, K = 1 | 3.4656 ± 0.0009 | 3 | 183-188k | 18-19 |
| d. + clamp ÷ φ | 3.4663 | 1 | 184k | 19 |
| e. + clamp ÷ 2 | 3.4665 | 1 | 175k | 20 |
| f. + criticality controller | 3.5760 ± 0.0143 | 3 | 77-85k | 40-44 |

- **Rule "settling counts only if c or f beats b": met on the letter.** c beats b by 0.381 (paired p < 0.001), f by 0.270 (p = 0.002). **But
  the R4 control is itself broken**: it's 0.415 worse than plain vanilla2. Beating it says nothing in settling's favour, and **settling
  makes the backbone worse: c − a = +0.035 (paired p < 0.001)** at ~3.2× the GPU time. At equal GPU time it's far behind: vanilla2 l100
  trains in the same ~19 min and reaches 3.141.
- **φ question (d vs e): no evidence φ is special.** 3.4663 vs 3.4665 at seed 1 (gap 0.0002, under the 0.005 threshold, so no stage 2).
  The clamp fires on 2-3% of steps in layer 0 and ~0 in layers 5-7 by the end of training.
- **Controller (f vs c): it doesn't count.** f is 0.110 worse (paired p = 0.006). It used the Lyapunov signal (not noisy: sd 0.0022 vs mean
  0.0167 at step 500) and held λ_T ≈ 0.000 by raising K to 5-11 in most layers. Edge-of-chaos dynamics hurt here.
- **Diagnostics (end of training, mean over layers):** c settles to a mildly contracting regime (λ_T ≈ −0.010) with **partial synchrony,
  R 0.43 → 0.37 over the 8 steps** (not R → 1). The controller's arm sits at R ≈ 0.59.
- **Caveats:** s25 only; one K, T, γ, β. The R4 control as specified (the same block repeated without dynamics) turned out to be a harmful
  architecture, not a neutral compute sink, so a future compute-matched control should be a standard one (a wider or deeper vanilla2 at
  equal GPU time).

## 2026-11-03 — Heartbeat retest on the fixed embedding (pre-registered; the heartbeat stays)

- **Design** (`experiments/plans/heartbeat_retest.md`, 2026-11-02 09:56, before any result): `aura_std` (heartbeat on) vs `aura_std_off`
  (`--heartbeat-strength 0`), s25 and m50, seeds 1-3, paired, 200M tokens. Rule: keep the heartbeat only if on beats off at m50 with paired
  p < 0.05.
- **Result:** off − on = **+0.041 at m50** (3.5952 ± 0.0072 vs 3.5542 ± 0.0035; paired t 9.55, **p = 0.011**), +0.053 at s25 (p < 0.001).
  **A coherence loss helps oscillator AURA**, and per the rule the heartbeat stays.
- **What this does not show:** that synchrony is the mechanism. True R is 0.30 with the heartbeat and 0.007-0.009 without, which shows the
  loss does what it's built to do, nothing more. There's no `live_control` arm on the fixed embedding, and on the old embedding the control
  (a coherence loss toward a random phase pattern) captured most of the heartbeat's gain.
- Scope: this is within the oscillator backbone. Per the oscillator verdict below, the next backbone is vanilla2, where the heartbeat as
  defined (on oscillator phases) doesn't apply. Carrying it over needs a phase-bearing module on vanilla2 (VISION.md §2).

## 2026-11-03 — ft_v2 chat factuality on the 2.2B / 5B / 10B bases (2 seeds each)

- Same fine-tune (fact-first ft_v2 set, seq 512, lr 1e-4, 1 epoch, replay) on each base, seeds 0 and 1; chat-format factuality (100
  questions, 3 sampled seeds, Qwen2.5-7B grader); paired bootstrap over questions with the 2 seeds pooled.
  `experiments/results/ft_v2_curve/{summary.md,compare.txt}`.

| base | joined (sampled) | pieces (sampled) |
|---|---|---|
| 2.2B | 0.156 | 0.166 |
| 5B | 0.180 | 0.183 |
| 10B | **0.206** | **0.235** |

- 10B − 2.2B: **+0.050 [+0.020, +0.079]** (joined), **+0.069 [+0.044, +0.095]** (pieces). 5B − 2.2B: +0.024 / +0.017 (CIs include 0).
  10B − 5B: +0.026 [−0.003, +0.053] (joined), +0.053 [+0.029, +0.077] (pieces).
- **The knowledge gain from more pretraining does show up in chat**, modestly: about +0.05 to +0.07 from 2.2B to 10B. Almost all of it is
  PARTIAL credit (fully correct answers: 1-4%), and false claims stay at 39-49% of replies.

## 2026-11-03 — Oscillator projections vs a standard transformer (pre-registered; rule 2 fired)

- **Design** (pre-registered in `experiments/plans/vanilla_embfix_decision.md`, 2026-11-02 04:22, addendum 04:30, both before any
  m50/l100 result): `aura_std` = AURA (oscillator projections, heartbeat on) with the standard recipe (token embeddings × √d_model, GELU,
  GPT init incl. the residual rule on the oscillator amplitudes) vs `vanilla2` = the same model with plain linear projections. Only the
  projections differ (params within 0.2%). s25 / m50 / l100, 3 seeds each, 200M tokens, same data, schedule and seeds (paired), v1 val.
  Results: `experiments/results/sweep/{results.tsv,summary.md}`.

| size | vanilla2 | aura_std | gap (aura_std − vanilla2) | paired t, p | embfix (AURA, tanh, default init) |
|---|---|---|---|---|---|
| s25 (23.5M) | 3.4311 ± 0.0004 | 3.7603 ± 0.0039 | **+0.329** | 136.2, < 0.001 | 3.7414 ± 0.0049 |
| m50 (50.2M) | 3.2552 ± 0.0014 | 3.5542 ± 0.0035 | **+0.299** | 130.4, < 0.001 | 3.5397 ± 0.0082 |
| l100 (103.7M) | 3.1410 ± 0.0013 | 3.4116 ± 0.0032 | **+0.271** | 111.2, < 0.001 | 3.3936 ± 0.0011 |

- **Verdict (rule 2): oscillator projections cost quality at this scale.** The gap is ~100× the seed sd at every size and shrinks only
  slowly with size (0.33 → 0.30 → 0.27). Per the pre-registered rule, the next designs use a standard backbone, with oscillator mechanisms
  added as modules that must each beat vanilla2 on their own test.
- **Compute** (450 W cap): vanilla2 is 1.9-2.3× faster (588k / 346k / 173k tok/s vs 313k / 149k / 81k). At equal GPU time it's further
  ahead: aura_std l100 (41 min/run) 3.412 vs vanilla2 l100 (19 min) 3.141; aura_std m50 (22 min) 3.554 vs vanilla2 l100 (19 min) 3.141.
  Loss improvement per GPU-hour over a fixed reference (s25 heartbeat-off, old embedding, 4.1266): vanilla2 7.36 / 5.43 / 3.06 vs aura_std
  2.07 / 1.54 / 1.04.
- **Recipe effect** (embfix vs aura_std, both oscillator): GELU + GPT init make AURA slightly *worse*, by 0.015-0.019 (paired p 0.06 / 0.16 /
  0.014). The recipe doesn't explain the gap.
- **Embedding fix** (embfix vs the earlier heartbeat-on arm, same seeds): −0.33 / −0.36 / −0.37. The std-0.02 token embedding next to a
  std-0.5 positional encoding had been costing AURA more than the whole oscillator-vs-linear gap.
- **Caveats:** one learning rate (6e-4, tuned on AURA) for both; the pre-registered lr check for vanilla2 (3e-4 / 1e-3, s25) is queued and
  can only widen the gap: **it did** (s25 seed 1: lr 3e-4 → 3.5785, 6e-4 → 3.4311, 1e-3 → 3.3820, so vanilla2 at 1e-3 is a further 0.05
  ahead, and the best lr may be higher still, since 1e-3 is the edge of the grid). 200M tokens per run (under-trained at l100). The earlier `vanilla` arm (2026-10-31) was a broken recipe and is
  not a result.

## 2026-11-02 — 10B-token pretrain of l100 (104M): knowledge at 10B (pre-set criterion met)

- **Model:** `checkpoints/pretrain/l100_10b/best_model.pt`: the main WSD run continued from 4.5B to the end of the 7.8B-token extension
  epoch (10.0B total), decayed to 0 over the last 10%. Heartbeat on, seed 1, **one training run per point** (CIs are over the 100 questions).
- **Criterion, fixed 2026-11-01 before stage 2 finished:** knowledge-probe gain over 5B with the paired-bootstrap 95% CI excluding zero.
- **Result: PASS.** PMI 0.357 (5B) → **0.383 (10B), gain +0.026, 95% CI [+0.011, +0.041]**
  (`experiments/results/pretrain_10b/knowledge_pmi_10b.json`). The true answer's log-prob per token rose too: +0.095 [+0.076, +0.115].

| tokens | v1 val loss | knowledge PMI (vs 2.2B) |
|---|---|---|
| 2.2B | 2.8769 | 0.316 |
| 5.0B | 2.7735 | 0.357 (+0.041 [+0.029, +0.054]) |
| 10.0B | **2.6802** | **0.383** (+0.067 over 2.2B) |

- Reading: knowledge keeps rising with more tokens at 104M, with diminishing returns per token (+0.041 for 2.8B more, +0.026 for 5B
  more). Chat-format factuality on the bases stays invalid (the 10B base scores 0.00 because "Aura:" now means arithmetic, see below).
- Still open: whether the knowledge shows up in chat (ft_v2 on the 2.2B/5B/10B bases × 2 seeds, then chat factuality).
- Run notes: two GPU driver faults (nvlddmkm 153, 10-31 12:40 and 11-01 05:47) hung the run. The hourly full-state saves and the stall
  watchdog bounded the loss; the final 17 h segment ran without a fault. Two resumes before 2026-11-01 ran ~15% slower (a resume bug, now
  fixed). Neither affects the numbers above. `checkpoints/pretrain/l100_10b/CRASHES.md`.

## 2026-11-01 — 10B-token pretrain of l100 (104M): knowledge at 5B vs 2.2B

- **Models:** `checkpoints/pretrain/l100_on_s1` (2.2B tokens, cosine to 0) vs the 5B branch `checkpoints/pretrain/l100_5b_branch`
  (continued from 2.2B on new shards of the same three sources, WSD, decayed to 0 at 5.0B total). Heartbeat on, seed 1. **One training run
  per point**: every CI below is over the 100 eval questions, not over seeds.
- **Val loss** (v1 val split, 10.96M tokens): 2.8769 → **2.7735**.
- **Knowledge probe** (`experiments/knowledge_pmi.py`; `experiments/results/pretrain_10b/knowledge_pmi_5b.json`): the PMI of the
  reference answer, i.e. log P(answer | its question) − log P(answer | 10 other questions), per token, in plain "Question: … Answer:" text.
  0.316 → **0.357, gain +0.041, 95% CI [+0.029, +0.054]** (paired bootstrap over questions). Knowledge rises from 2.2B to 5B.

### Post-hoc change of the gate metric (disclosed)

The plan's pre-set 5B gate was **chat-format factuality** (`factuality_gen.py` + Qwen2.5-7B grader; continue only if the gain is
≥ +0.05 with a CI above 0). It came out **0.062 → 0.017, CI [−0.072, −0.020]: FAIL**
(`experiments/results/pretrain_10b/gate_5b.json`). **The metric was switched after seeing the data**, to the probe above, because the
chat-format measure turned out not to measure knowledge for base models:

- The corpus extension left out the distill chat transcripts, so the only "You: … Aura: …" text in the 2.8B new tokens was programmatic
  arithmetic. The 5B base answers chat-format science questions with sums ("Why does bread rise?" → "1372 - 22 = 1350 / Answer: 1350";
  its arithmetic battery went 53% → 69%). The 2.2B base's partial credit came from the simile style of its distill slice, not from facts.
- So chat-format factuality on a base measures which chat style the corpus taught, not what the model knows. The probe uses no chat
  format, and because the answer text is the same on both sides, it cancels general fluency.
- The switch was decided by the user on 2026-11-01. The 10B criterion was then fixed **before** stage 2 finished: probe gain over 5B,
  95% CI excluding zero (`experiments/plans/pretrain_10b.md`).
- Still open: whether the knowledge shows up in chat. That test is the same ft_v2 fine-tune on the 2.2B, 5B and 10B bases (2 seeds
  each), then chat factuality.
