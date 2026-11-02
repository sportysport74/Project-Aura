# AURA results

Measured results only. Every entry states its evidence: seeds, baselines and controls, and the exact file it comes from. Vision and
plans live in `experiments/VISION.md` and `experiments/plans/`; work in progress lives in the Setup status of `CLAUDE.md`. If an
entry's design changed after the data came in, it says so here.

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
