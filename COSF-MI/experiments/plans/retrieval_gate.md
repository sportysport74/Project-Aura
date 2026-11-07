# Plan (DRAFT, not built, awaiting approval): retrieval confidence gate, chit-chat routing, multi-turn (written 2026-11-07)

**Why (Jon's terminal test of `checkpoints/rag/s0 --store corpus`):** earthquakes good; garbled input → confident falsehood; "lol" → a fact essay;
no memory between turns. Cause: chat store mode always retrieves and always shows the top 3 to the RAG model, which was trained only on
single-turn questions where sources were present (20% random, but the question was always a real question).

**Measured before writing this (no new code):**
- bge-small top-1 cosine similarity on the 100 factuality questions (corpus store): min 0.756, p10 0.794, median 0.847. The scores are
  compressed (a known bge trait), so it's **unknown whether junk like "lol" or "asdf kjh" scores clearly lower.** That's the first thing to
  measure, and the gate is only built if it separates (gate 0 below).
- Prompt budget: the question + 3 sources (75 words each) is **232 tokens median, 342 max** (24k tokenizer). With 160 reply tokens in a
  512 window, that leaves ~120 tokens median and ~10 worst case. **2-3 turns of history don't fit next to 3 sources** (one turn ≈ 100 tokens).

## Part 1: `--show-sources` and logging (tooling; no eval)
- `--show-sources` prints the 3 sources (title or store id, similarity, first ~20 words) under each answer.
- Every turn appends to `chat_logs/<date>.jsonl` (gitignored, local only): message, top-1 similarity, gate decision, which model answered, reply.
  Your real chats become the data for tuning the gate later (never the test set).

## Part 2: confidence gate + chit-chat routing (pre-registered)
**Data (built before any tuning; never the 100 factuality questions):**
- *On-topic held-out:* the ft_rag **val** prompts (explain + oasst2, ~2,200), which the RAG model never trained on. Split 50/50 into tune / test.
- *Off-topic set, 400 messages, split 200 tune / 200 test (different seeds):* garbled (keyboard mash, shuffled letters of real questions,
  word salad from real questions), chit-chat ("lol", "hi", "thanks!", "that's cool", "I'm bored"), and unanswerable or real-time questions
  ("what's the weather now", "who won last night"). Programmatic plus a few hand-written lists. ~1/3 each.
- *Chit-chat detector training data:* ft_v2 **train** prompts, category chat vs explain/oasst2. A logistic regression on the bge embeddings
  (already computed by the store's embedder). No new model.

**Gate 0 (separability, decided on the tune split only):** AUROC of top-1 similarity, on-topic vs garbled + unanswerable, ≥ 0.85. If it fails,
stop: report the score distributions and don't build the similarity gate (a different signal would be needed, e.g. the RAG model's own
answer likelihood). The chit-chat detector has the same rule (AUROC ≥ 0.85 on tune, chit-chat vs on-topic).

**Threshold (fixed on tune, then frozen):** the highest threshold that still passes **≥ 95% of on-topic tune questions** (so factuality can't
fall far). The chit-chat detector threshold is picked the same way (≥ 95% of on-topic tune questions classed as not chit-chat).

**Routing (arms):**
| arm | chit-chat | below the threshold | above |
|---|---|---|---|
| G0 (today) | RAG + sources | RAG + sources | RAG + sources |
| G1 | b10_s1, no sources | fixed reply: "I'm not sure what you mean. Can you say it another way?" | RAG + sources |
| G2 | b10_s1, no sources | b10_s1, no sources | RAG + sources |

G1's below-gate reply is a **template, not the model**. I'm proposing it because a model answering garbled input without sources still
invents facts (R0 false claims 33%). G2 tests whether answering closed-book is enough. b10_s1 is the trained no-source chat model (the RAG model
never saw a message without sources outside the 20% random ones), so both models are loaded (~0.4 GB each).

**Evals (all on test splits; same decoding as the re-score: top-p 0.9, T 0.7, penalty 1.15 excluding sources; 3 samples):**
1. **Factuality must not drop:** the 100 questions, G1 and G2 vs G0. Rule: point drop ≤ 0.02 (joined, sampled). Abstentions score 0.
2. **Off-topic test set, inappropriate-answer rate must drop:** the Qwen2.5-7B grader with a fixed rubric: for garbled/unanswerable,
   *inappropriate* = states facts as an answer instead of saying it isn't sure or asking; for chit-chat, *inappropriate* = an unprompted factual
   essay or a false claim. I hand-check 30 grades against the rubric before trusting it (as with the factuality grader). Rule: G1 (and G2) vs G0
   paired bootstrap 95% CI < 0.
3. **Chat quality (small):** a 30-message mixed set (10 fact, 10 chit-chat, 10 follow-ups): turn-ending rate and repeated-4-gram rate must not
   worsen by more than 5 points vs G0, plus a side-by-side markdown for Jon to read. The reading is the real check here; the numbers are a floor.

**Decision:** the gate goes into `chat_with_aura.py` (as the default in store mode) only if rules 1-3 pass. If both G1 and G2 pass, prefer the
one with the lower inappropriate rate. Report the gate's abstain rate on the factuality set either way.

## Part 3: multi-turn (two steps; only step A is in this approval)
- **A. Zero-shot measurement (cheap):** history = the last 1-2 turns, with the retrieval query = the previous user message + the current one
  (so "why does it do that?" retrieves about the earlier topic). To fit the window, sources drop to **k=2 at 60 words** when history is present.
  Eval: 40 two-turn follow-up conversations (turn 1 a factuality-style question, turn 2 a pronoun follow-up with a reference answer), graded by
  the factuality grader; single-turn factuality must be unchanged (empty history = today's prompt, bit-identical). Compare against "no history".
- **B. If A shows the model ignores or garbles history** (expected: it never saw history with sources): a multi-turn RAG fine-tune set
  (Qwen2.5-7B teacher, corpus sources only, same filters) and a fine-tune of ~1.5 GPU-h. That needs its own plan and approval.

## Cost
Code ~half a day (CPU). GPU: off-topic set grading and gate eval ~20 min; factuality re-run for G1/G2 ~40 min; multi-turn A ~20 min.
**≈ 1.5 GPU-h**, one unattended chain, smoke-tested, with FAILED + ntfy on any stop. Licences: nothing new (bge MIT, Qwen Apache-2.0, corpus
ODC-By; Wikipedia stays inference-only).
