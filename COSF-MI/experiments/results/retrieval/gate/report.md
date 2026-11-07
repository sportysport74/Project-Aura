# Retrieval gate eval (pre-registered rules applied)

Plan: `experiments/plans/retrieval_gate.md`. Grader hand checks: `handcheck_{v1,v2,v3,32b}.md` (all failed); rule 2 uses blind hand grades.

## Gate 0 and thresholds (tune split only)

| signal | AUROC (tune) | threshold | used |
|---|---|---|---|
| g garbled (b10_s1 NLL) | 0.969 | 6.4495 | yes |
| c chit-chat (max sim to tune examples) | 0.997 | 0.6074 | yes |
| s top-1 source similarity | 0.779 | 0.7114 | **no (AUROC < 0.85)** |

## Routing (test split, and the 100 factuality questions)

| kind / sub | n | rag | chitchat | unclear | nosources |
|---|---|---|---|---|---|
| chitchat | 164 | 1% | 82% | 18% | 0% |
| chitchat / a compliment to the assistant | 3 | 0% | 100% | 0% | 0% |
| chitchat / a greeting | 5 | 0% | 100% | 0% | 0% |
| chitchat / a one-word reaction | 2 | 0% | 50% | 50% | 0% |
| chitchat / a playful tease | 5 | 0% | 80% | 20% | 0% |
| chitchat / a reaction to something funny | 5 | 0% | 80% | 20% | 0% |
| chitchat / a short reaction with an emoji | 12 | 0% | 100% | 0% | 0% |
| chitchat / a sigh of relief | 13 | 0% | 100% | 0% | 0% |
| chitchat / agreeing with something | 8 | 0% | 100% | 0% | 0% |
| chitchat / complaining a little about homework | 9 | 0% | 100% | 0% | 0% |
| chitchat / excited about the weekend | 4 | 0% | 100% | 0% | 0% |
| chitchat / hand_short | 20 | 0% | 0% | 100% | 0% |
| chitchat / saying goodbye | 7 | 14% | 86% | 0% | 0% |
| chitchat / saying they're back after a break | 11 | 0% | 100% | 0% | 0% |
| chitchat / saying they're bored | 8 | 0% | 100% | 0% | 0% |
| chitchat / saying they're hungry | 5 | 0% | 100% | 0% | 0% |
| chitchat / saying they're tired | 12 | 0% | 83% | 17% | 0% |
| chitchat / sharing they got a good grade | 6 | 0% | 100% | 0% | 0% |
| chitchat / small talk about the weather outside their window (not a question) | 7 | 0% | 100% | 0% | 0% |
| chitchat / telling how their day went | 9 | 0% | 100% | 0% | 0% |
| chitchat / text-speak with slang | 7 | 0% | 43% | 57% | 0% |
| chitchat / thanking the assistant | 6 | 0% | 100% | 0% | 0% |
| factuality | 100 | 96% | 4% | 0% | 0% |
| garbled | 150 | 15% | 2% | 83% | 0% |
| garbled / mash | 50 | 6% | 6% | 88% | 0% |
| garbled / salad | 50 | 30% | 0% | 70% | 0% |
| garbled / shuffled | 50 | 10% | 0% | 90% | 0% |
| ontopic | 1072 | 93% | 6% | 0% | 0% |
| unanswerable | 60 | 93% | 7% | 0% | 0% |
| unanswerable / niche | 21 | 100% | 0% | 0% | 0% |
| unanswerable / personal | 31 | 87% | 13% | 0% | 0% |
| unanswerable / recent | 8 | 100% | 0% | 0% | 0% |

## Rule 1: factuality must not drop (100 questions, joined, sampled; composed from the re-score's graded rows)

| arm | score | false claims | template share | − G0 | 95% CI | rule (≥ −0.02) |
|---|---|---|---|---|---|---|
| G0 | 0.512 | 13.2% | 0% | | | |
| G1 | 0.512 | 13.3% | 0% | -0.0000 | [-0.0083, +0.0067] | PASS |
| G2 | 0.512 | 13.3% | 0% | -0.0000 | [-0.0083, +0.0067] | PASS |

## Rule 2: inappropriate answers on the off-topic test set must drop (150 messages graded: `grades_hand.jsonl`)

| arm | all | chit-chat | unanswerable | garbled | garbled: word salad (unseen type) | chit-chat: hand short | − G0 | 95% CI | rule (CI < 0) |
|---|---|---|---|---|---|---|---|---|---|
| G0 | 42.0% | 12.0% | 62.0% | 52.0% | 95.5% | 60.0% | | | |
| G1 | 27.3% | 6.0% | 64.0% | 12.0% | 27.3% | 0.0% | -0.147 | [-0.213, -0.087] | PASS |
| G2 | 48.7% | 18.0% | 64.0% | 64.0% | 72.7% | 80.0% | +0.067 | [+0.000, +0.133] | FAIL |

## Rule 3: the two gate replies, scored separately (G1)

| reply | fires on garbled | on unanswerable | on chit-chat | **mis-fires on on-topic test** | on factuality questions | G0's inappropriate rate on the messages it catches | ship (mis-fire ≤ 10%) |
|---|---|---|---|---|---|---|---|
| "I'm not sure what you mean, can you say it another way?" | 88% | 0% | 16% | **0.4%** | 0% | 44.2% (n=52) | yes |
| "I don't have good sources on that, so I'd rather not guess." | 0% | 0% | 0% | **0.0%** | 0% | nan% (n=0) | yes |

Chit-chat branch (b10_s1) on on-topic test questions: 6.4%.

## Rule 4: chat quality (30 test messages, 1 sampled reply; side by side in `chat_quality.md`)

| arm | turn ended | repeated 4-gram | rule (≤ 5 pts worse than G0) |
|---|---|---|---|
| G0 | 83% | 0% |  |
| G1 | 97% | 3% | PASS |
| G2 | 97% | 3% | PASS |

## Answer entropy vs false claims (diagnostic only; re-score answers, joined, sampled)

| answers | n | mean entropy, false claim | mean entropy, no false claim | AUROC (false claim) | 95% CI (bootstrap over questions) | predicts? |
|---|---|---|---|---|---|---|
| R2c (RAG + corpus sources) | 600 | 1.573 | 1.408 | 0.677 | [0.574, 0.769] | yes |
| R0 (b10_s1, no sources) | 300 | 1.724 | 1.587 | 0.666 | [0.589, 0.739] | yes |

## Multi-turn step A (38 two-turn conversations; RAG s0 + s1 × 3 samples; factuality grader)

| arm | score | − M0 | 95% CI | rule (CI > 0) |
|---|---|---|---|---|
| M0 follow-up alone | 0.090 | | | |
| M1 history + query of both turns, k=2 at 60 words | 0.353 | +0.263 | [+0.182, +0.346] | PASS: adopt |

M1 contexts longer than the window (cut from the left): 0 of 228.

## Decision (pre-registered)

- **G1: passes** rules 1 / 2 / 4: True / True / True.
- **G2: fails** rules 1 / 2 / 4: True / False / True.
- Templates: unclear shippable (mis-fire), no-sources shippable.
- **Gate default in store mode: G1** (lowest inappropriate rate among the passing arms).
- Multi-turn history (M1): adopted for chat.
