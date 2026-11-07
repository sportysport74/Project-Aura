# Pre-registered: retrieval re-score with the sources excluded from the repetition penalty (written 2026-11-07, before running)

**Why:** the first retrieval eval (RESULTS.md 2026-11-06) sampled with a repetition penalty of 1.15 over prompt + reply, so any token that
appears in the sources was penalized. That handicaps exactly the arms that should reuse source facts (oracle: 0.613 greedy vs 0.262 sampled).

**Change (the only one):** in sampled decoding, the penalized token set is the chat template tokens + the question's tokens + the reply so far.
Tokens that appear **only in the sources** are not penalized. Everything else is identical to the first eval: the same 100 questions, the same 9
model×arm runs (R0, R0b, R1c, R2c ×2, R2w ×2, R2-closed ×2, R2-oracle ×2), both encodings, greedy + 3 sampled seeds (top-p 0.9, temperature
0.7, penalty 1.15, the same seeds), the same retrieval results, the same grader. Arms without sources (R0, R0b, R2-closed) and greedy decoding are
unaffected by construction. They're regenerated anyway, so all arms come from one run and one grading pass.
Code: `experiments/factuality_gen.py --rep-scope no_sources` → `ab_eval.generate(..., rep_ids=...)`. Without the flag the old code path runs unchanged.

**Rules (the same as the first eval's, applied to this metric):**
1. Retrieval works if R2c beats R0 (joined, sampled; paired bootstrap 95% CI > 0), R2c's false-claim rate isn't higher than R0's by more than 0.02,
   and R2-closed isn't more than 0.02 below R0.
2. The Wikipedia store becomes the default only if R2w beats R2c (CI > 0).

**Status of the result:** a second, separately pre-registered eval. **It doesn't replace the first eval's verdicts**; both are reported side by
side. If the two disagree, that's reported as a metric-dependence, not resolved in favour of either.

**Also reported:** the oracle's sampled score (expected to move toward its greedy 0.613 if the penalty was the handicap), fully-correct and
false-claim rates, the "(Wikipedia: title)" tag rate, and the copy rate (answers with ≥ 8 consecutive words of a source). Copying may rise, since
the penalty no longer discourages it.

Cost ~45 GPU-min (generation ~30 + grading ~10). Smoke-tested first (`--limit 2`), FAILED marker and ntfy alert on any stop and at the end.
