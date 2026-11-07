# Pending for HISTORY.md (Jon, 2026-11-07): goes after Phase 6 of Jon's HISTORY.md draft (draft not yet in the repo)

Checked against RESULTS.md: every number below matches a measured entry.

## Phase 7: the verdict and the switch (November 3–4, 2026)
- Same-recipe test: oscillator projections lost to plain linear layers
  by 0.27–0.33 val loss at 25M/50M/104M, at about half the speed. Per
  the pre-registered rule, AURA moved to a standard backbone (vanilla2).
- The heartbeat still helped oscillator AURA on the fixed embedding.
- Within-token settling on vanilla2: worse (+0.035) at 3x compute;
  holding it at the edge of chaos was worse still; φ in the clamp made
  no difference.

## Phase 8: the new Aura (November 4–7, 2026)
- vanilla2 104M on the same 10B tokens beat the old curve at every
  point, in about half the GPU time (chat factuality 0.206 -> 0.302).
- Retrieval: factuality 0.308 -> 0.512, false claims 33% -> 13%.
- Confidence gate and one turn of memory: bad answers to garbled
  input 52% -> 12%; follow-up answers 0.090 -> 0.353.
- Next: Aura v3, a ~327M model with 1024 context on ~15B tokens.

## Where the oscillators stand
Oscillators were not abandoned; their role changed. As the backbone
(replacing every linear layer) they lost, and that question is closed.
As modules on a strong backbone they are still open: settling (retest
with a zero-init readout on multi-step reasoning), cooperative
attention, and oscillator-based regulation of the confidence signal.
Each must beat vanilla2 on its own test. Replay sleep, from the
oscillator phase, is still used in fine-tuning.

## Still to do when the draft arrives
- Update its "What carried through" table to match Phases 7-8 and the section above.
- Keep everything else; nothing gets deleted.
