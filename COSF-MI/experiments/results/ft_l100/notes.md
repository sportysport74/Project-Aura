# Distill fine-tune on the l100 base (104M), 2026-10-30

Base: `checkpoints/pretrain/l100_on_s1/best_model.pt` (full 2.2B-token pretrain, heartbeat on, val 2.8769). Recipe as the 33M
A/B `new` arm: pilot `data/distill_pilot/ft_v1` only (licence-clean; teachers and licences in its build report), re-tokenized
with the 24k tokenizer at seq 512 (0 of 316,502 conversations longer than 512), batch 32, lr 1e-4, 1 epoch = 9,890 steps,
replay sleep every 300 steps on corpus train, seeds 0 and 1 (`configs/ft_l100/`). Training runs: `runs/20261030-160816`
(the eval in that queue crashed on the base arm, since a pretrain checkpoint has no `chat_format`; fixed in `math_eval.py`),
eval rerun `runs/20261030-180355`.

Caveat: `ft_v1` train is ~1.5% of the pretraining corpus, so the base has already seen this data once.

`summary.md` and `readable.md` are written by `ab_eval.py`. Its intro paragraph (new/old arms) is boilerplate from the 33M A/B
and doesn't describe this run, and the val table repeats the base column (from `--base` and from the `base=` arm).

## Training

| | before | seed 0 | seed 1 | 33M `new` |
|---|---|---|---|---|
| pilot val | 2.4140 | 1.7605 | 1.7607 | 1.8950 / 1.8946 (16k tokenizer, not comparable per token) |
| corpus val (forgetting) | 2.8406 | +0.0621 | +0.0625 | +0.121 (its story val) |

Both seeds agree to 4 decimals. Forgetting is about half the 33M's (different forgetting sets, so this is indicative only).

## Arithmetic: the fine-tune keeps the base's arithmetic; it doesn't add any

There are two batteries, and they disagree by prompt encoding:

- `summary.md` (`ab_eval`): the prompt is encoded piece by piece (`You:` + question + `Aura:`), which matches the fine-tune format but
  not the pretraining corpus. Here the base scores 17% and the fine-tunes 30% (33M `new`: 13%).
- `corpus_format_battery/` (`base_arith_eval.py --max-new 160`): one SentencePiece encode of `You: {q}\nAura:`, the corpus's
  arithmetic slot format. Here the base scores 53/54% (what_is/equals), seed 0 50/48% and seed 1 52/49%.

So the "+13 points from fine-tuning" in `summary.md` is mostly the base being handicapped by the off-format prompt. In its own format
the base was already better, and fine-tuning costs 1-5 points: 1-digit × drops from 100% to 85%, 1-digit ÷ from 35% to 25-30%, and
2-digit + on the "= ?" template from 80% to about 50%. 2-digit − and 3-digit ± are slightly up. Multiplication beyond 1 digit stays at 0-5%.
With the default 48-token cap the fine-tunes score only 37-40%: they learned the pilot's "Imagine adding X and Y is like stacking two
piles of blocks..." preamble, which pushes the answer past 48 tokens, so the 160 cap is the fair one.

## Voice: no, 104M does not escape simile soup

The surface is better: grammatical sentences, no "the ball is the ball" loops, and fewer repeated 4-grams (sampled 25% vs 21% at 33M,
greedy 54% vs 56%). Stories (prompts 21-30) are clean, coherent TinyStories-style openings.

The chat and explain answers are still similes standing in for content, with the facts wrong underneath:

- "Yellow is red because of the extra pigment... Blue is red because it has three primary pigments"
- "Ice floats like a big, frozen blanket! It's super hot and can melt the water molecules"
- Thunder "happens when the wind blows in and out of your eyes"
- Sick dog: "Oh no, that sounds like a funny problem!" (tone failure on an empathy prompt)
- Gravity: "a giant invisible hug that pulls everything towards the center of the Earth". This is the one partly right
  explanation, and still wrapped in the hug simile.

Simile density (like a/an/the, imagine, as if, think of it, per 100 words) is about 1.0-1.4 for every arm (33M new 1.0-1.8, base 1.0-3.4),
so the problem isn't *more* similes. It's that the model reproduces the teacher's playful template without the knowledge to fill it. The
explain answers in the pilot data pair a simile with a real fact; at 104M the model keeps the simile and invents the fact. Repetition
penalty (sampled_rep) removes loops (repeated 4-grams 3%) and makes 67% of chat/explain replies end in a question (the pilot's
follow-up-question habit), but it doesn't improve the content.

Readable maths with sampled_rep is 2/10; under greedy it's 4/10 (s0 3, s1 5). The worked lines often hold the right equation and then the wrong
result ("6 × 12 = 70", "144 ÷ 12 = 10").

## Reading

- Scale fixed the fluency and halved the forgetting. It didn't fix factuality, which is what makes the voice read as soup.
- Next levers, cheapest first: (a) score the voice on factuality rather than repetition (a small graded explain set); (b) reduce the
  simile template in the explain slice of the pilot data, so that fact comes first and the simile is optional; (c) more pretraining
  knowledge (a bigger model or more tokens), since 104M at 2.2B tokens knows little science.
