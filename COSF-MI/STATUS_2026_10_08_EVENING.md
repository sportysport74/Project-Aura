# STATUS 2026-10-08 (evening): A/B tests on heartbeat, sleep, chirality

Follows `STATUS_2026_10_08.md`. All numbers come from `experiments/ab_run.py`:
tiny model, 10,000 steps, same seed and data order per arm, val loss on 512
fixed validation samples. Runs are deterministic: the same seed gives the same
number to 4 decimals. Seed-to-seed noise is about 0.02.
Raw results: `experiments/results/*.json` (round 1), `experiments/round2/*.json` (round 2).

## Headline: a phase bug was costing 0.6 val loss
The golden-angle phase offsets started at `index * 2.4 rad`, up to about **2,400 rad**.
sin/cos don't care about that size, but AdamW weight decay shrinks each weight in
proportion to its value, so every step it rotated those phases by about 0.007 rad.
That was noise the model had to keep undoing.

| arm (round 1, old init) | val loss | vs baseline |
|---|---|---|
| baseline (2 seeds) | 3.613 / 3.592 | |
| phase_nowd: phases excluded from weight decay | **2.992** | **-0.61** |
| phase_wrap: phases wrapped to [0, 2pi) every step | **2.993** | **-0.61** |

**Fixed:** `cosf_oscillator.py` now wraps the golden-angle init into [0, 2pi)
(the same function, only the stored numbers change). `train_overnight.py` also
keeps phases out of weight decay. Round 2, with the fixed model: **baseline 3.008**
(was 3.60).

## Sleep
| arm | val loss | val loss jump at each sleep |
|---|---|---|
| sleep_consol: 50% toward running average | 3.301 | +4.11, +0.06, +0.02, +0.01 |
| sleep_gentle: current (fatigue reset + phase wrap) | 3.304 | 0.0000 every time |
| sleep_decay0: frequencies x 0.9 | 3.338 | +3.54, +2.24, +1.56, +1.67 |
| sleep_phi: January, 50% toward phi | **6.214** | +3.96, +1.12, +0.25, +0.11 |
| no sleep (baseline) | 3.60 | |

- Gentle sleep's whole benefit came from wrapping the phases (see headline).
  It changed val loss by exactly 0.0000 at each sleep, and wrapping every step
  instead of every 2,000 steps doubles the gain.
- None of the sleeps that change the weights did better than gentle sleep.
  Consolidation tied it (3.301 vs 3.304) after a big first-sleep shock.
- The January phi sleep is destructive: +2.6 val loss. This confirms the
  Oct 8 audit, which found the January "drift reduction" was weight collapse.
- With the phase fix in, gentle sleep is now basically a no-op.
  An honest sleep test needs a mechanism that does something on its own:
  replaying stored batches, pruning, or noise injection.

## Heartbeat (live Kuramoto R on per-token phases)
New: `heartbeat_mode='live'` in `COSFOscillatorConfig`, `model.live_coherence()`.
The 100-epoch model has live R = 0.135 (random would be about 0.008), with
nothing training it.

| arm | val loss | live R |
|---|---|---|
| round 1 (old init): baseline | 3.60 | 0.007 |
| live_001: strength 0.01, push to 1 | 3.616 (tie) | 0.133 |
| live_01_t03: strength 0.1, lift to 0.3 | 3.613 (tie) | 0.246 |
| live_01: strength 0.1, push to 1 | 3.639 (+0.03) | 0.341 |
| round 2 (fixed init): baseline | 3.008 | 0.010 |
| recommended: live, 0.1, target 0.3 | **2.961 (-0.047)** | 0.303 |

The live heartbeat gives real, input-dependent synchrony for free. With the phase
bug fixed, it may also help a little: see the seed check below.

## Chirality
- Correction to the Oct 8 audit: chirality is a learned `nn.Parameter`, not a fixed +/-1.
  In the 100-epoch run, **785 of 17,408 features (4.5%) changed sign** compared with epoch 1
  (`experiments/chirality_history.py`), mostly in epochs 5-20 and 70-80.
  The flip counter showed 0 because it only counts manual `flip_chirality()` calls.
- But |chirality| fell from 1.05 to 0.20, and chirality multiplies the frequency, so
  the flips happen to values near 0. That looks like weight decay pushing values across
  zero, not the model choosing a new handedness.
- Within 10k steps nothing flips in any arm, and excluding chirality from weight decay
  makes no difference there (3.603). It's a long-run effect.
- Chirality is one fixed value per feature, the same for every token, so it *cannot*
  flip "on bridge words." That needs per-token, context-dependent handedness
  (`SemanticPolarityDetector` exists in the code but isn't wired in). That's a design step.

## Seed check (round 2)
| seed | baseline (fixed init) | recommended (live heartbeat, target 0.3) |
|---|---|---|
| 0 | 3.0081 | 2.9609 |
| 1 | 2.9911 | 2.9516 |
| 2 | 3.0102 | 2.9685 |
| **mean** | **3.003** | **2.960 (-0.043)** |

**Every heartbeat seed beat every baseline seed.** At this scale, the live heartbeat
held at R about 0.3 is a real (small) improvement to language modeling, not just free.
Pushing R toward 1 instead costs a little (live_01, +0.03). Moderate sync helps;
full sync hurts.

## Round 3 (Oct 9): context-dependent chirality
New: `chirality_context_groups` in the config. Each oscillator layer gets a small gate,
g = tanh(W·x + b), one value per token per group of 1/4 of the features, multiplied
into its handedness. It starts at +0.76 for every token, and a token's g can cross 0.
That reverses that feature group's rotation for that token only. It adds 70k parameters (0.7%).

| seed | recommended | + context chirality |
|---|---|---|
| 0 | 2.9609 | 2.9631 |
| 1 | 2.9516 | 2.9614 |
| 2 | 2.9685 | 2.9962 |
| mean | **2.960** | 2.974 (+0.014, within noise) |

Bridge-word probe (`experiments/bridge_words.py`): 6 words (bat, bark, ring, fly, park,
light), 3 sentences per meaning, with the clue placed *before* the word because the model
reads left to right.
- Meanings separated by the gate (across/within distance): 1.13, 1.07, 1.34 for the 3 seeds.
  The same score on the ordinary hidden state (control): 1.27, 1.20, 1.23.
- **0 of 96 feature groups have opposite signs for the two meanings**, for every word and seed.
  Only 0–3% of gate values at the bridge words are negative at all.
- Verdict at 10k steps: **no**. The gate stays near its starting value and doesn't separate
  meanings any better than the hidden state already does. Things to try: start the gate at 0
  instead of +0.76 so the sign is free from the start, train longer, or add a loss that
  rewards using both signs.

## Forgetting test (Oct 9): replay sleep works
`experiments/forgetting.py`: 6,000 steps on stories, then 4,000 steps on Shakespeare.
Every 500 Shakespeare steps, a sleep replays 50 batches. Replayed batches count toward
the same total of steps.

| arm | story loss after stories | story loss at end | **forgetting** | Shakespeare at end |
|---|---|---|---|---|
| no sleep (seed 0 / 1) | 3.070 / 3.061 | 4.470 / 4.486 | **+1.40 / +1.42** | 4.419 / 4.437 |
| replay old story batches | 3.070 / 3.061 | 3.371 / 3.425 | **+0.30 / +0.36** | 4.442 / 4.455 |
| replay recent Shakespeare (control) | 3.070 / 3.061 | 4.471 / 4.491 | +1.40 / +1.43 | 4.422 / 4.434 |

- **Replay sleep cut forgetting by about 76%** (+0.33 vs +1.41), on both seeds.
- It cost just +0.02 on Shakespeare, with 9% fewer fresh Shakespeare batches.
- The control replays just as often, but on recent data, and forgets exactly as much as no
  sleep. So **it's the replay of old memories that protects them**, not the act of replaying.
- Story loss went up at first, then *fell* during the replay arm
  (4.12 → 3.37 over phase B), while it kept rising without sleep (4.12 → 4.47).
- Honest framing: this is "rehearsal" in continual-learning terms, a known technique and not
  specific to oscillators. But it is the first sleep mechanism here that does a measurable job.
  Next step for AURA: build it into `SleepCycleHook` (keep a memory bank of past batches and
  replay some at each sleep). That matters for the chat fine-tune, where the model shouldn't
  forget how to tell stories.

## 10-epoch run with the fixes (done Oct 9 evening)
`checkpoints/heartbeat_fixed_10ep/best_model.pt`, live heartbeat (target 0.3), wrapped phases.
Final **val loss 1.742** after 10 epochs (~11 h). The old run needed **52 epochs (~2 days)**
to reach that, so it's about 5x faster. The old run's 100-epoch result (1.648) is still better:
it trained 10x longer. Sample stories from both are similar in quality (small slips in both).

## Scale contest result (Oct 9)
Same 25 minutes of solo GPU time each (`experiments/run_scale.sh`, results in `experiments/scale/`):

| setup | params | steps | val loss |
|---|---|---|---|
| **mid 512x6, batch 32, lr 5e-4** | 23M | 13,885 | **2.037** |
| tiny, batch 32, lr 6e-4 | 9M | 18,818 | 2.105 |
| big 512x8, batch 32, lr 4e-4 | 29M | 9,956 | 2.181 |
| tiny, batch 8, lr 3e-4 (old recipe) | 9M | 22,664 | 2.492 |

Batch 32 accounts for most of the gain; the mid model adds the rest. Big was step-starved
at 25 min and may win on longer budgets.
**Running overnight:** `train_overnight.py --model-size mid --batch-size 32 --lr 5e-4 --warmup 200
--epochs 20` -> `checkpoints/mid_b32_20ep`, log `logs/mid_b32_20ep.log` (~23 min/epoch, ~8 h).

## First chat model (Oct 9)
`checkpoints/aura_chat_tiny/best_model.pt`: the 10-epoch tiny model fine-tuned on 98k
TinyChat conversations (3,062 steps, 8.6 min). Chat val loss 4.65 -> **2.21**; story val
1.735 -> 1.886 (+0.15, with replay sleep every 300 steps). `chat.bat` now opens it by default.
Sample:
```
You:  I feel a little sad today.
AURA: I understand your feelings; sometimes days can feel heavy and heavy.
You:  My dog is sick.
AURA: I understand that feeling, dogs can feel heavy and sad sometimes.
You:  Thank you. What should I do?
AURA: You can share your feelings about your pet or how to help.
```
It takes turns, stays on topic and uses context; its facts are weak ("the sky is so bright
and red"), as expected for a 9M model trained on stories and small talk.
Next: redo the fine-tune on the mid model after the overnight run
(`--checkpoint checkpoints/mid_b32_20ep/best_model.pt --out checkpoints/aura_chat_mid`;
chat.bat picks aura_chat_mid automatically once it exists).

## Next: making AURA stronger (Oct 9, in progress)
- Throughput: the GPU was mostly idle at batch 8. Batch 32 runs at almost the same
  steps/sec, so it sees 3.3x more text in the same time (`experiments/bench.py`). bf16/TF32 give
  only +10–20% and risk phase precision, so we stay in fp32.
- Scale contest (`experiments/run_scale.sh`): tiny b8 / tiny b32 / mid 512x6 b32 /
  big 512x8 b32, each given the steps it can do in 25 min alone on the GPU.
- Replay sleep is now a reusable module: `cosf_mi/training/replay.py` (`ReplayMemory`,
  `replay_sleep`).
- Chat: `finetune_chat.py` (TinyChat data, `You:` / `Aura:` turns, loss only on Aura's
  replies, replay of story batches to keep storytelling). `chat_with_aura.py` switches to
  chat mode automatically for chat checkpoints.
  Data: `data/tinychat/tinychat_80mb.txt` (112k conversations, first 80 MB of
  starhopp3r/TinyChat, CC-BY-NC license, so personal/research use only).

## Round 5 (Oct 9): context chirality with the gate starting at 0
| seed | recommended | gate starts +0.76 | gate starts 0 |
|---|---|---|---|
| 0 | 2.9609 | 2.9631 | 3.0722 |
| 1 | 2.9516 | 2.9614 | 3.0738 |
| 2 | 2.9685 | 2.9962 | 3.0886 |
| mean | **2.960** | 2.974 | **3.078 (+0.12)** |

- Starting the gate at 0 hurts clearly. g = 0 means zero effective frequency, so the
  oscillators start nearly frozen.
- 42–53% of gate values are negative, but **the same share for every bridge word**
  (53.1% for bat, bark, ring, fly and light in seed 0). Each feature group just settled on
  one side of 0, whatever the token. 0–1 of 96 groups are opposed between meanings.
  Separation is no better than the hidden-state control (1.06–1.19 vs 1.16–1.26).
- **Verdict:** a per-token gate on chirality, trained only on next-word prediction, does not
  learn meaning-dependent handedness. Nothing in the loss asks for it. Ideas if we come back
  to this: a contrastive loss that pushes handedness apart for known two-meaning pairs, or
  data where the same word really shows up with both meanings (TinyStories has few such
  cases). Parked for now; the gate stays off by default (`chirality_context_groups=0`).

## Round 4 (Oct 9): sleeps that do work on their own
On top of 'recommended', with a sleep every 2,000 steps (seed 0; recommended seeds span 2.952–2.969):

| arm | val loss | change at each sleep |
|---|---|---|
| shy: frequencies x 0.98 | 2.959 | ±0.002 |
| replay of 200 random batches from the last wake period | 2.960 | 0 |
| no sleep (recommended) | 2.961 | |
| replay of the 200 hardest batches (seen 800 fewer fresh batches) | 2.964 | 0 |
| prune: zero the weakest 10% of frequencies | 2.964 | ±0.002 |

All ties. Pruning or shrinking barely moves the loss, so the model doesn't need its weakest
10% of frequencies. But in one pass over uniform data, sleep has nothing to fix: nothing gets
forgotten. So the next test gives sleep a real job:
`experiments/forgetting.py`, which trains on stories, switches to Shakespeare, and measures
how much story skill is lost, with and without replay of old story batches during sleep.

## Tank
`tank.bat`: one fish per running job (overnight run + every A/B arm log changed in the
last 6 hours), each with a progress bar, val loss, live R and ETA. Also fixed `fish.py`:
it couldn't read tqdm lines below 10 it/s (padding space), and it now follows the newest log.

## Housekeeping
`.gitignore` fixed: `models/` and `data/` are now anchored to the project root, so
`src/python/cosf_mi/models/` and `src/python/cosf_mi/data/` are tracked. Everything is
staged in git, but the commit and push were blocked by Claude Code's permission check.
To finish:
```
cd C:\Users\fatba\COSF-MI
git add -A COSF-MI
git commit -m "Track model code; phase-wrap fix; live heartbeat; A/B experiments"
git push origin main
```

## Next steps
1. **New training run with the fixes** (wrapped phases + live heartbeat at target 0.3):
   `TRAIN_TONIGHT` style. 1 epoch takes about 1 hour. Note: `--resume` can't load
   the old checkpoints' optimizer state, because the optimizer now has 2 param groups.
   Start fresh.
2. **Context-dependent chirality**: per-token handedness, then test bat/bark/ring.
3. **A sleep that does something on its own** (replay or pruning), A/B'd the same way.
4. Later: chat fine-tune.

## Oct 10 morning: mid run + mid chat model

- `checkpoints/mid_b32_20ep` (512x6, batch 32, live heartbeat) finished 20 epochs: best story val **1.5235** (old 100-epoch tiny run: 1.648).
- `checkpoints/aura_chat_mid` (finetune_chat.py on that base, ~6 min): chat val 4.256 -> **2.005** (tiny chat model: 2.205); story val 1.516 -> 1.613 (+0.10, replay sleep held it).
- Live heartbeat R stayed at 0.30 throughout fine-tuning.
- Side-by-side test chats: mid gives clearly more natural, on-topic replies ("I really like pizza because it is yummy", "sadness can be heavy like a stone"); tiny drifts ("My pet is a pet"). chat.bat now opens aura_chat_mid by default.
- Oct 10: ~0.3% of TinyChat conversations lack [INST] tags, so aura_chat_mid sometimes wrote "[/INST]". finetune_chat.py now drops those turns; retrained as checkpoints/aura_chat_mid2 (chat val 2.005, 0/64 leaks vs 2/64). chat.bat now defaults to mid2.

## Oct 10: smarter data ("smart" run)

- New pretraining mix in data/mix (prepare_mix.py), ~1.29 GB / 352M tokens: Simple English Wikipedia 250M chars, FineWeb-Edu 400M, Cosmopedia v2 textbooks 300M, GitHub Python (Avelina/python-edu-cleaned, int_score>=4) 130M, TinyStories 150M, generated arithmetic 40M.
- New tokenizer models/tokenizer_mix16k (train_tokenizer_mix.py): 16k BPE, keeps newlines/indentation (the 8k one deleted them), single digits, byte fallback.
- token_dataset_fast.py tokenizes in multithreaded batches into int32 (old list-of-ints path needed ~10 GB RAM).
- Run: `train_overnight.py --model-size big --epochs 4 --batch-size 32 --lr 5e-4 --warmup 200 --data-dir data/mix --tokenizer models/tokenizer_mix16k.model --checkpoint-dir checkpoints/smart_big_4ep` (log logs/smart_big_4ep.log), ~1.5 h/epoch. Val losses are NOT comparable with the story runs (different tokenizer and data).
- Chat fine-tune queued after it: checkpoints/aura_chat_smart (finetune_chat.py now picks tokenizer and replay data from the base model's vocab size).
