"""
Bridge-word probe for context-dependent chirality.

For words with two meanings, run sentences using each meaning and read the
per-token handedness g (tanh gate, one value per feature group per layer) at
the word. If handedness carries meaning, sentences with the same sense should
have similar g, and the two senses should differ, ideally with opposite signs
in some groups.

Score per word:
  within  = mean distance between g vectors of sentences with the SAME sense
  across  = mean distance between g vectors of sentences with DIFFERENT senses
  ratio   = across / within   (> 1: the gate separates the senses)
  signs   = feature groups whose mean g has opposite signs for the two senses
  control = the same ratio on the final hidden state (ln_f output), which any
            language model uses to tell senses apart; the gate only carries
            meaning beyond that if its ratio is comparable or higher

Usage: python experiments/bridge_words.py experiments/round3/context_chir.pt
"""
import itertools
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "python"))
from cosf_mi.models.cosf_oscillator import COSFOscillatorTransformer  # noqa: E402
from cosf_mi.data.tokenizer import BPETokenizer  # noqa: E402

# The model reads left to right, so every clue to the meaning comes BEFORE the word.
WORDS = {
    "bat": (
        ["Tom wanted to hit the ball, so he swung the bat",
         "At the baseball game, he held his new bat",
         "She grabbed the ball and the bat"],
        ["In the dark cave, a little bat",
         "It was night and the vampire bat",
         "Something with wings flew out of the cave. It was a bat"],
    ),
    "bark": (
        ["The dog saw the cat and began to bark",
         "Max the puppy heard a noise and started to bark",
         "When the mailman came, the big dog let out a loud bark"],
        ["She touched the old oak tree and felt its rough bark",
         "The tree trunk was covered in brown bark",
         "Bugs lived in the tree, under the bark"],
    ),
    "ring": (
        ["Mom put on her gold ring",
         "On her finger she wore a shiny ring",
         "He gave her a diamond ring"],
        ["They waited for the school bell to ring",
         "Mom heard the phone ring",
         "When Dad pressed the doorbell, it began to ring"],
    ),
    "fly": (
        ["A bug buzzed around the cake. It was a fly",
         "The frog stuck out its tongue and caught a fly",
         "There was a buzzing insect on the window, a small fly"],
        ["The bird spread its wings to fly",
         "The kite went up in the wind and began to fly",
         "The plane went down the runway and started to fly"],
    ),
    "park": (
        ["They played on the swings and slides at the park",
         "The kids ran to the playground in the park",
         "On a sunny day, Lily went to play at the park"],
        ["Dad drove the car and looked for a place to park",
         "Mom stopped the van and tried to park",
         "He turned the truck around to park"],
    ),
    "light": (
        ["The room was dark, so Mom turned on the light",
         "He flipped the switch and saw the bright light",
         "The lamp gave off a warm light"],
        ["The feather was not heavy at all. It was very light",
         "The empty box was easy to carry because it was light",
         "Her bag weighed almost nothing. It was so light"],
    ),
}


def word_position(tok, text, word):
    """Index of the last subword of `word` in the tokenized prompt."""
    ids = tok.encode("[STORY] " + text)
    before = text[:text.rindex(word)] + word
    prefix = tok.encode("[STORY] " + before)
    return ids, len(prefix) - 1


HIDDEN = {}


@torch.no_grad()
def gate_vector(model, ids, pos, device):
    model(torch.tensor([ids], device=device))
    HIDDEN["vec"] = HIDDEN["last"][0, pos].float().cpu()
    parts = []
    for o in model.live_oscillators():
        g = o.last_context_chirality[0, pos]  # [out_features]
        parts.append(g[::o.out_features // o.context_groups])  # one per group
    return torch.cat(parts).cpu()


def mean_pairwise(vectors_a, vectors_b=None):
    if vectors_b is None:
        pairs = list(itertools.combinations(vectors_a, 2))
    else:
        pairs = list(itertools.product(vectors_a, vectors_b))
    return sum((a - b).norm().item() for a, b in pairs) / len(pairs)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "experiments/round3/context_chir.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = COSFOscillatorTransformer(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.ln_f.register_forward_hook(lambda m, i, o: HIDDEN.__setitem__("last", o))
    tok = BPETokenizer(str(ROOT / "models/tokenizer_8k.model"))
    print(f"model: {path} (val loss {ckpt['val_loss']:.4f})")
    print(f"{'word':<7} {'within':>7} {'across':>7} {'ratio':>6} {'sign-opposed groups':>20} {'g<0 at word':>12} {'control':>8}")

    ratios, controls = [], []
    for word, (sense_a, sense_b) in WORDS.items():
        ga, gb, ha, hb = [], [], [], []
        for texts, out, hout in ((sense_a, ga, ha), (sense_b, gb, hb)):
            for t in texts:
                ids, pos = word_position(tok, t, word)
                out.append(gate_vector(model, ids, pos, device))
                hout.append(HIDDEN["vec"])
        control = mean_pairwise(ha, hb) / ((mean_pairwise(ha) + mean_pairwise(hb)) / 2)
        controls.append(control)
        within = (mean_pairwise(ga) + mean_pairwise(gb)) / 2
        across = mean_pairwise(ga, gb)
        ma, mb = torch.stack(ga).mean(0), torch.stack(gb).mean(0)
        opposed = ((ma * mb) < 0).sum().item()
        neg = (torch.stack(ga + gb) < 0).float().mean().item()
        ratios.append(across / within)
        print(f"{word:<7} {within:>7.3f} {across:>7.3f} {across / within:>6.2f} "
              f"{opposed:>14} / {ma.numel():<4} {neg:>11.1%} {control:>8.2f}")
    print(f"mean ratio: gate {sum(ratios) / len(ratios):.2f}, hidden-state control "
          f"{sum(controls) / len(controls):.2f}  (1.0 = senses not separated)")


if __name__ == "__main__":
    main()
