"""
Math exam for AURA chat models: 120 questions generated here (not by the teacher),
six kinds, 20 each. A reply counts as correct when the last number in it is the
answer. Decoding is near-greedy (temperature 0.1, fixed seed) so runs repeat.

Questions whose text already appears in the distill data are swapped out, so the
exam measures arithmetic the model wasn't shown word for word.

Usage:
    python experiments/math_eval.py checkpoints/aura_chat_smart10/best_model.pt [more checkpoints...]
"""

import json
import random
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "python"))
import chat_with_aura as chat  # noqa: E402
from cosf_mi.data.tokenizer import BPETokenizer  # noqa: E402

NAMES = ["Tom", "Mia", "Sam", "Lena", "Omar", "Ivy"]
THINGS = ["apples", "coins", "books", "stickers", "marbles", "cookies"]


def make_exam(seed=999, per_kind=20, skip=()):
    rng = random.Random(seed)
    kinds = {
        "add 1-digit": lambda: (lambda a, b: (f"What is {a}+{b}?", a + b))(rng.randint(1, 9), rng.randint(1, 9)),
        "add 2-digit": lambda: (lambda a, b: (f"What is {a} plus {b}?", a + b))(rng.randint(10, 99), rng.randint(10, 99)),
        "subtract": lambda: (lambda a, b: (f"What is {max(a, b)} minus {min(a, b)}?", abs(a - b)))(rng.randint(10, 99), rng.randint(1, 60)),
        "multiply": lambda: (lambda a, b: (f"What is {a} times {b}?", a * b))(rng.randint(2, 9), rng.randint(2, 9)),
        "sum of 3": lambda: (lambda a, b, c: (f"What is {a}+{b}+{c}?", a + b + c))(rng.randint(1, 9), rng.randint(1, 9), rng.randint(1, 9)),
        "word problem": lambda: (lambda n, t, a, b: (f"{n} has {a} {t} and gets {b} more. How many {t} does {n} have now?", a + b))(
            rng.choice(NAMES), rng.choice(THINGS), rng.randint(2, 40), rng.randint(2, 30)),
    }
    exam = []
    for kind, make in kinds.items():
        got = set()
        while len(got) < per_kind:
            q, a = make()
            if q.lower() not in skip and q not in got:
                got.add(q)
                exam.append((kind, q, a))
    return exam


def last_number(text):
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(nums[-1]) if nums else None


def run(checkpoint, exam, device):
    model, config, fmt = chat.load_model(checkpoint)
    tok = BPETokenizer(str(ROOT / ("models/tokenizer_mix16k.model" if config.vocab_size == 16000
                                   else "models/tokenizer_8k.model")))
    rows = []
    for i, (kind, q, a) in enumerate(exam):
        torch.manual_seed(i)
        reply = chat.generate_chat_reply(model, tok, [], q, fmt, temperature=0.1, device=device)
        rows.append({"kind": kind, "q": q, "answer": a, "reply": reply, "correct": last_number(reply) == a})
    del model
    torch.cuda.empty_cache()
    return rows


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    distill = ROOT / "data/cosf_distill_verified.jsonl"
    skip = set()
    if distill.exists():
        for line in open(distill, encoding="utf-8"):
            skip.add(json.loads(line)["turns"][0][0].lower())
    exam = make_exam(skip=skip)
    kinds = list(dict.fromkeys(k for k, _, _ in exam))
    out_dir = ROOT / "experiments/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for ckpt in sys.argv[1:]:
        rows = run(ckpt, exam, device)
        name = Path(ckpt).parent.name
        results[name] = rows
        (out_dir / f"math_eval_{name}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")

    print(f"\nMATH EXAM ({len(exam)} questions, correct = last number in the reply)")
    print(f"{'kind':14s}" + "".join(f"{n[:22]:>24s}" for n in results))
    for kind in kinds + ["TOTAL"]:
        cells = []
        for rows in results.values():
            sel = [r for r in rows if kind in ("TOTAL", r["kind"])]
            cells.append(f"{sum(r['correct'] for r in sel):>3d}/{len(sel):<3d} ({100 * sum(r['correct'] for r in sel) / len(sel):3.0f}%)")
        print(f"{kind:14s}" + "".join(f"{c:>24s}" for c in cells))
    for name, rows in results.items():
        print(f"\n{name} examples:")
        for r in rows[::20][:6]:
            print(f"  {'OK ' if r['correct'] else 'XX '} {r['q']}  ->  {r['reply'][:90]}")


if __name__ == "__main__":
    main()
