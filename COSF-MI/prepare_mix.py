"""
Build the "smarter AURA" pretraining mix in data/mix/.

Sources (streamed from Hugging Face, personal research use):
  wiki     - Simple English Wikipedia (facts, plain language)       CC-BY-SA
  web      - FineWeb-Edu sample-10BT (educational web pages)         ODC-BY
  textbook - Cosmopedia v2 (synthetic textbooks for small models)    ODC-BY
  code     - python-edu (GitHub Python, top educational score)       per-repo permissive
  stories  - TinyStories (keeps the storytelling)                    CDLA-Sharing
  math     - generated arithmetic / word problems with worked answers

Documents are separated by DOC_SEP. ~1% of each source goes to validation.
Writes data/mix/styled_train.txt + styled_val.txt (names train_overnight.py expects).

Usage:  python prepare_mix.py [--scale 1.0]
"""

import argparse
import random
from pathlib import Path

DOC_SEP = "\n<|endoftext|>\n"
OUT = Path("data/mix")

# target characters per source (scale=1 -> ~1.3 GB of text)
TARGETS = {
    "wiki": 250e6,
    "web": 400e6,
    "textbook": 300e6,
    "code": 130e6,
    "stories": 150e6,
    "math": 40e6,
}


def stream(name, config=None, text_key="text", keep=lambda ex: True):
    from datasets import load_dataset
    ds = load_dataset(name, config, split="train", streaming=True) if config else \
        load_dataset(name, split="train", streaming=True)
    for ex in ds:
        if keep(ex):
            t = (ex[text_key] or "").strip()
            if len(t) > 200:
                yield t


def wiki():
    return stream("wikimedia/wikipedia", "20231101.simple")


def web():
    return stream("HuggingFaceFW/fineweb-edu", "sample-10BT", keep=lambda ex: ex["int_score"] >= 3)


def textbook():
    return stream("HuggingFaceTB/smollm-corpus", "cosmopedia-v2")


def code():
    return stream("Avelina/python-edu-cleaned", keep=lambda ex: int(ex["int_score"]) >= 4)


def stories():
    return stream("roneneldan/TinyStories")


def math_problems(seed=0):
    """Endless arithmetic with worked answers, in a few plain-English formats."""
    rng = random.Random(seed)
    names = ["Tom", "Mia", "Sam", "Lily", "Ben", "Ana", "Max", "Zoe"]
    things = ["apples", "books", "coins", "marbles", "stickers", "cookies", "pencils"]
    while True:
        lines = []
        for _ in range(rng.randint(4, 10)):
            kind = rng.random()
            a, b = rng.randint(0, 999 if rng.random() < 0.3 else 99), rng.randint(0, 99)
            if kind < 0.25:
                lines.append(f"Q: What is {a} + {b}?\nA: {a} + {b} = {a + b}.")
            elif kind < 0.45:
                a, b = max(a, b), min(a, b)
                lines.append(f"Q: What is {a} - {b}?\nA: {a} - {b} = {a - b}.")
            elif kind < 0.65:
                a, b = rng.randint(0, 12), rng.randint(0, 12)
                lines.append(f"Q: What is {a} x {b}?\nA: {a} x {b} = {a * b}.")
            elif kind < 0.75:
                b = rng.randint(1, 12)
                q = rng.randint(0, 12)
                lines.append(f"Q: What is {b * q} / {b}?\nA: {b * q} / {b} = {q}.")
            elif kind < 0.9:
                n, t = rng.choice(names), rng.choice(things)
                a, b = rng.randint(1, 30), rng.randint(1, 30)
                if rng.random() < 0.5:
                    lines.append(f"Q: {n} has {a} {t}. {n} gets {b} more. How many {t} does {n} have now?\n"
                                 f"A: {n} starts with {a}. {a} + {b} = {a + b}. {n} has {a + b} {t}.")
                else:
                    a, b = max(a, b), min(a, b)
                    lines.append(f"Q: {n} has {a} {t} and gives away {b}. How many are left?\n"
                                 f"A: {a} - {b} = {a - b}. {n} has {a - b} {t} left.")
            else:
                x, y = rng.randint(1, 50), rng.randint(1, 50)
                bigger = "bigger" if x > y else "smaller" if x < y else "equal"
                if bigger == "equal":
                    lines.append(f"Q: Is {x} bigger or smaller than {y}?\nA: {x} and {y} are equal.")
                else:
                    lines.append(f"Q: Is {x} bigger or smaller than {y}?\nA: {x} is {bigger} than {y}.")
        yield "\n\n".join(lines)


SOURCES = {"wiki": wiki, "web": web, "textbook": textbook, "code": code,
           "stories": stories, "math": math_problems}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(1234)

    for name, gen in SOURCES.items():
        if args.only and name not in args.only:
            continue
        target = TARGETS[name] * args.scale
        part_tr, part_va = OUT / f"{name}_train.txt", OUT / f"{name}_val.txt"
        if part_tr.exists() and part_tr.stat().st_size >= target * 0.95:
            print(f"{name}: already done ({part_tr.stat().st_size / 1e6:.0f} MB)", flush=True)
            continue
        n_chars = n_docs = 0
        with open(part_tr, "w", encoding="utf-8") as ftr, open(part_va, "w", encoding="utf-8") as fva:
            for doc in gen():
                (fva if rng.random() < 0.01 else ftr).write(doc + DOC_SEP)
                n_chars += len(doc)
                n_docs += 1
                if n_docs % 2000 == 0:
                    pct = min(100, 100 * n_chars / target)
                    print(f"{name}: {pct:5.1f}%  {n_docs:,} docs  {n_chars / 1e6:.0f}M chars", flush=True)
                if n_chars >= target:
                    break
        print(f"{name}: done, {n_docs:,} docs, {n_chars / 1e6:.0f}M chars", flush=True)

    # shuffle documents from all sources together into the final files
    for split in ["train", "val"]:
        docs = []
        for name in SOURCES:
            p = OUT / f"{name}_{split}.txt"
            if p.exists():
                docs += [d for d in p.read_text(encoding="utf-8").split(DOC_SEP) if d.strip()]
        rng.shuffle(docs)
        with open(OUT / f"styled_{split}.txt", "w", encoding="utf-8") as f:
            for d in docs:
                f.write(d + DOC_SEP)
        print(f"styled_{split}.txt: {len(docs):,} docs", flush=True)
    print("Done.")


if __name__ == "__main__":
    main()
