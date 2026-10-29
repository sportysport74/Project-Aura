"""
Plain-format programmatic arithmetic (no teacher, licence "ours"): "47 × 38 = 1786" then "Answer: 1786".

Used for two things (scaling plan step 0 follow-ups, 2026-10-28):
  1. the × / ÷ data-vs-capacity test: --mix "*:0.4,/:0.4,+:0.1,-:0.1", --n 25000 / 100000 (finetune_chat.py --pilot dirs)
  2. the clean corpus's arithmetic slot: --mix "+:0.25,-:0.25,*:0.25,/:0.25" --n <millions> (build_corpus.py source aura_arith)

Problems: each operand's digit count is drawn uniformly from 1-4, so the 1-digit facts (the times tables and the one-digit
sums and differences) get as many problems as the 4-digit ones and repeat often, which drills them. ÷ is the inverse of ×:
(a × b) ÷ b, always exact, so the divisor and the quotient both have 1-4 digits. × and ÷ never use 0 or 1 as an operand
(battery convention: × 1-digit is 2-9); − keeps a ≥ b.

Held-out policy:
  * Every multi-digit triple an eval uses (arith_battery.excluded_triples: battery, readable set, math_eval exam) and every
    common.heldout() triple (one in ten) is never generated.
  * **Exception: facts.** Triples whose operands are both 1-digit (and ÷ facts: divisor and quotient both 1-digit) are always
    allowed. There are only 81 + 45 + 64 + 64 of them and the battery's 1-digit cells reserve 30 of each, so excluding them
    would leave half the times table untaught. From here on the battery's 1-digit cells measure fact recall, not
    generalization. --exclude-facts restores the strict rule.
  * Non-fact problems appear at most once; facts repeat.

Output (--out DIR): train.jsonl, val.jsonl (finetune_chat.py --pilot format: {"category", "meta", "prompt", "reply", "line"})
and report.json. Val problems are non-facts and never in train.

Usage (from COSF-MI/):
  .venv/bin/python scripts/distill/build_arith_plain.py --n 100000 --mix "*:0.4,/:0.4,+:0.1,-:0.1" --out data/arith_md/n100k
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "experiments"))
import common as C  # noqa: E402
from arith_battery import excluded_triples  # noqa: E402

SYM = {"+": "+", "-": "-", "*": "×", "/": "÷"}
ASKS = ["What is {a} {s} {b}?", "Can you work out {a} {s} {b}?", "{a} {s} {b} = ?", "Please calculate {a} {s} {b}."]
GOLD = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b, "/": lambda a, b: a // b}


def ndigits(n):
    return len(str(n))


def operand(rng, d, op):
    lo = (2 if op in "*/" else 1) if d == 1 else 10 ** (d - 1)
    return rng.randint(lo, 10 ** d - 1)


def problem(rng, op, da=None, db=None):
    """(op, a, b) with operand digit counts da, db (uniform 1-4 when None). For ÷, da/db are the quotient/divisor digits."""
    da, db = da or rng.randint(1, 4), db or rng.randint(1, 4)
    a, b = operand(rng, da, op), operand(rng, db, op)
    if op == "-" and b > a:
        a, b = b, a
    if op == "/":
        a = a * b  # a was the quotient
    return op, a, b


def is_fact(op, a, b):
    return (a // b < 10 and b < 10) if op == "/" else (a < 10 and b < 10)


def cell(op, a, b):
    """'× 3x2' style cell name: operand digit counts (÷: quotient x divisor)."""
    return f"{SYM[op]} {ndigits(a // b if op == '/' else a)}x{ndigits(b)}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, required=True, help="train problems")
    p.add_argument("--n-val", type=int, default=2_000)
    p.add_argument("--mix", default="+:0.25,-:0.25,*:0.25,/:0.25", help="op:weight,...")
    p.add_argument("--out", required=True)
    p.add_argument("--exclude-facts", action="store_true", help="also hold out the eval's 1-digit facts (strict rule)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    mix = {k: float(v) for k, v in (x.split(":") for x in args.mix.split(","))}
    rng = random.Random(args.seed)
    exclude = excluded_triples()
    seen, val, train = set(), [], []
    skipped = Counter()

    def ok(t):
        if is_fact(*t) and not args.exclude_facts:
            return True
        if t in exclude or C.heldout(*t):
            skipped["eval/heldout"] += 1
            return False
        if t in seen:
            skipped["dup"] += 1
            return False
        return True

    ops, weights = list(mix), list(mix.values())
    while len(val) < args.n_val:  # val first: non-facts only, so it never overlaps train
        t = problem(rng, rng.choices(ops, weights)[0])
        if not is_fact(*t) and ok(t):
            seen.add(t)
            val.append(t)
    while len(train) < args.n:
        t = problem(rng, rng.choices(ops, weights)[0])
        if ok(t):
            if not is_fact(*t):
                seen.add(t)
            train.append(t)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for split, rows in (("train", train), ("val", val)):
        with open(out / f"{split}.jsonl", "w") as f:
            for op, a, b in rows:
                g = GOLD[op](a, b)
                prompt = rng.choice(ASKS).format(a=a, s=SYM[op], b=b)
                reply = f"{a} {SYM[op]} {b} = {g}\n\nAnswer: {g}"
                f.write(json.dumps({"category": "arith", "meta": f"{op}|{a}|{b}", "prompt": prompt, "reply": reply,
                                    "line": f"[INST] {prompt} [/INST] {reply}"}, ensure_ascii=False) + "\n")
    report = {"teacher": "none (programmatic arithmetic)", "teacher_license": "ours", "format": "plain", "mix": mix,
              "train": len(train), "val": len(val), "unique_train": len(set(train)),
              "facts_in_train": sum(is_fact(*t) for t in train), "exclude_facts": args.exclude_facts,
              "ops": dict(Counter(t[0] for t in train)), "cells": dict(sorted(Counter(cell(*t) for t in train).items())),
              "skipped": dict(skipped), "seed": args.seed}
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != "cells"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
