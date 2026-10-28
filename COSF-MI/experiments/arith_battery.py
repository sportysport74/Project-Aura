"""
Arithmetic battery for the distill A/B: exact-match accuracy by operation × digits.

Problems come from the triples that scripts/distill/common.py reserves with heldout()
(one in ten, never generated). The 1-digit cells are too small for that (e.g. 81 sums
of two 1-digit numbers, ~8 held out), so they are filled up with other triples, and
excluded_triples() lists every triple used here (plus the readable set's maths and the
120-question math_eval exam) so build_finetune_set.py can drop them from training.

Usage (from COSF-MI/): imported by scripts/distill/build_finetune_set.py and experiments/ab_eval.py
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "distill"))
from common import heldout  # noqa: E402

SYM = {"+": "+", "-": "-", "*": "×", "/": "÷"}
GOLD = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b, "/": lambda a, b: a // b}
READABLE = ROOT / "experiments" / "readable_40.json"


def _space(op, d):
    """Every (a, b) in a cell. +, -, ×: both operands d digits (× 1-digit: 2-9, as in training).
    ÷: d-digit divisor, quotient 2-99, always exact."""
    lo, hi = (1 if d == 1 else 10 ** (d - 1)), 10 ** d - 1
    if op == "*" and d == 1:
        lo = 2
    if op == "/":
        return [(b * q, b) for b in range(max(2, lo), hi + 1) for q in range(2, 100)]
    pairs = [(a, b) for a in range(lo, hi + 1) for b in range(lo, hi + 1)]
    return [(a, b) for a, b in pairs if op != "-" or a >= b]


CELLS = [(op, d) for op in "+-*" for d in (1, 2, 3)] + [("/", 1), ("/", 2)]


def battery(per_cell=40, per_cell_1digit=30, seed=4242):
    """[{cell, op, a, b, prompt, gold, heldout}], fixed by the seed."""
    rng = random.Random(seed)
    out = []
    for op, d in CELLS:
        n = per_cell_1digit if d == 1 and op != "/" else per_cell
        space = _space(op, d)
        held = [p for p in space if heldout(op, *p)]
        rest = [p for p in space if not heldout(op, *p)]
        rng.shuffle(held)
        rng.shuffle(rest)
        pick = held[:n] + rest[:max(0, n - len(held))]
        cell = f"{SYM[op]} {d}-digit" + (" divisor" if op == "/" else "")
        for a, b in pick:
            out.append({"cell": cell, "op": op, "a": a, "b": b, "prompt": f"What is {a} {SYM[op]} {b}?",
                        "gold": GOLD[op](a, b), "heldout": heldout(op, a, b)})
    return out


def _math_eval_triples():
    """(op, a, b) of the 120-question math_eval exam (same skip set as experiments/math_eval.py)."""
    sys.path.insert(0, str(ROOT / "experiments"))
    import math_eval
    skip = set()
    distill = ROOT / "data/cosf_distill_verified.jsonl"
    if distill.exists():
        for line in open(distill, encoding="utf-8"):
            skip.add(json.loads(line)["turns"][0][0].lower())
    out = []
    for kind, q, a in math_eval.make_exam(skip=skip):
        nums = [int(x) for x in __import__("re").findall(r"\d+", q)]
        if kind in ("add 1-digit", "add 2-digit", "word problem"):
            out.append(("+", nums[-2], nums[-1]))
        elif kind == "subtract":
            out.append(("-", nums[0], nums[1]))
        elif kind == "multiply":
            out.append(("*", nums[0], nums[1]))
    return out


def excluded_triples():
    """Every (op, a, b) an eval uses; + and × in both orders."""
    ts = [(r["op"], r["a"], r["b"]) for r in battery()]
    ts += [tuple(t) for t in json.loads(READABLE.read_text())["maths_triples"]]
    ts += _math_eval_triples()
    out = set()
    for op, a, b in ts:
        out.add((op, a, b))
        if op in "+*":
            out.add((op, b, a))
    return out
