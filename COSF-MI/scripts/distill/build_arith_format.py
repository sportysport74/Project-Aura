"""
Arithmetic-format test data (scaling plan step 0): the same problems in two reply formats, no teacher.

  plain    "47 + 38 = 85" then "Answer: 85" (the distill data's format)
  scratch  the working written digit by digit, right to left, with every carry and borrow, then "Answer: 85".
           +: column addition; -: column subtraction with borrows; ×: one partial product per multiplier digit
           (each with its carries), then the partials added; ÷: long division, one "bring down" step per digit.

Problems cover the arithmetic battery's cells (+ − × with 1-3 digit operands, ÷ with a 1-2 digit divisor and a
2-99 quotient), and skip every eval triple (arith_battery.excluded_triples) and every held-out triple
(common.heldout). A problem is dropped from BOTH formats if either reply is longer than --max-aura-tokens, so the
two sets always hold the same problems. Output: <out>/{plain,scratch}/{train,val}.jsonl + report.json, in the
format finetune_chat.py --pilot reads.

Usage (from COSF-MI/; any venv with sentencepiece):
  .venv/bin/python scripts/distill/build_arith_format.py --n 150000 --out data/arith_format
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
PLACE = ["ones", "tens", "hundreds", "thousands", "ten-thousands", "hundred-thousands", "millions"]
ASKS = ["What is {a} {s} {b}?", "Can you work out {a} {s} {b}?", "{a} {s} {b} = ?", "Please calculate {a} {s} {b}."]


def digits(n):
    return [int(c) for c in str(n)][::-1]  # least significant first


def scratch_add(a, b):
    da, db = digits(a), digits(b)
    lines, carry, out = ["Add the columns from the right."], 0, []
    for i in range(max(len(da), len(db))):
        x, y = (da[i] if i < len(da) else 0), (db[i] if i < len(db) else 0)
        s = x + y + carry
        expr = f"{x} + {y}" + (f" + {carry}" if carry else "")
        carry, w = divmod(s, 10)
        lines.append(f"{PLACE[i].capitalize()}: {expr} = {s}, write {w}" + (f", carry {carry}." if carry else "."))
        out.append(w)
    if carry:
        lines.append(f"{PLACE[len(out)].capitalize()}: carry {carry}, write {carry}.")
    return lines


def scratch_sub(a, b):
    da, db = digits(a), digits(b)
    lines, borrow = ["Subtract the columns from the right."], 0
    for i in range(len(da)):
        x, y = da[i] - borrow, (db[i] if i < len(db) else 0)
        top = f"{da[i]} - {borrow}" if borrow else f"{da[i]}"
        if x < y:
            lines.append(f"{PLACE[i].capitalize()}: {top} is less than {y}, borrow 1: {x + 10} - {y} = {x + 10 - y}, "
                         f"write {x + 10 - y}.")
            borrow = 1
        else:
            lines.append(f"{PLACE[i].capitalize()}: {top} - {y} = {x - y}, write {x - y}." if borrow
                         else f"{PLACE[i].capitalize()}: {x} - {y} = {x - y}, write {x - y}.")
            borrow = 0
    return lines


def times_one_digit(a, d):
    """'347 × 6: 7 × 6 = 42, write 2 carry 4; ...' → (text, product)."""
    parts, carry = [], 0
    da = digits(a)
    for i, x in enumerate(da):
        p = x * d + carry
        expr = f"{x} × {d}" + (f" + {carry}" if carry else "")
        carry, w = divmod(p, 10)
        last = i == len(da) - 1
        parts.append(f"{expr} = {p}, write {p if last else w}" + ("" if last or not carry else f" carry {carry}"))
    return f"{a} × {d}: " + "; ".join(parts) + f", so {a * d}", a * d


def scratch_mul(a, b):
    lines, partials = ["Multiply by each digit of the second number, then add."], []
    for i, d in enumerate(digits(b)):
        text, p = times_one_digit(a, d)
        shift = 10 ** i
        lines.append(text + (f", times {shift} = {p * shift}." if shift > 1 else "."))
        partials.append(p * shift)
    if len(partials) > 1:
        total = 0
        for p in partials:
            prev, total = total, total + p
            if prev:
                lines.append(f"{prev} + {p} = {total}.")
    return lines


def scratch_div(a, b):
    lines, rem, q = [f"Long division: {a} ÷ {b}, one digit at a time from the left."], 0, []
    for ch in str(a):
        cur = rem * 10 + int(ch)
        qd = cur // b
        rem = cur - qd * b
        q.append(qd)
        lines.append(f"Bring down {ch}: {cur}. {cur} ÷ {b} = {qd}, {qd} × {b} = {qd * b}, remainder {rem}.")
    return lines


def problem(rng):
    op = rng.choice("+-*/")
    if op == "/":
        b = rng.randint(2, 9) if rng.random() < 0.5 else rng.randint(10, 99)
        return op, b * rng.randint(2, 99), b
    d = rng.choice([1, 2, 3])
    lo, hi = (2 if op == "*" else 1) if d == 1 else 10 ** (d - 1), 10 ** d - 1
    a, b = rng.randint(lo, hi), rng.randint(lo, hi)
    if op == "-" and b > a:
        a, b = b, a
    return op, a, b


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=150_000, help="train problems (val: --n-val more)")
    p.add_argument("--n-val", type=int, default=2_000)
    p.add_argument("--out", default="data/arith_format")
    p.add_argument("--max-aura-tokens", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = random.Random(args.seed)
    aura_len = C.AuraLength()
    exclude = excluded_triples()
    gold = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b, "/": lambda a, b: a // b}
    work = {"+": scratch_add, "-": scratch_sub, "*": scratch_mul, "/": scratch_div}
    seen, rows, dropped = set(), [], Counter()
    while len(rows) < args.n + args.n_val:
        op, a, b = problem(rng)
        if (op, a, b) in seen or (op, a, b) in exclude or C.heldout(op, a, b):
            continue
        seen.add((op, a, b))
        g = gold[op](a, b)
        prompt = rng.choice(ASKS).format(a=a, s=SYM[op], b=b)
        plain = f"{a} {SYM[op]} {b} = {g}\n\nAnswer: {g}"
        scratch = "\n".join(work[op](a, b)) + f"\nSo {a} {SYM[op]} {b} = {g}.\n\nAnswer: {g}"
        if max(aura_len(prompt, plain), aura_len(prompt, scratch)) > args.max_aura_tokens:
            dropped[op] += 1
            continue
        rows.append((op, a, b, prompt, plain, scratch))
    out = Path(args.out)
    for fmt, col in (("plain", 4), ("scratch", 5)):
        d = out / fmt
        d.mkdir(parents=True, exist_ok=True)
        for split, part in (("val", rows[:args.n_val]), ("train", rows[args.n_val:])):
            with open(d / f"{split}.jsonl", "w") as f:
                for r in part:
                    f.write(json.dumps({"category": "arith", "meta": f"{r[0]}|{r[1]}|{r[2]}", "prompt": r[3], "reply": r[col],
                                        "line": f"[INST] {r[3]} [/INST] {r[col]}"}) + "\n")
        tok = sum(aura_len(r[3], r[col]) for r in rows[args.n_val:])
        (d / "report.json").write_text(json.dumps({
            "teacher": "none (programmatic arithmetic)", "teacher_license": "ours", "format": fmt,
            "train": len(rows) - args.n_val, "val": args.n_val, "train_aura_tokens": tok,
            "ops": dict(Counter(r[0] for r in rows[args.n_val:])), "dropped_too_long": dict(dropped)}, indent=1))
        print(f"{fmt}: {len(rows) - args.n_val:,} train / {args.n_val:,} val, {tok / 1e6:.2f}M AURA tokens; "
              f"dropped too long: {dict(dropped)}")
    for op in "+-*/":
        ex = next(r for r in rows if r[0] == op and r[1] > 9)
        print(f"\n--- {ex[3]}\n{ex[5]}")


if __name__ == "__main__":
    main()
