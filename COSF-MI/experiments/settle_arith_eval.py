"""
Settling retest eval (2026-11-08, experiments/plans/settling_retest.md, frozen before any run).

  testset   Write the fixed test set: for every non-fact × and ÷ cell (cell = digits of a x digits of b; ÷: quotient x divisor),
            all held-out problems (common.heldout, never generated for training) up to 200 per cell, drawn with seed 0 when a cell
            has more. Records the actual n per cell. Cells with n < 50 are secondary only.
  eval      Greedy exact match of each checkpoint on the test set (+ a small + / − check), prompt "a × b = ?" in the corpus
            transcript format ("You: … \\nAura:", encoded in one go, as the arithmetic stream was written). Writes per-problem
            results to <out>/<NAME>.jsonl and per-cell accuracy to <out>/<NAME>.cells.json.
  floor     Step 1: vanilla2's mean accuracy per cell over the given runs; primary test range = cells with n >= 50 and mean
            accuracy in [20%, 60%]. Writes <out>/floor.md and <out>/primary_cells.json.

Usage (from COSF-MI/):
  .venv/bin/python experiments/settle_arith_eval.py testset --out experiments/results/settle_retest
  .venv/bin/python experiments/settle_arith_eval.py eval --out experiments/results/settle_retest A_s1=checkpoints/settle_retest/A_s1/final_model.pt
  .venv/bin/python experiments/settle_arith_eval.py floor --out experiments/results/settle_retest A_s1 A_s2
"""
import argparse
import itertools
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "scripts" / "distill"))
sys.path.insert(0, str(ROOT / "src" / "python"))

import build_arith_plain as B  # noqa: E402
import common as C  # noqa: E402

CAP, MIN_N, LO, HI = 200, 50, 0.20, 0.60
ENUM_LIMIT = 3_000_000  # cells with at most this many (a, b) pairs are enumerated in full; larger ones are sampled


def _range(d, op):
    lo = (2 if op in "*/" else 1) if d == 1 else 10 ** (d - 1)
    return lo, 10 ** d - 1


def _triple(op, x, b):  # x = a for ×, the quotient for ÷
    return (op, x * b, b) if op == "/" else (op, x, b)


def cell_problems(op, da, db, rng, limit=None):
    (la, ha), (lb, hb) = _range(da, op), _range(db, op)
    size = (ha - la + 1) * (hb - lb + 1)
    cap = limit or CAP
    if size <= ENUM_LIMIT:
        pool = [t for x, b in itertools.product(range(la, ha + 1), range(lb, hb + 1))
                if C.heldout(*(t := _triple(op, x, b)))]
        return (rng.sample(pool, cap) if len(pool) > cap else pool), len(pool)
    got, tries = set(), 0
    while len(got) < cap and tries < 5_000_000:
        tries += 1
        t = _triple(op, rng.randint(la, ha), rng.randint(lb, hb))
        if C.heldout(*t):
            got.add(t)
    return sorted(got), None  # pool size not enumerated (≈ size / 10)


def cmd_testset(args):
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    rows, cells = [], {}
    for op in "*/":
        for da in range(1, 5):
            for db in range(1, 5):
                if da == 1 and db == 1:
                    continue  # facts: always trained
                probs, pool = cell_problems(op, da, db, rng, args.limit)
                name = f"{B.SYM[op]} {da}x{db}"
                assert all(B.cell(*t) == name and not B.is_fact(*t) for t in probs), name
                cells[name] = {"n": len(probs), "heldout_pool": pool, "primary_eligible": len(probs) >= MIN_N}
                rows += [{"cell": name, "op": t[0], "a": t[1], "b": t[2]} for t in probs]
    # small + / − check (secondary; the plan reports + and − cells)
    for op in "+-":
        for d in (2, 3):
            probs, pool = cell_problems(op, d, d, rng, 50 if args.limit is None else args.limit)
            name = f"{B.SYM[op]} {d}x{d}"
            cells[name] = {"n": len(probs), "heldout_pool": pool, "primary_eligible": False}
            rows += [{"cell": name, "op": t[0], "a": t[1], "b": t[2]} for t in probs]
    with open(out / "test_set.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "test_cells.json").write_text(json.dumps(cells, indent=1, ensure_ascii=False))
    for k, v in cells.items():
        print(f"{k}: n={v['n']}{'' if v['primary_eligible'] else '  (secondary only)'}")
    print(f"{len(rows)} problems -> {args.out}/test_set.jsonl")


def cmd_eval(args):
    import torch
    import ab_eval as ab
    from cosf_mi.models.cosf_oscillator import COSFOscillatorTransformer
    out = ROOT / args.out
    rows = [json.loads(line) for line in open(out / "test_set.jsonl")]
    if args.max_per_cell:
        per = defaultdict(int)
        keep = []
        for r in rows:
            if per[r["cell"]] < args.max_per_cell:
                per[r["cell"]] += 1
                keep.append(r)
        rows = keep
    for spec in args.ckpts:
        name, path = spec.split("=", 1)
        ck = torch.load(ROOT / path, map_location="cpu", weights_only=False)
        model = COSFOscillatorTransformer(ck["config"]).to("cuda").eval()
        model.load_state_dict(ck["model_state_dict"])
        tok = ab.BPETokenizer(str(ROOT / "models/tokenizer_clean24k.model"))
        prompts = [f"{r['a']} {B.SYM[r['op']]} {r['b']} = ?" for r in rows]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            gen = ab.generate(model, tok, prompts, greedy=True, max_new=40, bs=args.bs, joined=True)
        by = defaultdict(list)
        with open(out / f"{name}.jsonl", "w") as f:
            for r, (reply, _) in zip(rows, gen):
                ok = ab.last_number(reply) == B.GOLD[r["op"]](r["a"], r["b"])
                by[r["cell"]].append(ok)
                f.write(json.dumps({"cell": r["cell"], "op": r["op"], "a": r["a"], "b": r["b"], "ok": bool(ok),
                                    "reply": reply[:120]}, ensure_ascii=False) + "\n")
        cells = {c: {"acc": sum(v) / len(v), "n": len(v)} for c, v in by.items()}
        (out / f"{name}.cells.json").write_text(json.dumps({"checkpoint": path, "cells": cells}, indent=1, ensure_ascii=False))
        print(f"[{name}] " + "  ".join(f"{c} {v['acc']:.0%}" for c, v in cells.items()), flush=True)
        del model
        torch.cuda.empty_cache()


def cmd_floor(args):
    out = ROOT / args.out
    tc = json.loads((out / "test_cells.json").read_text())
    runs = {n: json.loads((out / f"{n}.cells.json").read_text())["cells"] for n in args.names}
    L = ["# Settling retest, step 1: vanilla2 floor check\n",
         f"vanilla2 s25 trained on arithmetic only (200M tokens), runs: {', '.join(args.names)}. Greedy exact match on held-out problems.",
         f"Primary test range (frozen rule): cells with n >= {MIN_N} and mean accuracy over the runs in [{LO:.0%}, {HI:.0%}].\n",
         "| cell | n | " + " | ".join(args.names) + " | mean | class |", "|---|---|" + "---|" * (len(args.names) + 2)]
    primary = []
    for c, info in tc.items():
        accs = [runs[n][c]["acc"] for n in args.names]
        m = sum(accs) / len(accs)
        if c[0] in "+-":
            cls = "secondary (+/−)"
        elif not info["primary_eligible"]:
            cls = f"secondary (n < {MIN_N})"
        elif LO <= m <= HI:
            cls = "**primary**"
            primary.append({"cell": c, "floor_acc": m, "n": info["n"]})
        else:
            cls = "easy (> 60%)" if m > HI else "hard (< 20%)"
        L.append(f"| {c} | {info['n']} | " + " | ".join(f"{a:.1%}" for a in accs) + f" | {m:.1%} | {cls} |")
    primary.sort(key=lambda r: -r["floor_acc"])  # easiest first (the difficulty order for the trend rule)
    L.append(f"\n**Primary cells: {len(primary)}** ({', '.join(p['cell'] for p in primary) or 'none'}).")
    if len(primary) < 3:
        L.append("**Fewer than 3 cells in range: per the plan, nothing is changed; this goes to Jon and the queue stops.**")
    else:
        k = len(primary) // 2  # middle cell (odd count) goes to the harder half
        L.append(f"Difficulty order (easier half first): easier = {', '.join(p['cell'] for p in primary[:k])}; "
                 f"harder = {', '.join(p['cell'] for p in primary[k:])}.")
    (out / "primary_cells.json").write_text(json.dumps({"primary": primary, "enough": len(primary) >= 3}, indent=1, ensure_ascii=False))
    (out / "floor.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return len(primary) >= 3


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("testset")
    a.add_argument("--out", required=True)
    a.add_argument("--limit", type=int, default=None, help="per-cell cap (default 200; smoke tests use a small one)")
    a = sub.add_parser("eval")
    a.add_argument("--out", required=True)
    a.add_argument("--bs", type=int, default=256)
    a.add_argument("--max-per-cell", type=int, default=0)
    a.add_argument("ckpts", nargs="+", help="NAME=path/to/final_model.pt (repo-relative)")
    a = sub.add_parser("floor")
    a.add_argument("--out", required=True)
    a.add_argument("names", nargs="+")
    args = p.parse_args()
    if args.cmd == "testset":
        cmd_testset(args)
    elif args.cmd == "eval":
        cmd_eval(args)
    else:
        sys.exit(0 if cmd_floor(args) else 3)


if __name__ == "__main__":
    main()
