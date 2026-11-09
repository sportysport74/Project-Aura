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


def _two_level(diff, rng, n_boot):
    """diff: (seeds, problems) paired per-problem differences. Resample seeds, then problems within each drawn seed."""
    S, P = diff.shape
    si = rng.integers(0, S, size=(n_boot, S))
    pi = rng.integers(0, P, size=(n_boot, S, P))
    return diff[si[:, :, None], pi].mean(axis=(1, 2))


def cmd_verdict(args):
    """Step-2 verdict, rules frozen in experiments/plans/settling_retest.md (Decision rules)."""
    import numpy as np
    out = ROOT / args.out
    prim = json.loads((out / "primary_cells.json").read_text())["primary"]
    order = [p["cell"] for p in prim]  # easiest first
    k = len(order) // 2
    easier, harder = set(order[:k]), set(order[k:])
    seeds = [1, 2, 3]
    arms = ["A", "B", "C", "D"]
    ref = None
    ok, rows_by = {}, {}
    for a in arms:
        for s in seeds:
            rows = [json.loads(line) for line in open(out / f"{a}_s{s}.jsonl")]
            key = [(r["cell"], r["a"], r["b"]) for r in rows]
            if ref is None:
                ref = key
            assert key == ref, f"{a}_s{s}: test-set order differs (pairing broken)"
            ok[a, s] = np.array([r["ok"] for r in rows], dtype=np.float64)
            rows_by[a, s] = rows
    cell = np.array([c for c, _, _ in ref])
    pm = np.isin(cell, order)
    em, hm = np.isin(cell, list(easier)), np.isin(cell, list(harder))
    M = {a: np.stack([ok[a, s] for s in seeds]) for a in arms}  # (3, n_test)
    acc = {a: M[a][:, pm].mean() for a in arms}
    rng = np.random.default_rng(20261108)
    L = ["# Settling retest, step 2: verdict\n",
         f"Primary cells: {', '.join(order)} ({int(pm.sum())} problems × 3 seeds per arm). Two-level bootstrap, {args.n_boot:,} resamples, "
         "seeds then problems within seed, paired across arms. Rules frozen in `experiments/plans/settling_retest.md`.\n",
         "| arm | seed 1 | seed 2 | seed 3 | pooled |", "|---|---|---|---|---|"]
    names = {"A": "A vanilla2 L8", "B": "B settling T=8", "C": "C repeat ×8", "D": "D vanilla2 L27"}
    for a in arms:
        L.append(f"| {names[a]} | " + " | ".join(f"{M[a][i, pm].mean():.1%}" for i in range(3)) + f" | {acc[a]:.1%} |")
    L += ["", "| contrast | diff | 95% CI | CI > 0 | B wins seeds | pass |", "|---|---|---|---|---|---|"]
    passes = []
    for c in ["A", "C", "D"]:
        d = (M["B"] - M[c])[:, pm]
        bs = _two_level(d, rng, args.n_boot)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        wins = [bool(M["B"][i, pm].mean() > M[c][i, pm].mean()) for i in range(3)]
        p = bool(lo > 0) and all(wins)
        passes.append(p)
        L.append(f"| B − {c} | {d.mean():+.1%} | [{lo:+.1%}, {hi:+.1%}] | {'yes' if lo > 0 else 'no'} | {sum(wins)}/3 | {'yes' if p else 'no'} |")
    verdict = "PASS" if all(passes) else "FAIL"
    L.append(f"\n**Verdict: {verdict}** (pass needs all three contrasts: CI above 0 and B ahead on every seed).")
    best = max(["A", "C", "D"], key=lambda a: acc[a])
    dh = (M["B"] - M[best])[:, hm]
    de = (M["B"] - M[best])[:, em]
    bh, be = _two_level(dh, rng, args.n_boot), _two_level(de, rng, args.n_boot)
    tr = bh - be
    lo, hi = np.percentile(tr, [2.5, 97.5])
    L.append(f"\nDifficulty trend (secondary): best control = {best}; (B − {best}) harder half {dh.mean():+.1%}, easier half {de.mean():+.1%}, "
             f"trend {dh.mean() - de.mean():+.1%} [{lo:+.1%}, {hi:+.1%}] → {'gap grows with difficulty' if lo > 0 else 'no trend shown'}.")
    # all cells, per arm (mean over seeds)
    cells = list(dict.fromkeys(cell.tolist()))
    L += ["\n## All cells (mean over 3 seeds)\n", "| cell | n | A | B | C | D | class |", "|---|---|---|---|---|---|---|"]
    for c in cells:
        m = cell == c
        cls = "primary" if c in order else ("+/−" if c[0] in "+-" else "other")
        L.append(f"| {c} | {int(m.sum())} | " + " | ".join(f"{M[a][:, m].mean():.1%}" for a in arms) + f" | {cls} |")
    # val loss + W_out movement from the checkpoints
    import torch
    L += ["\n## Val loss and settling weights\n", "| run | val loss | ‖w_out‖ per layer (mean) |", "|---|---|---|"]
    for a in arms:
        for s in seeds:
            ck = torch.load(ROOT / args.ckpt_dir / f"{a}_s{s}" / "final_model.pt", map_location="cpu", weights_only=False)
            wo = [v.float().norm().item() for kk, v in ck["model_state_dict"].items() if kk.endswith("settle.w_out.weight")]
            L.append(f"| {a}_s{s} | {ck.get('val_loss', float('nan')):.4f} | {sum(wo) / len(wo):.3f} ({len(wo)} layers) |" if wo
                     else f"| {a}_s{s} | {ck.get('val_loss', float('nan')):.4f} | — |")
    (out / "verdict.md").write_text("\n".join(L) + "\n")
    (out / "verdict.json").write_text(json.dumps({"verdict": verdict, "acc": {a: float(v) for a, v in acc.items()}, "passes": dict(zip("ACD", passes)), "best_control": best},
                                                 indent=1))
    print("\n".join(L))
    return verdict


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
    a = sub.add_parser("verdict")
    a.add_argument("--out", required=True)
    a.add_argument("--ckpt-dir", default="checkpoints/settle_retest")
    a.add_argument("--n-boot", type=int, default=10000)
    args = p.parse_args()
    if args.cmd == "testset":
        cmd_testset(args)
    elif args.cmd == "eval":
        cmd_eval(args)
    elif args.cmd == "verdict":
        cmd_verdict(args)
    else:
        sys.exit(0 if cmd_floor(args) else 3)


if __name__ == "__main__":
    main()
