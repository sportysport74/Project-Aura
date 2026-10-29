"""
× / ÷ data-vs-capacity test (2026-10-28): greedy exact match per digit cell, on problems the model trained on ("fit") and
on held-out problems it never saw ("held"), plus the 410-problem arithmetic battery.

  fit   15 problems per (op, digits a, digits b) cell, drawn from data/arith_md/n25k/train.jsonl (non-facts). n25k is a
        prefix of n100k (same seed), so both arms trained on them: the 25k arm 4×, the 100k arm once.
  held  15 per cell from common.heldout() triples (never generated for training). The 1x1 cells are facts, always trained.

Reading (decided before the run): capacity if the model can't fit even its training problems and 4× the unique data doesn't
help; data if held-out accuracy rises with the 100k arm (same steps, 4× unique problems).

Usage (from COSF-MI/): .venv/bin/python experiments/arith_md_eval.py --out experiments/results/arith_md NAME=CKPT [...]
"""
import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "scripts" / "distill"))
import ab_eval as ab  # noqa: E402
import build_arith_plain as B  # noqa: E402
import common as C  # noqa: E402
from arith_battery import battery  # noqa: E402

PER_CELL = 15


def sets(seed=7):
    rng = random.Random(seed)
    train = [tuple(json.loads(line)["meta"].split("|")) for line in open(ROOT / "data/arith_md/n25k/train.jsonl")]
    train = [(op, int(a), int(b)) for op, a, b in train]
    fit, held = defaultdict(list), defaultdict(list)
    for t in rng.sample(train, len(train)):
        if t[0] in "*/" and not B.is_fact(*t) and len(fit[B.cell(*t)]) < PER_CELL and t not in fit[B.cell(*t)]:
            fit[B.cell(*t)].append(t)
    for op in "*/":
        for da in range(1, 5):
            for db in range(1, 5):
                tries = 0
                while len(held[f"{B.SYM[op]} {da}x{db}"]) < PER_CELL and tries < 200_000:
                    tries += 1
                    t = B.problem(rng, op, da, db)
                    c = B.cell(*t)
                    if (B.is_fact(*t) or C.heldout(*t)) and t not in held[c]:
                        held[c].append(t)
    return fit, held


def score(model, items):
    prompts = [f"What is {a} {B.SYM[op]} {b}?" for op, a, b in items]
    out = ab.generate(model, ab.tok_global, prompts, greedy=True, max_new=40)
    return [ab.last_number(r) == B.GOLD[op](a, b) for (op, a, b), (r, _) in zip(items, out)], [r for r, _ in out]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpts", nargs="+", help="NAME=path/to/best_model.pt (repo-relative)")
    p.add_argument("--out", default="experiments/results/arith_md")
    args = p.parse_args()
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    ab.tok_global = ab.BPETokenizer(str(ROOT / "models/tokenizer_mix16k.model"))
    fit, held = sets()
    bat = battery()
    results = {}
    for spec in args.ckpts:
        name, path = spec.split("=", 1)
        ckpt = torch.load(ROOT / path, map_location="cpu", weights_only=False)
        model = ab.COSFOscillatorTransformer(ckpt["config"]).to("cuda")
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        res = {"name": name, "checkpoint": path, "fit": {}, "held": {}, "battery": {}}
        for kind, cells in (("fit", fit), ("held", held)):
            for c, items in sorted(cells.items()):
                ok, replies = score(model, items)
                res[kind][c] = {"acc": sum(ok) / len(ok), "n": len(ok),
                                "examples": [[f"{a} {B.SYM[op]} {b}", r] for (op, a, b), r in zip(items[:3], replies[:3])]}
        ok, _ = score(model, [(b["op"], b["a"], b["b"]) for b in bat])
        by = defaultdict(list)
        for b, k in zip(bat, ok):
            by[b["cell"]].append(k)
        res["battery"] = {c: sum(v) / len(v) for c, v in by.items()} | {"ALL": sum(ok) / len(ok)}
        (out_dir / f"{name}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
        results[name] = res
        print(f"[{name}] battery {res['battery']['ALL']:.0%}", flush=True)
        del model
        torch.cuda.empty_cache()
    names = list(results)
    L = ["# × / ÷ data-vs-capacity test\n", "Greedy exact match, 15 problems per cell. fit = trained on (both arms); "
         "held = never generated. 1x1 = facts (always trained). Cell = digits of a x b (÷: quotient x divisor).\n"]
    for op in "×÷":
        L.append(f"\n## {op}\n\n| cell | " + " | ".join(f"{n} fit | {n} held" for n in names) + " |\n|---|"
                 + "---|---|" * len(names))
        for c in sorted(held):
            if c.startswith(op):
                L.append(f"| {c} | " + " | ".join(f"{results[n]['fit'].get(c, {}).get('acc', float('nan')):.0%} | "
                                                 f"{results[n]['held'][c]['acc']:.0%}" for n in names) + " |")
    L.append("\n## Battery (410)\n\n| cell | " + " | ".join(names) + " |\n|---|" + "---|" * len(names))
    for c in results[names[0]]["battery"]:
        L.append(f"| {c} | " + " | ".join(f"{results[n]['battery'][c]:.0%}" for n in names) + " |")
    (out_dir / "summary.md").write_text("\n".join(L) + "\n")
    print(re.sub(r"\n{3,}", "\n\n", "\n".join(L)))


if __name__ == "__main__":
    main()
