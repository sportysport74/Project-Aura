"""
Paired comparison of knowledge-probe results across runs (2026-11-04): reads the per-question PMI from knowledge_pmi.py JSON files and
reports, for each requested pair, mean(B − A) with a paired bootstrap 95% CI over the 100 questions (10,000 draws, seed 1).

Usage (from COSF-MI/):
  .venv/bin/python experiments/knowledge_compare.py --json old=experiments/results/pretrain_10b/knowledge_pmi_10b.json \
      --json new=experiments/results/v2base/knowledge_pmi.json --pair old:base,new:b2 --pair new:b5,new:b10 [--out FILE.md]
A model is RUN:NAME (NAME as stored in that JSON's per_model).
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", action="append", required=True, help="RUN=path to a knowledge_pmi.py output")
    p.add_argument("--pair", action="append", required=True, help="RUN:NAME,RUN:NAME (A,B; reports B − A)")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    runs = {k: json.loads((ROOT / v).read_text())["per_model"] for k, v in (j.split("=", 1) for j in args.json)}

    def get(spec):
        run, name = spec.split(":", 1)
        return runs[run][name]["pmi"]
    L = ["| A | B | PMI A | PMI B | B − A | 95% CI |", "|---|---|---|---|---|---|"]
    rng = random.Random(1)
    for pair in args.pair:
        a_s, b_s = pair.split(",")
        a, b = get(a_s), get(b_s)
        assert len(a) == len(b), "different question sets"
        d = [y - x for x, y in zip(a, b)]
        boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(10_000))
        L.append(f"| {a_s} | {b_s} | {sum(a) / len(a):.4f} | {sum(b) / len(b):.4f} | {sum(d) / len(d):+.4f} | "
                 f"[{boots[249]:+.4f}, {boots[9749]:+.4f}] |")
    text = "\n".join(L) + "\n"
    print(text)
    if args.out:
        (ROOT / args.out).write_text(text)


if __name__ == "__main__":
    main()
