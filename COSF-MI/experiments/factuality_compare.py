"""
Paired comparison of two models on the factuality eval: per-question mean score (seeds pooled), difference B − A, and a
paired bootstrap over the 100 questions (resampling questions, 10,000 draws) for the 95% CI.

Used by the 10B-pretrain gate (experiments/plans/pretrain_10b.md): continue past 5B only if B − A ≥ +0.05 and the CI's
lower bound is above 0.

Usage (from COSF-MI/): .venv/bin/python experiments/factuality_compare.py DIR MODEL_A MODEL_B [--encoding joined]
  [--decoding sampled_rep] [--min-gain 0.05]
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCORE = {"CORRECT": 1.0, "PARTIAL": 0.5}


def per_question(rows, model, encoding, decoding):
    by = defaultdict(list)
    for r in rows:
        if r["model"] == model and r["encoding"] == encoding and r["decoding"] == decoding:
            by[r["id"]].append(SCORE.get(r["verdict"], 0.0))
    return {q: sum(v) / len(v) for q, v in by.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--encoding", default="joined")
    p.add_argument("--decoding", default="sampled_rep")
    p.add_argument("--min-gain", type=float, default=0.05)
    p.add_argument("--draws", type=int, default=10_000)
    args = p.parse_args()
    rows = [json.loads(line) for line in open(ROOT / args.dir / "grades.jsonl")]
    a = per_question(rows, args.a, args.encoding, args.decoding)
    b = per_question(rows, args.b, args.encoding, args.decoding)
    qs = sorted(set(a) & set(b))
    assert qs, f"no common questions for {args.a} and {args.b} ({args.encoding}, {args.decoding})"
    d = [b[q] - a[q] for q in qs]
    rng = random.Random(0)
    boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(args.draws))
    lo, hi = boots[int(0.025 * args.draws)], boots[int(0.975 * args.draws) - 1]
    gain = sum(d) / len(d)
    ok = gain >= args.min_gain and lo > 0
    res = {"a": args.a, "b": args.b, "encoding": args.encoding, "decoding": args.decoding, "questions": len(qs),
           "score_a": round(sum(a[q] for q in qs) / len(qs), 4), "score_b": round(sum(b[q] for q in qs) / len(qs), 4),
           "gain": round(gain, 4), "ci95": [round(lo, 4), round(hi, 4)], "min_gain": args.min_gain,
           "gate": "PASS" if ok else "FAIL"}
    (ROOT / args.dir / f"compare_{args.a}_vs_{args.b}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
