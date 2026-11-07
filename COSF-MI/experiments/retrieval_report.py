"""
Retrieval experiment report (experiments/plans/retrieval_prereg.md): applies the pre-registered rules to the graded factuality run.

Arms (model names in replies/grades): R0 (b10_s1, no sources), R0b (b10_s0, for the pooled R0), R1c, R2c_s0/s1 (corpus), R2w_s0/s1 (wiki),
R2closed_s0/s1, R2oracle_s0/s1. Primary metric: joined encoding, sampled_rep; paired bootstrap over questions (10k draws, seed 1), RAG seeds
pooled. Also: false-claim rate, fully-correct rate, "(Wikipedia)" tag rate (from replies.jsonl), and the share of answers copying >= 8
consecutive words of one of their sources.

Usage (from COSF-MI/): .venv/bin/python experiments/retrieval_report.py experiments/results/retrieval/factuality [--recall-dir DIR]
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCORE = {"CORRECT": 1.0, "PARTIAL": 0.5}
ARMS = {"R0": ["R0"], "R0 pooled (b10 s0+s1)": ["R0", "R0b"], "R1c": ["R1c"], "R2c": ["R2c_s0", "R2c_s1"], "R2w": ["R2w_s0", "R2w_s1"],
        "R2-closed": ["R2closed_s0", "R2closed_s1"], "R2-oracle": ["R2oracle_s0", "R2oracle_s1"]}
WORDS = re.compile(r"[a-z0-9']+")


def per_q(rows, models, enc, dec, field):
    by = defaultdict(list)
    for r in rows:
        if r["model"] in models and r["encoding"] == enc and r["decoding"] == dec:
            by[r["id"]].append(field(r))
    return {q: sum(v) / len(v) for q, v in by.items()}


def paired(a, b, draws=10_000):
    qs = sorted(set(a) & set(b))
    d = [b[q] - a[q] for q in qs]
    rng = random.Random(1)
    boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(draws))
    return sum(d) / len(d), boots[int(0.025 * draws)], boots[int(0.975 * draws) - 1], len(qs)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir")
    p.add_argument("--recall-dir", default=None)
    args = p.parse_args()
    d = ROOT / args.dir
    grades = [json.loads(line) for line in open(d / "grades.jsonl")]
    replies = [json.loads(line) for line in open(d / "replies.jsonl")]
    score = lambda r: SCORE.get(r["verdict"], 0.0)  # noqa: E731
    false = lambda r: float(bool(r["false_claim"]))  # noqa: E731
    correct = lambda r: float(r["verdict"] == "CORRECT")  # noqa: E731
    L = ["# Retrieval experiment: factuality (pre-registered rules applied)\n"]
    if args.recall_dir:
        for f in sorted((ROOT / args.recall_dir).glob("*.summary.json")):
            s = json.loads(f.read_text())
            L.append(f"- recall `{s['file']}`: @1 {s['recall@1']:.2f}, @3 **{s['recall@3']:.2f}**")
        L.append("")
    for enc in ("joined", "pieces"):
        L += [f"\n## {enc} encoding, sampled_rep (and greedy score)\n",
              "| arm | score | greedy score | fully correct | false claims | (Wikipedia) tag | copies ≥ 8 words of a source |",
              "|---|---|---|---|---|---|---|"]
        for arm, models in ARMS.items():
            s = per_q(grades, models, enc, "sampled_rep", score)
            if not s:
                continue
            g = per_q(grades, models, enc, "greedy", score)
            fc = per_q(grades, models, enc, "sampled_rep", false)
            cc = per_q(grades, models, enc, "sampled_rep", correct)
            rr = [r for r in replies if r["model"] in models and r["encoding"] == enc and r["decoding"] == "sampled_rep"]
            tag = sum(bool(re.search(r"\(Wikipedia(: [^)]*)?\)$", r.get("reply_shown", r["reply"]))) for r in rr) / max(1, len(rr))
            m = lambda x: sum(x.values()) / max(1, len(x))  # noqa: E731
            cp = sum(bool(r.get("copies8")) for r in rr) / max(1, len(rr))
            L.append(f"| {arm} | **{m(s):.3f}** | {m(g):.3f} | {m(cc):.1%} | {m(fc):.1%} | {tag:.1%} | {cp:.1%} |")
    L.append("\n## Paired comparisons (joined, sampled_rep; B − A, 95% CI)\n\n| A | B | B − A | CI | n |\n|---|---|---|---|---|")
    enc, dec = "joined", "sampled_rep"
    res = {}
    for a, b in [("R0", "R2c"), ("R2c", "R2w"), ("R0", "R2-closed"), ("R0", "R1c"), ("R0", "R2-oracle"), ("R0 pooled (b10 s0+s1)", "R2c")]:
        A, B = per_q(grades, ARMS[a], enc, dec, score), per_q(grades, ARMS[b], enc, dec, score)
        if A and B:
            res[(a, b)] = paired(A, B)
            g, lo, hi, n = res[(a, b)]
            L.append(f"| {a} | {b} | {g:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {n} |")
    fcr = {a: per_q(grades, ARMS[a], enc, dec, false) for a in ("R0", "R2c")}
    fcm = {a: sum(v.values()) / max(1, len(v)) for a, v in fcr.items()}
    L.append("\n## Verdicts (pre-registered)\n")
    if ("R0", "R2c") in res and ("R0", "R2-closed") in res:
        g, lo, _, _ = res[("R0", "R2c")]
        closed = res[("R0", "R2-closed")][0]
        works = lo > 0 and fcm["R2c"] <= fcm["R0"] + 0.02 and closed >= -0.02
        L.append(f"1. **Retrieval works: {'YES' if works else 'NO'}.** R2c − R0 {g:+.4f} (CI lower {lo:+.4f}; needs > 0); false claims R2c "
                 f"{fcm['R2c']:.1%} vs R0 {fcm['R0']:.1%} (needs ≤ R0 + 2 pts); R2-closed − R0 {closed:+.4f} (needs ≥ −0.02).")
    if ("R2c", "R2w") in res:
        g, lo, _, _ = res[("R2c", "R2w")]
        L.append(f"2. **Wikipedia store as the default chat store: {'YES' if lo > 0 else 'NO (corpus store stays the default)'}.** "
                 f"R2w − R2c {g:+.4f} (CI lower {lo:+.4f}; needs > 0).")
    (d / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
