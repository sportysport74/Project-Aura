"""
Fine-tune set from the full distill run (data/distill_pilot/full/kept.jsonl), in this order:

1. Simile cap: per category, the share of replies with a simile opener ("It's like", "Imagine",
   "Think of it", ...) is cut to --simile-cap by dropping simile replies at random.
2. Quip thinning (arith, gsm8k): every recurring playful line (a number-free sentence, names and
   objects normalised) and every metaphor family in METAPHORS is capped at --quip-cap of the
   category's replies, dropping replies that contain it at random. A family only counts when its
   words aren't in the prompt ("a pile of stickers" for a stickers question is not a metaphor).
3. Eval problems out: arith samples whose (op, a, b) is used by an eval (experiments/arith_battery.py) are dropped.
4. Val split: --val-per-category held out per category (fewer for the small gsm8k/oasst2).
5. Arith cut: arith train is subsampled to --arith-share of train tokens, stratified by operation so the
   op mix of the full run is kept (thinning removes more division: its replies carry the candy/pizza quips).

Writes <out>/train.jsonl, <out>/val.jsonl, <out>/report.json and <out>/report.md.

Usage (from COSF-MI/):
  .venv/bin/python -I scripts/distill/build_finetune_set.py --out data/distill_pilot/ft_v1
"""
import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "experiments"))
from common import NAMES, THINGS  # noqa: E402
from arith_battery import excluded_triples  # noqa: E402

SIMILE = re.compile(
    r"(?:^|[.!?:\n]\s*|—\s*)(?:well, |okay, |oh, |so, |sure thing! )?"
    r"(it'?s (?:kind of |sort of |a bit |a little |just )?like|it is like|(?:let'?s )?imagine|(?:let'?s )?pretend"
    r"|picture (?:this|a|an|it)|think of (?:it|this|that|them|him|her|my|your|a|an|the)\b|just like|like a|like an)\b",
    re.I)

# Recurring metaphors and quips in arith/gsm8k playful lines (from the measurement in report.md).
# Generic adjectives (big, tiny, giant) and "imagine" (covered by the simile cap) are left alone.
METAPHORS = {
    "sure thing": r"\bsure thing\b",
    "that's a lot of": r"\bthat'?s (?:a whole |quite a |such a |a )lot\b",
    "pile": r"\bpiles?\b",
    "collection grew": r"\bcollection (?:grew|grows|is growing|just grew|has grown|got)\b",
    "mountain": r"\bmountains?\b",
    "candies": r"\bcand(?:y|ies)\b",
    "tower / stacking": r"\b(?:towers?|stack(?:s|ing|ed)?)\b",
    "jar": r"\bjars?\b",
    "stash": r"\bstash\b",
    "snowball / hill": r"\b(?:snowball|hill)\b",
    "library": r"\blibrar(?:y|ies)\b",
    "party": r"\bpart(?:y|ies)\b",
    "pizza / slices": r"\b(?:pizzas?|slices?|pies?)\b",
    "treasure": r"\btreasures?\b",
    "stars / sky": r"\b(?:stars?|sky)\b",
    "garden": r"\bgardens?\b",
    "felt lighter": r"\blighter\b",
    "fair share": r"\bfair share\b",
    "puzzle": r"\bpuzzles?\b",
}
METAPHORS = {k: re.compile(v, re.I) for k, v in METAPHORS.items()}
NAME_RE = re.compile(r"\b(?:" + "|".join(n.lower() for n in NAMES) + r")(?:'s)?\b")
THING_RE = re.compile(r"\b(?:" + "|".join(t for t in THINGS) + r")\b")
SMALL_VAL = {"gsm8k": 300, "oasst2": 200}


def playful_sentences(reply):
    """Number-free sentences that aren't the Answer line: the playful/connective lines of a maths reply."""
    for ln in reply.splitlines():
        for s in re.split(r"(?<=[.!?])\s+", ln.strip()):
            if s and not re.search(r"\d|=|^answer", s, re.I):
                yield s


def norm_line(s):
    s = s.lower().replace("’", "'")
    s = THING_RE.sub("<thing>", NAME_RE.sub("<name>", s))
    w = re.sub(r"[^a-z<>' ]+", " ", s).split()
    return " ".join(w) if len(w) >= 2 else None


def quip_features(r):
    """Normalised playful lines + metaphor families (not counted when their word is in the prompt)."""
    sents = list(playful_sentences(r["reply"].replace("’", "'")))
    feats = {"line: " + n for n in map(norm_line, sents) if n}
    text = " ".join(sents)
    for name, rx in METAPHORS.items():
        if rx.search(text) and not rx.search(r["prompt"]):
            feats.add("metaphor: " + name)
    return feats


def simile_cap(rows, cap, rng):
    sim = [r for r in rows if SIMILE.search(r["reply"])]
    plain = [r for r in rows if not SIMILE.search(r["reply"])]
    before = len(sim) / len(rows)
    if before > cap:
        keep = int(cap / (1 - cap) * len(plain))
        sim = rng.sample(sim, keep)
    out = plain + sim
    rng.shuffle(out)
    return out, before, len(sim) / len(out)


def feature_counts(rows, feats):
    c = collections.Counter()
    for r in rows:
        c.update(feats[id(r)])
    return c


def thin_quips(rows, cap, rng):
    """Drop random replies carrying the most over-represented feature until none is above `cap`."""
    feats = {id(r): quip_features(r) for r in rows}
    counts = feature_counts(rows, feats)
    before = counts.copy()
    n0 = len(rows)
    while True:
        f, k = counts.most_common(1)[0]
        if k <= cap * len(rows):
            break
        target = int(0.98 * cap * len(rows))  # a little under, so the loop converges as n shrinks
        holders = [r for r in rows if f in feats[id(r)]]
        drop = {id(r) for r in rng.sample(holders, k - target)}
        rows = [r for r in rows if id(r) not in drop]
        counts = feature_counts(rows, feats)
    return rows, before, counts, n0


def op_of(r):
    return r["meta"].split("|")[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/distill_pilot/full/kept.jsonl")
    p.add_argument("--out", default="data/distill_pilot/ft_v1")
    p.add_argument("--simile-cap", type=float, default=0.30)
    p.add_argument("--quip-cap", type=float, default=0.01)
    p.add_argument("--arith-share", type=float, default=0.25)
    p.add_argument("--val-per-category", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = random.Random(args.seed)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    cats = collections.defaultdict(list)
    for line in open(ROOT / args.src, encoding="utf-8"):
        r = json.loads(line)
        cats[r["category"]].append(r)
    tokens = lambda rs: sum(r["aura_tokens"] for r in rs)  # noqa: E731
    rep = {"src": args.src, "args": vars(args), "teacher": "Qwen/Qwen2.5-7B-Instruct",
           "teacher_license": "Apache-2.0 (Copyright 2024 Alibaba Cloud)", "steps": {}}
    sizes = {c: {"start": [len(rs), tokens(rs)]} for c, rs in cats.items()}
    arith_op_tokens = collections.Counter()
    for r in cats["arith"]:
        arith_op_tokens[op_of(r)] += r["aura_tokens"]

    # 1. simile cap
    rep["steps"]["simile"] = {}
    for c in cats:
        cats[c], before, after = simile_cap(cats[c], args.simile_cap, rng)
        rep["steps"]["simile"][c] = {"before": before, "after": after}
        sizes[c]["simile_cap"] = [len(cats[c]), tokens(cats[c])]

    # 2. quip thinning
    rep["steps"]["quips"] = {}
    for c in ("arith", "gsm8k"):
        rows, before, after, n0 = thin_quips(cats[c], args.quip_cap, rng)
        cats[c] = rows
        lines = lambda cnt: [(f[6:], k) for f, k in cnt.most_common() if f.startswith("line: ")][:20]  # noqa: E731
        mets = lambda cnt: [(f[10:], k) for f, k in cnt.most_common() if f.startswith("metaphor: ")]  # noqa: E731
        rep["steps"]["quips"][c] = {
            "n_before": n0, "n_after": len(rows),
            "top20_lines_before": [(f, k / n0) for f, k in lines(before)],
            "top20_lines_cover_before": sum(k for _, k in lines(before)) / n0,
            "top20_lines_after": [(f, k / len(rows)) for f, k in lines(after)],
            "metaphors_before": [(f, k / n0) for f, k in mets(before)],
            "metaphors_after": [(f, after["metaphor: " + f] / len(rows)) for f, _ in mets(before)],
            "max_feature_after": max(after.values()) / len(rows),
        }
        sizes[c]["quip_thin"] = [len(rows), tokens(rows)]

    # 3. eval problems out of arith
    ex = excluded_triples()
    keep = []
    for r in cats["arith"]:
        op, a, b = r["meta"].split("|")
        if (op, int(a), int(b)) not in ex:
            keep.append(r)
    rep["steps"]["eval_excluded"] = len(cats["arith"]) - len(keep)
    cats["arith"] = keep

    # 4. val split
    train, val = {}, {}
    for c, rs in cats.items():
        rng.shuffle(rs)
        k = SMALL_VAL.get(c, args.val_per_category)
        val[c], train[c] = rs[:k], rs[k:]

    # 5. arith cut, stratified by op (shares of the full run's arith tokens)
    other = sum(tokens(rs) for c, rs in train.items() if c != "arith")
    want = args.arith_share / (1 - args.arith_share) * other
    total_op = sum(arith_op_tokens.values())
    by_op = collections.defaultdict(list)
    for r in train["arith"]:
        by_op[op_of(r)].append(r)
    cut = []
    have = tokens(train["arith"])
    rep["steps"]["arith_cut"] = {"target_tokens": want, "available_tokens": have, "cut": have > want, "by_op": {}}
    for op, rs in sorted(by_op.items()):
        # below target after thinning: keep all of it rather than cut further
        target = want * arith_op_tokens[op] / total_op if have > want else float("inf")
        rng.shuffle(rs)
        got, t = [], 0
        for r in rs:
            if t >= target:
                break
            got.append(r)
            t += r["aura_tokens"]
        rep["steps"]["arith_cut"]["by_op"][op] = {"available": len(rs), "kept": len(got), "tokens": t,
                                                  "share_full_run": arith_op_tokens[op] / total_op,
                                                  "short": t < 0.98 * target if have > want else None}
        cut += got
    train["arith"] = cut

    for c in cats:
        sizes[c]["train"] = [len(train[c]), tokens(train[c])]
        sizes[c]["val"] = [len(val[c]), tokens(val[c])]
    rep["sizes"] = sizes
    tot = sum(tokens(rs) for rs in train.values())
    rep["train_token_share"] = {c: tokens(rs) / tot for c, rs in train.items()}
    rep["simile_rate_train"] = {c: sum(bool(SIMILE.search(r["reply"])) for r in rs) / max(1, len(rs))
                                for c, rs in train.items()}

    all_train = [r for rs in train.values() for r in rs]
    rng.shuffle(all_train)
    keys = ("category", "source", "prompt_license", "teacher", "teacher_license", "aura_tokens", "meta", "prompt", "reply", "line")
    for name, rows in (("train", all_train), ("val", [r for rs in val.values() for r in rs])):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({k: r.get(k) for k in keys}, ensure_ascii=False) + "\n")
    (out / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    write_md(out / "report.md", rep)
    print((out / "report.md").read_text())


def write_md(path, rep):
    s, L = rep["sizes"], []
    L.append("# Fine-tune set build report\n")
    L.append(f"Source `{rep['src']}`, args `{rep['args']}`.\n")
    L.append("## Sizes (samples / AURA tokens)\n")
    L.append("| category | start | simile cap | quip thin | train | val | train token share |")
    L.append("|---|---|---|---|---|---|---|")
    f = lambda v: f"{v[0]:,} / {v[1] / 1e6:.2f}M" if v else "-"  # noqa: E731
    for c in s:
        L.append(f"| {c} | {f(s[c]['start'])} | {f(s[c]['simile_cap'])} | {f(s[c].get('quip_thin'))} | "
                 f"{f(s[c]['train'])} | {f(s[c]['val'])} | {100 * rep['train_token_share'][c]:.1f}% |")
    tot = lambda k: [sum(s[c][k][0] for c in s), sum(s[c][k][1] for c in s)]  # noqa: E731
    L.append(f"| **total** | {f(tot('start'))} | {f(tot('simile_cap'))} | | {f(tot('train'))} | {f(tot('val'))} | |\n")
    L.append(f"Arith samples dropped because an eval uses their (op, a, b): {rep['steps']['eval_excluded']}.\n")
    L.append("## Simile openers (share of replies)\n")
    L.append("| category | before | after cap | in train |")
    L.append("|---|---|---|---|")
    for c, v in rep["steps"]["simile"].items():
        L.append(f"| {c} | {100 * v['before']:.1f}% | {100 * v['after']:.1f}% | {100 * rep['simile_rate_train'][c]:.1f}% |")
    for c, q in rep["steps"]["quips"].items():
        L.append(f"\n## Playful lines, {c} ({q['n_before']:,} → {q['n_after']:,} replies; "
                 f"max single feature after: {100 * q['max_feature_after']:.2f}%)\n")
        L.append(f"Top 20 lines before cover {100 * q['top20_lines_cover_before']:.1f}% of replies (sum of shares).\n")
        L.append("| # | line before | share | line after | share |")
        L.append("|---|---|---|---|---|")
        for i in range(20):
            b = q["top20_lines_before"][i] if i < len(q["top20_lines_before"]) else ("", 0)
            a = q["top20_lines_after"][i] if i < len(q["top20_lines_after"]) else ("", 0)
            L.append(f"| {i + 1} | {b[0]} | {100 * b[1]:.2f}% | {a[0]} | {100 * a[1]:.2f}% |")
        L.append("\n| metaphor family | before | after |")
        L.append("|---|---|---|")
        for (m, b), (_, a) in zip(q["metaphors_before"], q["metaphors_after"]):
            L.append(f"| {m} | {100 * b:.2f}% | {100 * a:.2f}% |")
    L.append("\n## Arith cut by operation\n")
    ac = rep["steps"]["arith_cut"]
    L.append(f"Target {ac['target_tokens'] / 1e6:.2f}M tokens, available after thinning {ac['available_tokens'] / 1e6:.2f}M: "
             + ("cut, stratified by op." if ac["cut"] else "**below target, nothing cut** (all arith train kept).") + "\n")
    L.append("| op | available | kept | tokens | full-run share | short of target |")
    L.append("|---|---|---|---|---|---|")
    for op, v in rep["steps"]["arith_cut"]["by_op"].items():
        L.append(f"| {op} | {v['available']:,} | {v['kept']:,} | {v['tokens'] / 1e6:.2f}M | "
                 f"{100 * v['share_full_run']:.1f}% | {'-' if v['short'] is None else 'YES' if v['short'] else 'no'} |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
