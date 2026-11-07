"""
Retrieval gate report (pre-registered rules: experiments/plans/retrieval_gate.md). CPU only. Run only after the 30-grade hand check.

Reads <res>/{signals,replies,grades,entropy}.jsonl, <res>/thresholds.json, <res>/multiturn/grades.jsonl and the re-score's graded rows
(experiments/results/retrieval/factuality_rescore/grades.jsonl), and writes <res>/report.md and <res>/chat_quality.md.

Arms: G0 = RAG + sources for everything; G1 = templates for unclear / no-sources, b10_s1 for chit-chat; G2 = b10_s1 for every gated branch.
Per message, an arm's value is the mean over its replies (RAG: seeds 0 and 1 × samples 1-3; b10_s1: samples 1-3); templates score 0
(factuality, no false claim) and are never inappropriate. Paired bootstrap over messages / questions, 10k draws, seed 1.

Usage (from COSF-MI/): .venv/bin/python experiments/gate_report.py [--res experiments/results/retrieval/gate]
"""
import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "python"))
from cosf_mi.chat_gate import TEMPLATE_NOSOURCES, TEMPLATE_UNCLEAR, route  # noqa: E402

RESCORE = ROOT / "experiments/results/retrieval/factuality_rescore/grades.jsonl"
SCORE = {"CORRECT": 1.0, "PARTIAL": 0.5}
OFF = ("chitchat", "unanswerable", "garbled")
BRANCHES = ("rag", "chitchat", "unclear", "nosources")


def mean(x):
    x = list(x)
    return sum(x) / len(x) if x else float("nan")


def paired(a, b, draws=10_000):
    """B − A over the shared keys: mean and 95% bootstrap CI."""
    ks = sorted(set(a) & set(b))
    d = [b[k] - a[k] for k in ks]
    rng = random.Random(1)
    boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(draws))
    return sum(d) / len(d), boots[int(0.025 * draws)], boots[int(0.975 * draws) - 1], len(ks)


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def rep4(text):
    w = re.findall(r"[a-z0-9']+", text.lower())
    grams = [" ".join(w[i:i + 4]) for i in range(len(w) - 3)]
    return len(grams) != len(set(grams))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--res", default="experiments/results/retrieval/gate")
    p.add_argument("--grades", default="grades_hand.jsonl", help="off-topic grades (the blind hand grades: no LLM grader passed its hand check)")
    args = p.parse_args()
    d = ROOT / args.res
    th_all = json.loads((d / "thresholds.json").read_text())
    th = th_all["thresholds"]
    sig = [json.loads(line) for line in open(d / "signals.jsonl")]
    for r in sig:
        r["branch"] = route(r, th)
    L = ["# Retrieval gate eval (pre-registered rules applied)\n", "Plan: `experiments/plans/retrieval_gate.md`. Grader hand checks: `handcheck_{v1,v2,v3,32b}.md` (all failed); rule 2 uses blind hand grades.\n",
         "## Gate 0 and thresholds (tune split only)\n", "| signal | AUROC (tune) | threshold | used |", "|---|---|---|---|"]
    names = {"g": "g garbled (b10_s1 NLL)", "c": "c chit-chat (max sim to tune examples)", "s": "s top-1 source similarity"}
    for k in ("g", "c", "s"):
        L.append(f"| {names[k]} | {th_all['auroc_tune'][k]:.3f} | {th_all['candidate_thresholds'][k]:.4f} | "
                 f"{'yes' if th[k] is not None else '**no (AUROC < 0.85)**'} |")

    # routing on the test split and the factuality questions
    L += ["\n## Routing (test split, and the 100 factuality questions)\n", "| kind / sub | n | " + " | ".join(BRANCHES) + " |",
          "|---|---|" + "---|" * len(BRANCHES)]
    groups = defaultdict(list)
    for r in sig:
        if r["split"] in ("test", "eval"):
            groups[r["kind"]].append(r)
            groups[f"{r['kind']} / {r['sub']}"].append(r)
    for g in sorted(groups, key=lambda x: (x.split(" / ")[0], " / " in x, x)):
        if g.startswith("factuality /") or g.startswith("ontopic /"):
            continue
        c = Counter(r["branch"] for r in groups[g])
        n = len(groups[g])
        L.append(f"| {g} | {n} | " + " | ".join(f"{c[b] / n:.0%}" for b in BRANCHES) + " |")
    on_test = groups["ontopic"]
    fq = groups["factuality"]

    # rule 1: factuality composition from the re-score's graded rows
    rs = [json.loads(line) for line in open(RESCORE)]
    rs = [r for r in rs if r["encoding"] == "joined" and r["decoding"] == "sampled_rep"]
    by = defaultdict(list)
    for r in rs:
        if r["model"] in ("R2c_s0", "R2c_s1"):
            by[("rag", r["id"])].append(r)
        elif r["model"] == "R0":
            by[("b10", r["id"])].append(r)
    fbranch = {r["id"]: r["branch"] for r in fq}
    sc = lambda rows: mean(SCORE.get(r["verdict"], 0.0) for r in rows)  # noqa: E731
    fc = lambda rows: mean(float(bool(r["false_claim"])) for r in rows)  # noqa: E731
    arm_f = {"G0": {}, "G1": {}, "G2": {}}
    arm_fc = {"G0": {}, "G1": {}, "G2": {}}
    for q, b in fbranch.items():
        arm_f["G0"][q], arm_fc["G0"][q] = sc(by[("rag", q)]), fc(by[("rag", q)])
        for arm in ("G1", "G2"):
            if b == "rag":
                src = by[("rag", q)]
            elif b == "chitchat" or arm == "G2":
                src = by[("b10", q)]
            else:
                arm_f[arm][q], arm_fc[arm][q] = 0.0, 0.0
                continue
            arm_f[arm][q], arm_fc[arm][q] = sc(src), fc(src)
    L += ["\n## Rule 1: factuality must not drop (100 questions, joined, sampled; composed from the re-score's graded rows)\n",
          "| arm | score | false claims | template share | − G0 | 95% CI | rule (≥ −0.02) |", "|---|---|---|---|---|---|---|"]
    res = {}
    for arm in ("G0", "G1", "G2"):
        tshare = mean(float(arm == "G1" and fbranch[q] in ("unclear", "nosources")) for q in fbranch)
        if arm == "G0":
            L.append(f"| G0 | {mean(arm_f[arm].values()):.3f} | {mean(arm_fc[arm].values()):.1%} | 0% | | | |")
            continue
        g, lo, hi, n = paired(arm_f["G0"], arm_f[arm])
        res[f"rule1_{arm}"] = g >= -0.02
        L.append(f"| {arm} | {mean(arm_f[arm].values()):.3f} | {mean(arm_fc[arm].values()):.1%} | {tshare:.0%} | {g:+.4f} | "
                 f"[{lo:+.4f}, {hi:+.4f}] | {'PASS' if g >= -0.02 else 'FAIL'} |")

    # rule 2: off-topic inappropriate rate
    grades = [json.loads(line) for line in open(d / args.grades)]
    gby = defaultdict(list)
    for r in grades:
        gby[("rag" if r["model"].startswith("RAG") else "b10", r["id"])].append(float(bool(r["inappropriate"])))
    graded = {r["id"] for r in grades}
    off = [r for r in sig if r["split"] == "test" and r["kind"] in OFF and r["id"] in graded]  # the graded messages only
    arm_i = {"G0": {}, "G1": {}, "G2": {}}
    for r in off:
        i = r["id"]
        arm_i["G0"][i] = mean(gby[("rag", i)])
        for arm in ("G1", "G2"):
            b = r["branch"]
            arm_i[arm][i] = (arm_i["G0"][i] if b == "rag" else
                             mean(gby[("b10", i)]) if (b == "chitchat" or arm == "G2") else 0.0)
    L += [f"\n## Rule 2: inappropriate answers on the off-topic test set must drop ({len(off)} messages graded: `{args.grades}`)\n",
          "| arm | all | chit-chat | unanswerable | garbled | garbled: word salad (unseen type) | chit-chat: hand short | − G0 | 95% CI | rule (CI < 0) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    kind = {r["id"]: r for r in off}
    for arm in ("G0", "G1", "G2"):
        v = arm_i[arm]
        cell = lambda f: f"{mean(x for i, x in v.items() if f(kind[i])):.1%}"  # noqa: E731
        row = (f"| {arm} | {mean(v.values()):.1%} | {cell(lambda r: r['kind'] == 'chitchat')} | {cell(lambda r: r['kind'] == 'unanswerable')} | "
               f"{cell(lambda r: r['kind'] == 'garbled')} | {cell(lambda r: r['sub'] == 'salad')} | {cell(lambda r: r['sub'] == 'hand_short')} |")
        if arm == "G0":
            L.append(row + " | | |")
            continue
        g, lo, hi, n = paired(arm_i["G0"], v)
        res[f"rule2_{arm}"] = hi < 0
        L.append(row + f" {g:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {'PASS' if hi < 0 else 'FAIL'} |")

    # rule 3: the two templates, scored separately
    L += ["\n## Rule 3: the two gate replies, scored separately (G1)\n",
          "| reply | fires on garbled | on unanswerable | on chit-chat | **mis-fires on on-topic test** | on factuality questions | "
          "G0's inappropriate rate on the messages it catches | ship (mis-fire ≤ 10%) |", "|---|---|---|---|---|---|---|---|"]
    for b, text in (("unclear", TEMPLATE_UNCLEAR), ("nosources", TEMPLATE_NOSOURCES)):
        rate = lambda rows: mean(float(r["branch"] == b) for r in rows)  # noqa: E731
        caught = [arm_i["G0"][r["id"]] for r in off if r["branch"] == b]
        mis = rate(on_test)
        res[f"rule3_{b}"] = mis <= 0.10
        L.append(f"| \"{text}\" | {rate([r for r in off if r['kind'] == 'garbled']):.0%} | "
                 f"{rate([r for r in off if r['kind'] == 'unanswerable']):.0%} | {rate([r for r in off if r['kind'] == 'chitchat']):.0%} | "
                 f"**{mis:.1%}** | {rate(fq):.0%} | {mean(caught):.1%} (n={len(caught)}) | {'yes' if mis <= 0.10 else '**no**'} |")
    L.append(f"\nChit-chat branch (b10_s1) on on-topic test questions: {mean(float(r['branch'] == 'chitchat') for r in on_test):.1%}.")

    # rule 4: chat quality, 30 messages, seed 1
    reps = [json.loads(line) for line in open(d / "replies.jsonl")]
    rep_by = {(r["model"], r["id"], r["seed"]): r for r in reps}
    rng = random.Random(0)
    test_by = defaultdict(list)
    for r in sig:
        if r["split"] == "test":
            test_by[r["kind"]].append(r)
    on_ids = {r["id"] for r in reps if r["kind"] == "ontopic"}
    pick = lambda k, n: rng.sample(test_by[k], min(n, len(test_by[k])))  # noqa: E731
    cq = [r for r in test_by["ontopic"] if r["id"] in on_ids][:10] + pick("chitchat", 10) + pick("garbled", 5) + pick("unanswerable", 5)
    cq = [r for r in cq if ("B10", r["id"], 1) in rep_by]

    def reply_for(arm, r):
        b = r["branch"]
        if arm == "G0" or b == "rag":
            return rep_by[("RAG_s0", r["id"], 1)]["reply"], rep_by[("RAG_s0", r["id"], 1)]["ended"]
        if arm == "G1" and b in ("unclear", "nosources"):
            return (TEMPLATE_UNCLEAR if b == "unclear" else TEMPLATE_NOSOURCES), True
        return rep_by[("B10", r["id"], 1)]["reply"], rep_by[("B10", r["id"], 1)]["ended"]
    L += ["\n## Rule 4: chat quality (30 test messages, 1 sampled reply; side by side in `chat_quality.md`)\n",
          "| arm | turn ended | repeated 4-gram | rule (≤ 5 pts worse than G0) |", "|---|---|---|---|"]
    q = {}
    for arm in ("G0", "G1", "G2"):
        rr = [reply_for(arm, r) for r in cq]
        q[arm] = (mean(float(e) for _, e in rr), mean(float(rep4(t)) for t, _ in rr))
        ok = arm == "G0" or (q[arm][0] >= q["G0"][0] - 0.05 and q[arm][1] <= q["G0"][1] + 0.05)
        if arm != "G0":
            res[f"rule4_{arm}"] = ok
        L.append(f"| {arm} | {q[arm][0]:.0%} | {q[arm][1]:.0%} | {'' if arm == 'G0' else 'PASS' if ok else 'FAIL'} |")
    C = ["# Chat quality: 30 test messages side by side (seed 1; G0 = RAG s0 + sources)\n"]
    for n, r in enumerate(cq, 1):
        C.append(f"## {n}. [{r['kind']}/{r['sub']}, routed: {r['branch']}] {r['text']}\n")
        for arm in ("G0", "G1", "G2"):
            C.append(f"- **{arm}:** {reply_for(arm, r)[0]}")
        C.append("")
    (d / "chat_quality.md").write_text("\n".join(C) + "\n")

    # answer entropy (diagnostic)
    ent = {(r["model"], r["seed"], r["id"]): r["entropy"] for r in map(json.loads, open(d / "entropy.jsonl"))}
    L += ["\n## Answer entropy vs false claims (diagnostic only; re-score answers, joined, sampled)\n",
          "| answers | n | mean entropy, false claim | mean entropy, no false claim | AUROC (false claim) | 95% CI (bootstrap over questions) | predicts? |",
          "|---|---|---|---|---|---|---|"]
    for label, models in (("R2c (RAG + corpus sources)", ("R2c_s0", "R2c_s1")), ("R0 (b10_s1, no sources)", ("R0",))):
        rows = [(r["id"], ent[(r["model"], r["seed"], r["id"])], bool(r["false_claim"])) for r in rs
                if r["model"] in models and (r["model"], r["seed"], r["id"]) in ent and ent[(r["model"], r["seed"], r["id"])] == ent[(r["model"], r["seed"], r["id"])]]
        pos = [e for _, e, f in rows if f]
        neg = [e for _, e, f in rows if not f]
        a = auroc(pos, neg)
        qids = sorted({i for i, _, _ in rows})
        byq = defaultdict(list)
        for i, e, f in rows:
            byq[i].append((e, f))
        rng = random.Random(1)
        boots = []
        for _ in range(1000):
            samp = [x for _ in qids for x in byq[rng.choice(qids)]]
            boots.append(auroc([e for e, f in samp if f], [e for e, f in samp if not f]))
        boots.sort()
        lo, hi = boots[25], boots[974]
        L.append(f"| {label} | {len(rows)} | {mean(pos):.3f} | {mean(neg):.3f} | {a:.3f} | [{lo:.3f}, {hi:.3f}] | "
                 f"{'yes' if lo > 0.5 or hi < 0.5 else 'no'} |")

    # multi-turn step A
    mt = [json.loads(line) for line in open(d / "multiturn/grades.jsonl")]
    mt = [r for r in mt if not r["model"].startswith("control_")]
    arm_m = {}
    for arm in ("M0", "M1"):
        byq = defaultdict(list)
        for r in mt:
            if r["model"].startswith(arm):
                byq[r["id"]].append(SCORE.get(r["verdict"], 0.0))
        arm_m[arm] = {i: mean(v) for i, v in byq.items()}
    g, lo, hi, n = paired(arm_m["M0"], arm_m["M1"])
    res["multiturn_M1"] = lo > 0
    notes = json.loads((d / "multiturn/notes.json").read_text())
    L += ["\n## Multi-turn step A (38 two-turn conversations; RAG s0 + s1 × 3 samples; factuality grader)\n",
          "| arm | score | − M0 | 95% CI | rule (CI > 0) |", "|---|---|---|---|---|",
          f"| M0 follow-up alone | {mean(arm_m['M0'].values()):.3f} | | | |",
          f"| M1 history + query of both turns, k=2 at 60 words | {mean(arm_m['M1'].values()):.3f} | {g:+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
          f"{'PASS: adopt' if lo > 0 else 'FAIL: not adopted'} |",
          f"\nM1 contexts longer than the window (cut from the left): {notes['m1_contexts_cut_at_the_window']} of {len(arm_m['M1']) * 6}."]

    # decision
    ship = {arm: res.get(f"rule1_{arm}") and res.get(f"rule2_{arm}") and res.get(f"rule4_{arm}") for arm in ("G1", "G2")}
    L += ["\n## Decision (pre-registered)\n"]
    for arm in ("G1", "G2"):
        L.append(f"- **{arm}: {'passes' if ship[arm] else 'fails'}** rules 1 / 2 / 4: {res.get(f'rule1_{arm}')} / {res.get(f'rule2_{arm}')} / "
                 f"{res.get(f'rule4_{arm}')}.")
    L.append(f"- Templates: unclear {'shippable' if res['rule3_unclear'] else 'NOT shippable'} (mis-fire), no-sources "
             f"{'shippable' if res['rule3_nosources'] else 'NOT shippable'}.")
    passing = [a for a in ("G1", "G2") if ship[a]]
    if passing:
        best = min(passing, key=lambda a: mean(arm_i[a].values()))
        L.append(f"- **Gate default in store mode: {best}** (lowest inappropriate rate among the passing arms).")
    else:
        L.append("- **No gate becomes the default**; it stays behind `--gate`.")
    L.append(f"- Multi-turn history (M1): {'adopted for chat' if res['multiturn_M1'] else 'not adopted (step B, a fine-tune, needs its own plan)'}.")
    (d / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
