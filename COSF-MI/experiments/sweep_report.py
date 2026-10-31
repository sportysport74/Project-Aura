"""
Scaling-sweep report: experiments/results/sweep/results.tsv (written by scripts/sweep_queue.sh) -> summary.md.

Per size: val loss per arm (mean ± sd over seeds), and for each pair of arms the gap of means, Welch t, exact Mann-Whitney,
and the paired-by-seed t test (same seed = same init and the same data order, so the arms are paired). With 3 seeds per cell,
effects below ~2-3× the seed sd won't resolve; the report says so instead of calling them.

Usage (from COSF-MI/): .venv/bin/python experiments/sweep_report.py
"""
import csv
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "experiments/results/sweep"
PARAMS = {"s25": 23.5e6, "m50": 50.2e6, "l100": 103.7e6}
ARMS = ("on", "off", "control", "vanilla")
PAIRS = (("off", "on"), ("control", "on"), ("off", "control"), ("vanilla", "on"), ("vanilla", "off"))


def main():
    rows = list(csv.DictReader(open(RES / "results.tsv"), delimiter="\t"))
    val = defaultdict(dict)
    speed = defaultdict(list)
    for r in rows:
        if r["val_loss"]:
            val[(r["size"], r["arm"])][int(r["seed"])] = float(r["val_loss"])
            if r["median_tok_s"]:
                speed[(r["size"], r["arm"])].append(float(r["median_tok_s"]))
    L = ["# Scaling sweep: heartbeat on / off / control\n",
         "200M tokens per run, seq 512, batch 32, lr 6e-4 cosine to 0; val = the whole corpus val split. "
         "Gap = first arm minus second (positive = the second arm is better).\n"]
    L.append("| size | params | arm | n | val mean ± sd | per seed | median tok/s |\n|---|---|---|---|---|---|---|")
    for size in PARAMS:
        for arm in ARMS:
            v = val.get((size, arm))
            if not v:
                continue
            xs = [v[s] for s in sorted(v)]
            sd = st.stdev(xs) if len(xs) > 1 else float("nan")
            sp = st.median(speed[(size, arm)]) if speed[(size, arm)] else float("nan")
            L.append(f"| {size} | {PARAMS[size] / 1e6:.1f}M | {arm} | {len(xs)} | {st.mean(xs):.4f} ± {sd:.4f} | "
                     + ", ".join(f"{x:.4f}" for x in xs) + f" | {sp:,.0f} |")
    L.append("\n| size | pair | gap | Welch t, p | Mann-Whitney U, exact p | paired t, p (n) |\n|---|---|---|---|---|---|")
    for size in PARAMS:
        for a, b in PAIRS:
            va, vb = val.get((size, a), {}), val.get((size, b), {})
            if len(va) < 2 or len(vb) < 2:
                continue
            xa, xb = list(va.values()), list(vb.values())
            w = stats.ttest_ind(xa, xb, equal_var=False)
            mw = stats.mannwhitneyu(xa, xb, method="exact")
            common = sorted(set(va) & set(vb))
            paired = stats.ttest_rel([va[s] for s in common], [vb[s] for s in common]) if len(common) > 1 else None
            ps = f"{paired.statistic:.2f}, {paired.pvalue:.3f} ({len(common)})" if paired is not None and \
                not math.isnan(paired.statistic) else "-"
            L.append(f"| {size} | {a} − {b} | {st.mean(xa) - st.mean(xb):+.4f} | {w.statistic:.2f}, {w.pvalue:.3f} | "
                     f"{mw.statistic:.0f}, {mw.pvalue:.3f} | {ps} |")
    L.append("\nGain of the heartbeat (off − on) against size:\n\n| size | log10 params | off − on |\n|---|---|---|")
    for size in PARAMS:
        on, off = val.get((size, "on")), val.get((size, "off"))
        if on and off:
            L.append(f"| {size} | {math.log10(PARAMS[size]):.2f} | {st.mean(off.values()) - st.mean(on.values()):+.4f} |")
    (RES / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
