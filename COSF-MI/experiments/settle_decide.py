"""
Stage-2 selection for the settling experiment (experiments/plans/settling_vanilla2.md, rule fixed before running): after seed 1, an arm
gets seeds 2 and 3 if it beats its comparison by more than the seed sd (0.005); both arms of a qualifying comparison get them (the paired
test needs both), except vanilla2 (arm a), which already has 3 seeds. Prints the space-separated arm names for stage 2 (may be empty) and
writes the reasoning to experiments/results/settling/stage2_selection.md.

Usage: python experiments/settle_decide.py [results.tsv]
"""
import csv
import sys
from pathlib import Path

SD = 0.005
COMPARISONS = [("settle_r4", "vanilla2"), ("settle_c", "settle_r4"), ("settle_f", "settle_r4"), ("settle_f", "settle_c"),
               ("settle_d", "settle_e")]


def main():
    res = Path(sys.argv[1] if len(sys.argv) > 1 else "experiments/results/sweep/results.tsv")
    rows = [r for r in csv.DictReader(open(res), delimiter="\t") if r["size"] == "s25" and r["seed"] == "1" and r["val_loss"]]
    val = {r["arm"]: float(r["val_loss"]) for r in rows}
    need, lines = set(), ["# Settling experiment: stage-2 selection (seed sd 0.005)\n", "| comparison | seed-1 val | gap | qualifies |",
                          "|---|---|---|---|"]
    for a, b in COMPARISONS:
        if a not in val or b not in val:
            lines.append(f"| {a} vs {b} | missing | - | no |")
            continue
        gap = val[b] - val[a]  # positive = a better (lower loss)
        q = abs(gap) > SD
        if q:
            need |= {a, b}
        lines.append(f"| {a} vs {b} | {val[a]:.4f} vs {val[b]:.4f} | {gap:+.4f} ({a} {'better' if gap > 0 else 'worse'}) | {'YES' if q else 'no'} |")
    need.discard("vanilla2")
    out = Path("experiments/results/settling")
    out.mkdir(parents=True, exist_ok=True)
    lines.append(f"\nStage 2 arms (seeds 2, 3): {' '.join(sorted(need)) or 'none'}")
    (out / "stage2_selection.md").write_text("\n".join(lines) + "\n")
    print(" ".join(sorted(need)))


if __name__ == "__main__":
    main()
