"""
Aura v3 base, steps 1-2 report (rules pre-registered in experiments/plans/aura_v3_base.md). Reads <ck>/<run>/train.log for every run, writes
<res>/results.tsv and <res>/report.md.

  --verdict scaling  prints PASS if x330's val loss is lower than l100's on both seeds (1 and 2), else FAIL (or INCOMPLETE)
  --verdict lr_edge  prints low / high if the best x330 lr (seed 1: 2e-4, 3e-4, 4.5e-4, 6e-4) is at an edge of that grid, else none

Usage (from COSF-MI/): .venv/bin/python experiments/v3_checks_report.py [--ck checkpoints/v3_checks] [--res experiments/results/v3_checks]
"""
import argparse
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GRID = ["2e-4", "4.5e-4", "6e-4"]


def parse(log):
    t = log.read_text(errors="replace") if log.exists() else ""
    val = re.findall(r"val_loss=([0-9.]+)", t)
    tps = [int(x) for x in re.findall(r"tok/s=(\d+)", t)][3:]  # skip the compile/warm-up windows
    peak = re.findall(r"peak VRAM ([0-9.]+)", t)
    return (float(val[-1]) if val else None, statistics.median(tps) if tps else None, float(peak[-1]) if peak else None)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ck", default="checkpoints/v3_checks")
    p.add_argument("--res", default="experiments/results/v3_checks")
    p.add_argument("--verdict", choices=["scaling", "lr_edge"])
    args = p.parse_args()
    ck, res = ROOT / args.ck, ROOT / args.res
    runs = {d.name: parse(d / "train.log") for d in sorted(ck.iterdir()) if d.is_dir()} if ck.exists() else {}
    v = {k: r[0] for k, r in runs.items()}
    scale = {(m, s): v.get(f"scale_{m}_s{s}") for m in ("l100", "x330") for s in (1, 2)}
    if None in scale.values():
        scaling = "INCOMPLETE"
    else:
        scaling = "PASS" if all(scale[("x330", s)] < scale[("l100", s)] for s in (1, 2)) else "FAIL"
    lrs = {"3e-4": v.get("scale_x330_s1"), **{lr: v.get(f"lr_x330_{lr}") for lr in GRID}}
    for extra in ("1.3e-4", "9e-4"):
        if v.get(f"lr_x330_{extra}") is not None:
            lrs[extra] = v[f"lr_x330_{extra}"]
    done = {k: x for k, x in lrs.items() if x is not None}
    best = min(done, key=done.get) if done else None
    base_grid = ["2e-4", "3e-4", "4.5e-4", "6e-4"]
    edge = "none"
    if all(lrs.get(k) is not None for k in base_grid):
        b = min(base_grid, key=lambda k: lrs[k])
        edge = "low" if b == "2e-4" else "high" if b == "6e-4" else "none"
    if args.verdict == "scaling":
        print(scaling)
        return
    if args.verdict == "lr_edge":
        print(edge)
        return
    res.mkdir(parents=True, exist_ok=True)
    with open(res / "results.tsv", "w") as f:
        f.write("run\tval_loss\tmedian_tok_s\tpeak_gib\n")
        for k, (vl, tps, pk) in runs.items():
            f.write(f"{k}\t{vl if vl is not None else ''}\t{tps or ''}\t{pk or ''}\n")
    fmt = lambda x: "-" if x is None else f"{x:.4f}"  # noqa: E731
    L = ["# Aura v3 base: scaling check and lr check (rules pre-registered in `experiments/plans/aura_v3_base.md`)\n",
         "200M tokens per run (3,052 steps × 65,536 tokens), seq 1024, vanilla2, val loss on the whole v1 val split at 1024 tokens "
         "(not comparable with the 512-token numbers of earlier bases).\n",
         "## Step 1: scaling check (l100 lr 6e-4, x330 lr 3e-4)\n", "| seed | l100 | x330 | x330 − l100 |", "|---|---|---|---|"]
    for s in (1, 2):
        a, b = scale[("l100", s)], scale[("x330", s)]
        L.append(f"| {s} | {fmt(a)} | {fmt(b)} | {'-' if None in (a, b) else f'{b - a:+.4f}'} |")
    L.append(f"\n**Rule (x330 lower on both seeds): {scaling}.**\n")
    L += ["## Step 2: lr check at x330 (seed 1)\n", "| lr | val loss |", "|---|---|"]
    for k in sorted(lrs, key=float):
        L.append(f"| {k} | {fmt(lrs[k])}{' **(best)**' if k == best else ''} |")
    L.append(f"\nGrid edge after the 4-point grid: {edge}. **Best lr so far: {best or '-'}.**\n")
    L += ["## Throughput and memory (median tok/s after warm-up; torch peak)\n", "| run | tok/s | peak GiB |", "|---|---|---|"]
    for k, (_, tps, pk) in runs.items():
        L.append(f"| {k} | {f'{tps:,.0f}' if tps else '-'} | {pk or '-'} |")
    (res / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
