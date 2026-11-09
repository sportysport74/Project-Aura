"""Loss-spike statistics from existing train.log files, to set the v3 spike guard (experiments/plans/aura_v3_base.md, spike guard).

The trainer shows the single-step task loss in the progress bar every 10 steps (`N/TOTAL [... loss=X`), so a log gives one reading per
10 steps. The guard is defined on exactly these readings:
  signal(s)   = mean of the last W readings (W = 10, i.e. the last 100 steps)
  baseline(s) = mean of the B readings before those (B = 200, i.e. the 2,000 steps before)
  excess(s)   = signal − baseline
For each log this prints the distribution of excess after warm-up (from step --from), the longest run of consecutive readings with
excess > m for several margins m, and the first step where the excess crosses each margin.
Usage: spike_stats.py [--from 2000] LOG ...
"""
import argparse
import re
from pathlib import Path

import numpy as np

PAT = re.compile(r"(\d+)/\d+ \[[^\]]*loss=([0-9.]+|nan|inf)")


def readings(path):
    last = {}
    for chunk in Path(path).read_text(errors="replace").replace("\r", "\n").split("\n"):
        for n, x in PAT.findall(chunk):
            n = int(n)
            if n % 10 == 0:
                last[n] = float(x)  # the postfix is set every 10 steps; keep the last value shown at that step
    steps = np.array(sorted(last))
    return steps, np.array([last[s] for s in steps])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="start", type=int, default=2000)
    p.add_argument("--W", type=int, default=10)
    p.add_argument("--B", type=int, default=200)
    p.add_argument("--margins", default="0.05,0.1,0.15,0.2,0.3,0.5")
    p.add_argument("logs", nargs="+")
    a = p.parse_args()
    ms = [float(m) for m in a.margins.split(",")]
    for log in a.logs:
        s, x = readings(log)
        if len(x) < a.W + a.B + 1:
            print(f"{log}: only {len(x)} readings")
            continue
        c = np.concatenate([[0.0], np.cumsum(x)])
        idx = np.arange(a.W + a.B, len(x) + 1)  # window end (exclusive)
        sig = (c[idx] - c[idx - a.W]) / a.W
        base = (c[idx - a.W] - c[idx - a.W - a.B]) / a.B
        ex = sig - base
        st = s[idx - 1]
        keep = st >= a.start
        ex, st = ex[keep], st[keep]
        q = np.percentile(ex, [50, 99, 99.9, 100])
        print(f"\n{log}: {len(x):,} readings (steps {s[0]}-{s[-1]}), {len(ex):,} after step {a.start}; single-reading sd "
              f"{np.std(np.diff(x)) / np.sqrt(2):.3f}")
        print(f"  excess p50 {q[0]:+.3f}  p99 {q[1]:+.3f}  p99.9 {q[2]:+.3f}  max {q[3]:+.3f} (at step {st[np.argmax(ex)]})")
        for m in ms:
            over = ex > m
            run = best = 0
            for o in over:
                run = run + 1 if o else 0
                best = max(best, run)
            first = st[np.argmax(over)] if over.any() else None
            print(f"  m={m:.2f}: {over.mean():7.3%} of readings over, longest run {best * 10:6d} steps, first at step {first}")


if __name__ == "__main__":
    main()
