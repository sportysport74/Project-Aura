"""Print a table of A/B results. Usage: python experiments/summarize.py [results_dir]"""
import json
import sys
from pathlib import Path

results = {}
for f in sorted(Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "results").glob("*.json")):
    r = json.loads(f.read_text())
    results[r["name"]] = r

base = [r["final"]["val_loss"] for n, r in results.items() if n.startswith("baseline")]
ref = sum(base) / len(base) if base else None
if len(base) > 1:
    print(f"baseline seeds: {', '.join(f'{v:.4f}' for v in base)}  (noise ~ {max(base) - min(base):.4f})")

print(f"{'arm':<16} {'val loss':>9} {'vs base':>8} {'live R':>7} {'static R':>9} {'|chir|':>7} {'flipped':>8}")
for name, r in sorted(results.items(), key=lambda kv: kv[1]["final"]["val_loss"]):
    f = r["final"]
    delta = f"{f['val_loss'] - ref:+.4f}" if ref is not None else ""
    print(f"{name:<16} {f['val_loss']:>9.4f} {delta:>8} {f['live_R']:>7.3f} "
          f"{f['static_R']:>9.3f} {f['chir_abs_mean']:>7.3f} {f['chir_flipped']:>8}")

for name, r in results.items():
    if r["sleeps"]:
        jumps = [s["val_after"] - s["val_before"] for s in r["sleeps"]]
        print(f"{name}: val loss change at each sleep: " + ", ".join(f"{j:+.4f}" for j in jumps))
