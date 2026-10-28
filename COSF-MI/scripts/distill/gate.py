"""
Quality gate for a distillation sample run, before a full generation run is allowed to start.

Fails if, in any category:
  - identity rejects > 0 (the reply mentioned Qwen / Alibaba / "language model" / "as an AI"),
  - the near-duplicate rate (exact + near) is >= --max-dup,
  - the keep rate DROPPED by more than --max-drop points vs the reference run (a rise never fails: it's a recovery
    or an improvement; the first version of this gate also failed rises, which wrongly blocked sample4's story recovery),
  - the share of kept replies ending in a question is >= --max-question.
The reference should be the last *accepted* sample run.

Usage: python scripts/distill/gate.py data/distill_pilot/sample4/stats.json --ref data/distill_pilot/sample2/stats.json
Exit code 0 = pass, 1 = fail.
"""
import argparse
import json
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="stats.json of the sample run to check")
    p.add_argument("--ref", required=True, help="stats.json of the last accepted sample run")
    p.add_argument("--max-drop", type=float, default=10.0, help="largest allowed keep-rate drop, in percentage points")
    p.add_argument("--max-dup", type=float, default=0.10)
    p.add_argument("--max-question", type=float, default=0.35)
    args = p.parse_args()
    run, ref = json.load(open(args.run)), json.load(open(args.ref))

    fails = []
    if run["identity_reject_rate_overall"] > 0:
        fails.append(f"identity rejects overall {run['identity_reject_rate_overall']:.1%}")
    print(f"{'category':8} {'keep ref':>9} {'keep run':>9} {'change':>7} {'dup':>6} {'ident':>6} {'ends ?':>7}")
    for c in sorted(run["per_category"]):
        b = run["per_category"][c]
        a = ref["per_category"].get(c)
        change = (b["keep_rate"] - a["keep_rate"]) * 100 if a else 0.0
        print(f"{c:8} {a['keep_rate'] if a else float('nan'):9.1%} {b['keep_rate']:9.1%} {change:+7.1f} "
              f"{b['dup_rate']:6.1%} {b['identity_reject_rate']:6.1%} {b.get('question_ending_rate', 0):7.1%}")
        if b["identity_reject_rate"] > 0:
            fails.append(f"{c}: identity rejects {b['identity_reject_rate']:.1%}")
        if b["dup_rate"] >= args.max_dup:
            fails.append(f"{c}: near-duplicate rate {b['dup_rate']:.1%} >= {args.max_dup:.0%}")
        if change < -args.max_drop:
            fails.append(f"{c}: keep rate dropped {change:.1f} points ({a['keep_rate']:.1%} -> {b['keep_rate']:.1%})")
        if b.get("question_ending_rate", 0) >= args.max_question:
            fails.append(f"{c}: {b['question_ending_rate']:.1%} of replies end in a question (>= {args.max_question:.0%})")
    print("\nGATE:", "PASS" if not fails else "FAIL")
    for f in fails:
        print("  -", f)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
