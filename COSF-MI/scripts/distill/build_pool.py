"""
Stage 1 of the distillation pilot: build the whole prompt pool at full scale, deduplicate it and gate it.

Requested prompts per category come from the token-share sizing (common.size_requests, calibrated from a sample run's
stats.json). Then:
  stage 0  the teacher splits each topic (common.TOPICS) into --subtopics-per-topic subtopics; exact dups dropped
  explain  the teacher writes one user message per request; requests are spread evenly over subtopics, each with
  chat     its own random asker, style and angle, and its own vLLM seed. Messages are deduplicated (exact, then
           near-duplicate via MinHash-LSH on word 5-grams, Jaccard >= 0.7) and capped per subtopic at
           ceil(--cap-factor × requested / subtopics) so no subtopic dominates.
  arith    distinct (op, a, b) problems from wide ranges, sampled without replacement, held-out eval problems excluded
  story    distinct (character, situation, setting, tone) combos, without replacement (180,000 possible)
  gsm8k    every GSM8K train question (MIT); oasst2: every non-technical OASST2 English root prompt (Apache-2.0)
Nothing is ever repeated to fill a quota.

Gate: in every category at least --min-survival of the requested prompts must survive as unique. The report lists the
20 most common subtopics (and topics) with their share. Output: data/distill_pilot/pool/<category>.jsonl and
pool_stats.json. Exit code 0 = gate passed, 1 = failed.

Usage (vLLM venv):
  VLLM_USE_FLASHINFER_SAMPLER=0 ~/venvs/vllm/bin/python -I scripts/distill/build_pool.py \
      --calibrate-from data/distill_pilot/sample4/stats.json --target-tokens 50e6
"""
import argparse
import json
import math
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calibrate-from", required=True, help="a sample run's stats.json (tokens/sample, keep rate)")
    p.add_argument("--target-tokens", type=float, default=50e6)
    p.add_argument("--out", default=str(C.POOL_DIR))
    p.add_argument("--subtopics-per-topic", type=int, default=30)
    p.add_argument("--cap-factor", type=float, default=1.5, help="per-subtopic cap = factor × requested / subtopics")
    p.add_argument("--min-survival", type=float, default=0.80)
    p.add_argument("--chunk", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from vllm import LLM, SamplingParams  # noqa: E402  (vLLM venv only)

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    gsm = [json.loads(line) for line in open(C.ROOT / "data/distill_pilot/src_gsm8k.jsonl")]
    oas = [json.loads(line) for line in open(C.ROOT / "data/distill_pilot/src_oasst2.jsonl")]
    oas = [o for o in oas if not C.TECH_PROMPT.search(o["prompt"])]
    cal = json.load(open(args.calibrate_from))
    requested, per_tok, keep = C.size_requests(args.target_tokens, cal, {"gsm8k": len(gsm), "oasst2": len(oas)})
    print(f"Requested prompts: {requested} (total {sum(requested.values()):,}); expected "
          f"{C.expected_tokens(requested, per_tok, keep) / 1e6:.1f}M AURA tokens if all survive", flush=True)

    llm = LLM(model=C.TEACHER, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.85, seed=args.seed)

    # Stage 0: subtopics
    topic_convs = [[{"role": "system", "content": C.SUBTOPIC_SYSTEM},
                    {"role": "user", "content": C.SUBTOPIC_USER.format(n=args.subtopics_per_topic, topic=t)}]
                   for t in C.TOPICS]
    sps = [SamplingParams(temperature=0.7, top_p=0.95, max_tokens=500, seed=s)
           for s in C.per_request_seeds(args.seed + 1, len(topic_convs))]
    subtopics, seen = [], set()
    for topic, o in zip(C.TOPICS, llm.chat(topic_convs, sps, use_tqdm=False)):
        for line in o.outputs[0].text.splitlines():
            s = C.clean_subtopic(line)
            key = " ".join(C.NearDup.norm(s)) if s else None
            if s and key not in seen:
                seen.add(key)
                subtopics.append((topic, s))
    print(f"stage 0: {len(subtopics):,} unique subtopics from {len(C.TOPICS)} topics "
          f"({time.time() - t0:.0f} s)", flush=True)
    (out / "subtopics.json").write_text(json.dumps(subtopics, indent=0))

    # Stage 1: teacher-written explain and chat prompts
    pools, stage1 = {}, {}
    nd = C.NearDup(seed=args.seed)
    for k_i, kind in enumerate(("explain", "chat")):
        n_req = requested[kind]
        cap = math.ceil(args.cap_factor * n_req / len(subtopics))
        order = list(range(len(subtopics)))
        rng.shuffle(order)
        assign = [order[i % len(order)] for i in range(n_req)]  # even spread: each subtopic n/S or n/S + 1 times
        rng.shuffle(assign)
        seeds = C.per_request_seeds(args.seed + 100 + k_i, n_req)
        counts, per_sub, pool = Counter(), Counter(), []
        for i in range(0, n_req, args.chunk):
            idx = assign[i:i + args.chunk]
            convs = [C.writer_request(rng, kind, *subtopics[j]) for j in idx]
            sps = [SamplingParams(temperature=1.0, top_p=0.95, max_tokens=60, seed=s) for s in seeds[i:i + args.chunk]]
            for j, o in zip(idx, llm.chat(convs, sps, use_tqdm=False)):
                t = C.clean_written_prompt(o.outputs[0].text)
                if not t:
                    counts["junk"] += 1
                    continue
                dup = nd.check_and_add(kind, t)
                if dup:
                    counts[dup] += 1
                    continue
                if per_sub[j] >= cap:
                    counts["over_cap"] += 1
                    continue
                per_sub[j] += 1
                topic, sub = subtopics[j]
                pool.append({"category": kind, "source": "qwen-written", "license": "Apache-2.0 (Qwen output)",
                             "prompt": t, "meta": f"{topic}|{sub}"})
            print(f"stage 1 {kind}: {min(i + args.chunk, n_req):,}/{n_req:,} written, {len(pool):,} unique kept "
                  f"({time.time() - t0:.0f} s); " + ", ".join(f"{k} {v:,}" for k, v in counts.items()), flush=True)
        pools[kind] = pool
        top_sub = per_sub.most_common(20)
        top_topic = Counter(subtopics[j][0] for j, n in per_sub.items() for _ in range(n)).most_common(20)
        stage1[kind] = {"cap_per_subtopic": cap, "rejects": dict(counts), "subtopics_used": len(per_sub),
                        "top20_subtopics": [[" / ".join(subtopics[j]), n, round(n / max(1, len(pool)), 5)]
                                            for j, n in top_sub],
                        "top20_topics": [[t, n, round(n / max(1, len(pool)), 4)] for t, n in top_topic]}

    # Programmatic and fixed pools
    pools["arith"] = C.arith_problems(requested["arith"], rng)
    pools["story"] = C.story_requests(requested["story"], rng)
    pools["gsm8k"] = [dict(g, category="gsm8k") for g in gsm[:requested["gsm8k"]]]
    pools["oasst2"] = [dict(o, category="oasst2") for o in oas[:requested["oasst2"]]]

    for c, pool in pools.items():
        with open(out / f"{c}.jsonl", "w") as f:
            for r in pool:
                f.write(json.dumps(r) + "\n")

    have = {c: len(pools[c]) for c in requested}
    survival = {c: have[c] / requested[c] for c in requested}
    fails = [f"{c}: {survival[c]:.1%} of requested prompts survived as unique (< {args.min_survival:.0%})"
             for c in requested if survival[c] < args.min_survival]
    stats = {
        "requested": requested, "unique": have, "survival": {c: round(v, 4) for c, v in survival.items()},
        "expected_kept_aura_tokens": round(C.expected_tokens(have, per_tok, keep)),
        "calibration": {"from": args.calibrate_from, "tokens_per_sample": per_tok, "keep_rate": keep},
        "subtopics": len(subtopics), "subtopics_per_topic_asked": args.subtopics_per_topic, "stage1": stage1,
        "gate": {"min_survival": args.min_survival, "passed": not fails, "fails": fails},
        "minutes": round((time.time() - t0) / 60, 1), "teacher": C.TEACHER, "teacher_license": C.TEACHER_LICENSE,
        "git_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=C.ROOT, capture_output=True,
                                     text=True).stdout.strip(),
    }
    (out / "pool_stats.json").write_text(json.dumps(stats, indent=1))

    print(f"\n{'category':8} {'requested':>10} {'unique':>9} {'survival':>9}")
    for c in requested:
        print(f"{c:8} {requested[c]:>10,} {have[c]:>9,} {survival[c]:>9.1%}")
    print(f"expected kept AURA tokens from this pool: {stats['expected_kept_aura_tokens'] / 1e6:.1f}M")
    for kind in ("explain", "chat"):
        print(f"\n{kind}: {stage1[kind]['subtopics_used']:,} subtopics used, cap {stage1[kind]['cap_per_subtopic']}/subtopic; "
              f"20 most common subtopics:")
        for name, n, share in stage1[kind]["top20_subtopics"]:
            print(f"  {share:7.3%}  {n:>4}  {name}")
        print("  20 most common topics: " + ", ".join(f"{t} {s:.2%}" for t, _, s in stage1[kind]["top20_topics"]))
    print("\nPOOL GATE:", "PASS" if not fails else "FAIL")
    for f in fails:
        print("  -", f)
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
