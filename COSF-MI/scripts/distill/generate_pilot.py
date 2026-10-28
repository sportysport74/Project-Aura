"""
Stage 2 of the distillation pilot: teacher replies for the prompt pool built by build_pool.py.

Teacher: Qwen/Qwen2.5-7B-Instruct, Apache-2.0 (Copyright 2024 Alibaba Cloud). Outputs may be used to train other
models; credit the teacher in run configs and any dataset or model card.

Prompts come only from data/distill_pilot/pool/<category>.jsonl and are never repeated: --sample N draws ~N of them
(category shares as in common.SHARES, without replacement); a full run uses the whole pool. Every reply is written as
"Aura" (common.SYSTEM; for chat/explain/oasst2 a per-request coin decides whether it ends with a follow-up question)
and goes through:
  truncated -> identity (Qwen / Alibaba / "language model" / "as an AI") -> format ([INST]) -> latex -> code
  -> answer (math: 'Answer: N' must be correct AND the last line; others: no 'Answer:' line)
  -> too_long (> --max-aura-tokens, counted exactly as finetune_chat.encode_chat does)
  -> no_ending (stories the teacher judges UNFINISHED) -> exact_dup / near_dup (MinHash-LSH, per category).
GSM8K gets common.GSM8K_SAMPLES replies per question; the shortest one that passes is kept. If any category's
near-duplicate rate reaches --max-dup-rate (after --dup-min-count candidates), generation stops.

Output (data/distill_pilot/<run>/): kept.jsonl (with a finetune_chat.py "line" field), rejected.jsonl, stats.json,
run_config.json, and review_20.md (20 random kept samples) for sample runs. Check sample runs with gate.py.

Usage (vLLM venv; WSL has no nvcc, so FlashInfer's sampler must be off):
  VLLM_USE_FLASHINFER_SAMPLER=0 ~/venvs/vllm/bin/python -I scripts/distill/generate_pilot.py --sample 2000 --out ...
  VLLM_USE_FLASHINFER_SAMPLER=0 ~/venvs/vllm/bin/python -I scripts/distill/generate_pilot.py --out data/distill_pilot/full
"""
import argparse
import json
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", type=int, default=0, help="draw ~N prompts from the pool (0 = the whole pool)")
    p.add_argument("--pool", default=str(C.POOL_DIR))
    p.add_argument("--out", required=True)
    p.add_argument("--chunk", type=int, default=16000, help="requests per vLLM call (stats and the stop check run per chunk)")
    p.add_argument("--max-dup-rate", type=float, default=0.10)
    p.add_argument("--dup-min-count", type=int, default=200, help="candidates per category before the stop check applies")
    p.add_argument("--max-aura-tokens", type=int, default=240)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=320)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from vllm import LLM, SamplingParams  # noqa: E402  (vLLM venv only)

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    aura_len = C.AuraLength()

    pool = {c: [json.loads(line) for line in open(Path(args.pool) / f"{c}.jsonl")] for c in C.SHARES}
    if args.sample:
        want = {c: min(max(1, round(args.sample * s)), len(pool[c])) for c, s in C.SHARES.items()}
        requests = [r for c in C.SHARES for r in rng.sample(pool[c], want[c])]
    else:
        want = {c: len(pool[c]) for c in C.SHARES}
        requests = [r for c in C.SHARES for r in pool[c]]
    rng.shuffle(requests)  # interleave categories so every chunk mixes them
    for r in requests:
        r["ask"] = r["category"] in C.ASK_CATEGORIES and rng.random() < C.ASK_RATE
    seeds = C.per_request_seeds(args.seed + 7, len(requests))

    run_config = {
        "teacher": C.TEACHER, "teacher_license": C.TEACHER_LICENSE, "method": "sequence-level (text) distillation",
        "pool": args.pool, "requests_per_category": want, "system_prompts": C.SYSTEM,
        "story_judge": [C.JUDGE_SYSTEM, C.JUDGE_USER], "gsm8k_samples_per_question": C.GSM8K_SAMPLES,
        "followup_question": {"rate": C.ASK_RATE, "categories": sorted(C.ASK_CATEGORIES), "yes": C.ASK_YES, "no": C.ASK_NO},
        "temperature": args.temperature, "top_p": args.top_p, "max_new_tokens": args.max_new_tokens,
        "max_aura_tokens": args.max_aura_tokens, "seed": args.seed, "max_dup_rate": args.max_dup_rate,
        "prompt_licences": {"arith": "ours", "story": "ours", "explain": "Qwen output (Apache-2.0)",
                            "chat": "Qwen output (Apache-2.0)", "gsm8k": "MIT", "oasst2": "Apache-2.0"},
        "git_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=C.ROOT, capture_output=True,
                                     text=True).stdout.strip(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / "run_config.json").write_text(json.dumps(run_config, indent=1))
    print(f"Requests per category: {want} (total {len(requests):,})", flush=True)

    llm = LLM(model=C.TEACHER, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.85, seed=args.seed)
    t_start = time.time()

    def check(r, reply, finish):
        """First failing filter (None if it passes) and the AURA token count."""
        c = r["category"]
        if finish != "stop":
            return "truncated", 0
        if C.IDENTITY.search(reply):
            return "identity", 0
        if "INST]" in reply or "INST]" in r["prompt"]:
            return "format", 0
        if C.LATEX.search(reply):
            return "latex", 0
        if C.CODE.search(reply):
            return "code", 0
        if c in C.MATH:
            m = C.ANSWER.findall(reply)
            try:
                ok = bool(m) and float(m[-1].replace(",", "").rstrip(".")) == float(r["gold"])
            except ValueError:
                ok = False
            if not ok:
                return "wrong_answer", 0
            if not C.answer_is_last(reply):
                return "answer_not_last", 0
        elif "Answer:" in reply:
            return "answer_leak", 0
        n_tok = aura_len(r["prompt"], reply)
        return ("too_long", n_tok) if n_tok > args.max_aura_tokens else (None, n_tok)

    nd = C.NearDup(seed=args.seed)
    stats = defaultdict(Counter)
    kept_n, kept_tokens = Counter(), Counter()
    sp_judge = SamplingParams(temperature=0.0, max_tokens=4)
    stopped = None
    with open(out / "kept.jsonl", "w") as fk, open(out / "rejected.jsonl", "w") as fr:
        def reject(r, reason, reply):
            stats[r["category"]][reason] += 1
            fr.write(json.dumps({"category": r["category"], "reason": reason, "prompt": r["prompt"], "reply": reply}) + "\n")

        for i in range(0, len(requests), args.chunk):
            part = requests[i:i + args.chunk]
            convs = [[{"role": "system", "content": C.SYSTEM[r["category"]]
                       + ((C.ASK_YES if r["ask"] else C.ASK_NO) if r["category"] in C.ASK_CATEGORIES else "")},
                      {"role": "user", "content": r["prompt"]}] for r in part]
            sps = [SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_new_tokens,
                                  n=C.GSM8K_SAMPLES if r["category"] == "gsm8k" else 1, seed=s)
                   for r, s in zip(part, seeds[i:i + args.chunk])]
            # Phase 1: filters on every output; GSM8K keeps the shortest of its passing replies
            passed = []
            for r, o in zip(part, llm.chat(convs, sps, use_tqdm=False)):
                stats[r["category"]]["generated"] += 1
                stats[r["category"]]["replies_generated"] += len(o.outputs)
                ok, reasons = [], []
                for out_i in o.outputs:
                    reply = out_i.text.strip()
                    reason, n_tok = check(r, reply, out_i.finish_reason)
                    (reasons.append((reason, reply)) if reason else ok.append((n_tok, reply)))
                if ok:
                    n_tok, reply = min(ok)
                    passed.append((r, reply, n_tok))
                else:
                    reject(r, Counter(rs for rs, _ in reasons).most_common(1)[0][0], reasons[0][1])
            # Phase 2: the teacher judges whether each story reaches a clear ending
            stories = [k for k, (r, _, _) in enumerate(passed) if r["category"] == "story"]
            unfinished = set()
            if stories:
                jconvs = [[{"role": "system", "content": C.JUDGE_SYSTEM},
                           {"role": "user", "content": C.JUDGE_USER.format(story=passed[k][1])}] for k in stories]
                unfinished = {k for k, o in zip(stories, llm.chat(jconvs, sp_judge, use_tqdm=False))
                              if "UNFINISHED" in o.outputs[0].text.upper()}
            # Phase 3: near-duplicate check, then keep
            for k, (r, reply, n_tok) in enumerate(passed):
                c = r["category"]
                if k in unfinished:
                    reject(r, "no_ending", reply)
                    continue
                stats[c]["dup_checked"] += 1
                dup = nd.check_and_add(c, reply)
                if dup:
                    reject(r, dup, reply)
                    continue
                kept_n[c] += 1
                kept_tokens[c] += n_tok
                stats[c]["kept_question"] += C.ends_with_question(reply)
                stats[c]["kept_ask_assigned"] += r["ask"]
                fk.write(json.dumps({"category": c, "source": r["source"], "prompt_license": r["license"],
                                     "followup_asked": r["ask"], "teacher": C.TEACHER,
                                     "teacher_license": C.TEACHER_LICENSE, "aura_tokens": n_tok, "meta": r.get("meta"),
                                     "prompt": r["prompt"], "reply": reply,
                                     "line": f"[INST] {r['prompt']} [/INST] {reply}"}) + "\n")
            fk.flush()
            fr.flush()
            done = min(i + args.chunk, len(requests))
            dup = {c: (s["exact_dup"] + s["near_dup"]) / s["dup_checked"] for c, s in stats.items() if s["dup_checked"]}
            print(f"[{done:,}/{len(requests):,} requests, {(time.time() - t_start) / 60:.1f} min] kept "
                  f"{sum(kept_n.values()):,} samples, {sum(kept_tokens.values()) / 1e6:.2f}M AURA tokens; dup rate "
                  + " ".join(f"{c} {v:.1%}" for c, v in sorted(dup.items())), flush=True)
            over = [c for c, v in dup.items() if v >= args.max_dup_rate and stats[c]["dup_checked"] >= args.dup_min_count]
            if over:
                stopped = f"near-duplicate rate at or above {args.max_dup_rate:.0%} in {over} after {done:,} requests"
                print("STOPPING: " + stopped, flush=True)
                break

    summary = {"stopped_early": stopped, "minutes": round((time.time() - t_start) / 60, 1),
               "kept_samples": dict(kept_n), "kept_aura_tokens": dict(kept_tokens),
               "kept_aura_tokens_total": sum(kept_tokens.values()),
               "per_category": {c: dict(s) for c, s in stats.items()}}
    for c, s in stats.items():
        g = s["generated"]
        summary["per_category"][c].update({
            "identity_reject_rate": round(s["identity"] / g, 4) if g else 0,
            "dup_rate": round((s["exact_dup"] + s["near_dup"]) / s["dup_checked"], 4) if s["dup_checked"] else 0,
            "keep_rate": round(kept_n[c] / g, 4) if g else 0,
            "question_ending_rate": round(s["kept_question"] / kept_n[c], 4) if kept_n[c] else 0})
    allgen = sum(s["generated"] for s in stats.values())
    summary["identity_reject_rate_overall"] = round(sum(s["identity"] for s in stats.values()) / max(1, allgen), 4)
    (out / "stats.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))

    if args.sample:
        kept = [json.loads(line) for line in open(out / "kept.jsonl")]
        by = defaultdict(list)
        for k in kept:
            by[k["category"]].append(k)
        pick = []
        for c in sorted(by):  # at least 2 per category, rest at random
            pick += rng.sample(by[c], min(2, len(by[c])))
        rest = [k for k in kept if k not in pick]
        pick += rng.sample(rest, max(0, min(len(rest), 20 - len(pick))))
        rng.shuffle(pick)
        with open(out / "review_20.md", "w") as f:
            for n, k in enumerate(pick[:20], 1):
                f.write(f"### {n}. [{k['category']}] ({k['aura_tokens']} AURA tokens)\n**You:** {k['prompt']}\n\n"
                        f"**Aura:** {k['reply']}\n\n")
        print(f"\n20 random samples written to {out / 'review_20.md'}")


if __name__ == "__main__":
    main()
