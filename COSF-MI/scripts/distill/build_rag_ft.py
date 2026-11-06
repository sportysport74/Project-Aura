"""
RAG fine-tune set `data/distill_pilot/ft_rag` (experiments/plans/retrieval_prereg.md): ft_v2 with sources for explain / chat / oasst2.

  retrieve  (project venv)  each ft_v2 explain/chat/oasst2 row (train + val) gets the CORPUS store's top-3 passages; 20% get 3 random corpus
                            passages instead (to learn to ignore bad sources). → <out>/requests.jsonl. The Wikipedia store is never used here.
  generate  (vLLM venv)     Qwen2.5-7B-Instruct (Apache-2.0) rewrites the reply given the sources (ft_v2 persona + mood + follow-up rules +
                            source rules), 2 candidates; ft_v2's filters plus: no ≥ 12-word verbatim copy from a source, AURA length
                            (24k tokenizer) ≤ 500 for the whole conversation. Arithmetic / story / gsm8k rows are copied from ft_v2 unchanged.
                            → <out>/{train,val}.jsonl, rejected.jsonl, run_config.json, report.md

Usage (from COSF-MI/):
  .venv/bin/python scripts/distill/build_rag_ft.py retrieve --store data/retrieval/corpus --out data/distill_pilot/ft_rag
  VLLM_USE_FLASHINFER_SAMPLER=0 ~/venvs/vllm/bin/python -I scripts/distill/build_rag_ft.py generate --out data/distill_pilot/ft_rag
  (--limit N on both for smoke tests)
"""
import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src" / "python"))
import common as C  # noqa: E402
from build_factfirst import JOKE, SIMILE, SYSTEM_V2, sentences  # noqa: E402

SRC = ROOT / "data/distill_pilot/ft_v2"
RAG_CATS = ("explain", "chat", "oasst2")
SOURCE_RULES = (" You are given numbered sources. Use them when they answer the question: state the facts they give, in your own simple words. "
                "Do not copy sentences from them, do not mention them ('according to source 1'), and do not add facts that are in neither the "
                "sources nor common knowledge. If the sources don't answer the question, ignore them and answer briefly from general knowledge.")
WORDS = re.compile(r"[a-z0-9']+")


def copies(reply, passages, n=12):
    w = WORDS.findall(reply.lower())
    grams = {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}
    for p in passages:
        pw = WORDS.findall(p["text"].lower())
        if grams & {" ".join(pw[i:i + n]) for i in range(len(pw) - n + 1)}:
            return True
    return False


def load_split(split, limit):
    rows = [json.loads(line) for line in open(SRC / f"{split}.jsonl")]
    for r in rows:
        r["split"] = split
    return rows[:limit] if limit else rows


def cmd_retrieve(args):
    from cosf_mi.retrieval import Store
    st = Store(ROOT / args.store)
    rng = random.Random(args.seed)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    reqs = [r for split in ("train", "val") for r in load_split(split, args.limit) if r["category"] in RAG_CATS]
    t0 = time.time()
    res = st.search([r["prompt"] for r in reqs], k=3)
    with open(out / "requests.jsonl", "w") as f:
        for r, ps in zip(reqs, res):
            kind = "retrieved"
            if rng.random() < args.random_frac:
                ps, kind = [dict(st.get(rng.randrange(len(st))), score=None) for _ in range(3)], "random"
            f.write(json.dumps({**r, "sources": ps, "sources_kind": kind, "store": args.store}, ensure_ascii=False) + "\n")
    print(f"retrieved for {len(reqs):,} rows in {time.time() - t0:.0f} s -> {out / 'requests.jsonl'}")


def cmd_generate(args):
    import importlib.util  # load retrieval.py alone: the cosf_mi package __init__ needs scipy, which the vLLM venv lacks
    spec = importlib.util.spec_from_file_location("aura_retrieval", ROOT / "src/python/cosf_mi/retrieval.py")
    retrieval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(retrieval)
    format_prompt = retrieval.format_prompt
    from vllm import LLM, SamplingParams
    out = ROOT / args.out
    reqs = [json.loads(line) for line in open(out / "requests.jsonl")]
    if args.limit:
        reqs = reqs[:args.limit]
    aura_len = C.AuraLength(ROOT / "models/tokenizer_clean24k.model")
    llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", dtype="bfloat16", max_model_len=3072, gpu_memory_utilization=0.85, seed=args.seed)
    nd = C.NearDup(seed=args.seed)
    stats = defaultdict(Counter)
    kept = {}

    def check(r, prompt, reply, finish):
        if finish != "stop":
            return "truncated"
        if C.IDENTITY.search(reply):
            return "identity"
        if "INST]" in reply or "Sources:" in reply:
            return "format"
        if C.LATEX.search(reply):
            return "latex"
        if C.CODE.search(reply):
            return "code"
        if "Answer:" in reply:
            return "answer_leak"
        if re.search(r"\b(source|sources)\s*\[?\d", reply, re.I) or re.search(r"according to (the )?source", reply, re.I):
            return "mentions_source"
        if copies(reply, r["sources"]):
            return "copies_source"
        if aura_len(prompt, reply) > args.max_aura_tokens:
            return "too_long"
        sents = sentences(reply)
        if sents and SIMILE.search(sents[0]):
            return "simile_first"
        if len(SIMILE.findall(reply)) > 1:
            return "too_many_similes"
        if r.get("serious") and (SIMILE.search(reply) or JOKE.search(reply)):
            return "playful_serious"
        return None

    t0 = time.time()
    with open(out / "rejected.jsonl", "w") as fr:
        for i in range(0, len(reqs), args.chunk):
            part = reqs[i:i + args.chunk]
            prompts = [format_prompt(r["prompt"], r["sources"]) for r in part]
            convs = [[{"role": "system", "content": SYSTEM_V2[r["category"]] + (C.ASK_YES if r.get("followup_asked") else C.ASK_NO)
                       + SOURCE_RULES},
                      {"role": "user", "content": p}] for r, p in zip(part, prompts)]
            sps = [SamplingParams(temperature=0.7, top_p=0.95, max_tokens=320, n=2, seed=args.seed * 1_000_003 + i + j)
                   for j in range(len(part))]
            for r, p, o in zip(part, prompts, llm.chat(convs, sps, use_tqdm=False)):
                c = r["category"]
                stats[c]["requests"] += 1
                reasons, choice = [], None
                for oi in o.outputs:
                    reply = oi.text.strip()
                    why = check(r, p, reply, oi.finish_reason) or nd.check_and_add(c, reply)
                    if why:
                        reasons.append(why)
                        continue
                    choice = reply
                    break
                if choice is None:
                    why = Counter(reasons).most_common(1)[0][0]
                    stats[c][why] += 1
                    fr.write(json.dumps({"category": c, "reason": why, "prompt": p, "reply": o.outputs[0].text}, ensure_ascii=False) + "\n")
                    continue
                stats[c]["kept"] += 1
                stats[c][f"kept_{r['sources_kind']}"] += 1
                kept[(r["split"], r["prompt"])] = (p, choice, r)
            print(f"  {min(i + args.chunk, len(reqs)):,}/{len(reqs):,} [{time.time() - t0:.0f} s]", flush=True)
    for split in ("train", "val"):
        n = 0
        with open(out / f"{split}.jsonl", "w") as f:
            for r in load_split(split, 0):
                if r["category"] in RAG_CATS:
                    k = kept.get((split, r["prompt"]))
                    if k is None:
                        continue
                    p, reply, req = k
                    row = {"category": r["category"], "prompt": p, "reply": reply, "line": f"[INST] {p} [/INST] {reply}",
                           "sources_kind": req["sources_kind"], "source_ids": [s["id"] for s in req["sources"]],
                           "teacher": "Qwen/Qwen2.5-7B-Instruct", "teacher_license": "Apache-2.0 (Copyright 2024 Alibaba Cloud)",
                           "sources_license": "ODC-By 1.0 (Cosmopedia v2 / FineWeb-Edu passages; corpus store only)"}
                else:
                    if args.limit:  # smoke: keep the non-RAG rows small too
                        if n >= 2 * args.limit:
                            continue
                    row = r
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
        print(f"{split}: {n:,} rows")
    cfg = {"teacher": "Qwen/Qwen2.5-7B-Instruct", "teacher_license": "Apache-2.0 (Copyright 2024 Alibaba Cloud)", "src": str(SRC.relative_to(ROOT)),
           "store": reqs[0]["store"] if reqs else None, "rag_categories": RAG_CATS, "source_rules": SOURCE_RULES,
           "max_aura_tokens": args.max_aura_tokens, "seed": args.seed, "limit": args.limit,
           "note": "Sources come from the CORPUS store only; the Wikipedia (CC BY-SA) store is never used for training data."}
    (out / "run_config.json").write_text(json.dumps(cfg, indent=1))
    L = ["# ft_rag build report\n", "| category | requests | kept | kept retrieved / random | top reject reasons |", "|---|---|---|---|---|"]
    for c, s in sorted(stats.items()):
        rej = ", ".join(f"{k} {v}" for k, v in s.most_common() if k not in ("requests", "kept") and not k.startswith("kept_"))[:120]
        L.append(f"| {c} | {s['requests']:,} | {s['kept']:,} ({s['kept'] / max(1, s['requests']):.1%}) | {s['kept_retrieved']:,} / "
                 f"{s['kept_random']:,} | {rej} |")
    (out / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("retrieve")
    r.add_argument("--store", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--random-frac", type=float, default=0.2)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--seed", type=int, default=0)
    g = sub.add_parser("generate")
    g.add_argument("--out", required=True)
    g.add_argument("--limit", type=int, default=0)
    g.add_argument("--chunk", type=int, default=8000)
    g.add_argument("--max-aura-tokens", type=int, default=500)
    g.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    {"retrieve": cmd_retrieve, "generate": cmd_generate}[args.cmd](args)


if __name__ == "__main__":
    main()
