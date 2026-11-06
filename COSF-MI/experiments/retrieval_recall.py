"""
Retrieval recall@k on the 100 factuality questions (experiments/plans/retrieval_prereg.md; the gate: corpus recall@3 >= 0.6).

  retrieve STORE --out FILE   (project venv)  top-k passages per question → FILE (jsonl)
  grade FILE [FILE ...]       (vLLM venv)     Qwen2.5-7B-Instruct (Apache-2.0), temperature 0, judges each passage: does it state the
                                              question's key fact? → recall@1 / @3 per file, written next to each file as *.summary.json

Usage (from COSF-MI/):
  .venv/bin/python experiments/retrieval_recall.py retrieve data/retrieval/corpus --out experiments/results/retrieval/recall_corpus.jsonl
  VLLM_USE_FLASHINFER_SAMPLER=0 ~/venvs/vllm/bin/python -I experiments/retrieval_recall.py grade experiments/results/retrieval/recall_*.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "experiments/factuality/questions.jsonl"
SYSTEM = "You check whether a text passage answers a children's science or general-knowledge question. Reply with exactly one word."
USER = """Question: {question}
Key facts a correct answer must contain: {key_facts}

Passage:
\"\"\"{passage}\"\"\"

Does this passage state the key fact (or most of it)? Reply YES or NO."""


def retrieve(args):
    sys.path.insert(0, str(ROOT / "src" / "python"))
    from cosf_mi.retrieval import Store
    qs = [json.loads(line) for line in open(QUESTIONS)]
    qs = qs[:args.limit] if args.limit else qs
    res = Store(ROOT / args.store).search([q["question"] for q in qs], k=args.k)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for q, ps in zip(qs, res):
            f.write(json.dumps({"id": q["id"], "store": args.store, "passages": ps}, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(qs)} questions × {args.k})")


def grade(args):
    from vllm import LLM, SamplingParams
    qs = {q["id"]: q for q in map(json.loads, open(QUESTIONS))}
    files = [ROOT / f for f in args.files]
    rows = {f: [json.loads(line) for line in open(f)] for f in files}
    convs, keys = [], []
    for f, rs in rows.items():
        for r in rs:
            q = qs[r["id"]]
            for j, p in enumerate(r["passages"]):
                convs.append([{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": USER.format(question=q["question"], key_facts=q["key_facts"], passage=p["text"])}])
                keys.append((f, r["id"], j))
    llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.85, seed=0)
    outs = llm.chat(convs, SamplingParams(temperature=0.0, max_tokens=3), use_tqdm=False)
    hit = {}
    for (f, qid, j), o in zip(keys, outs):
        hit[(f, qid, j)] = "YES" in o.outputs[0].text.upper()
    for f, rs in rows.items():
        r1 = sum(hit[(f, r["id"], 0)] for r in rs) / len(rs)
        r3 = sum(any(hit[(f, r["id"], j)] for j in range(min(3, len(r["passages"])))) for r in rs) / len(rs)
        summ = {"file": f.name, "questions": len(rs), "recall@1": round(r1, 3), "recall@3": round(r3, 3),
                "grader": "Qwen/Qwen2.5-7B-Instruct (Apache-2.0), temperature 0"}
        (f.with_suffix(".summary.json")).write_text(json.dumps(summ, indent=1))
        print(json.dumps(summ))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("retrieve")
    r.add_argument("store")
    r.add_argument("--out", required=True)
    r.add_argument("--k", type=int, default=3)
    r.add_argument("--limit", type=int, default=0)
    g = sub.add_parser("grade")
    g.add_argument("files", nargs="+")
    args = p.parse_args()
    {"retrieve": retrieve, "grade": grade}[args.cmd](args)


if __name__ == "__main__":
    main()
