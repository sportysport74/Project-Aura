"""
Factuality eval, step 2 of 2: Qwen2.5-7B-Instruct (Apache-2.0, the pilot's teacher) grades each reply in
<dir>/replies.jsonl against the question's reference answer and key facts (experiments/factuality/questions.jsonl).

Verdicts: CORRECT (states the key facts, no false claims about the topic) = 1, PARTIAL (some key fact right, but incomplete or
with a minor error) = 0.5, INCORRECT (key fact wrong or missing, or a major false claim) = 0, OFF_TOPIC (doesn't attempt the
question) = 0. FALSE_CLAIM flags any false factual statement; a comparison clearly offered as a comparison ("like a
blanket") is not a claim. Temperature 0.

Two control rows check the grader before any model is read: every reference answer graded as a reply (should be CORRECT),
and every question paired with another question's reference (should be OFF_TOPIC or INCORRECT).

Writes <dir>/grades.jsonl, <dir>/summary.md and <dir>/spot_check_seed<N>.md (20 random graded model replies, for a hand check).

Usage (from COSF-MI/; vLLM venv, WSL has no nvcc):
  VLLM_USE_FLASHINFER_SAMPLER=0 ~/venvs/vllm/bin/python -I experiments/factuality_grade.py experiments/results/factuality
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "experiments/factuality/questions.jsonl"
GRADER = "Qwen/Qwen2.5-7B-Instruct"
GRADER_LICENSE = "Apache-2.0 (Copyright 2024 Alibaba Cloud)"
SCORE = {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0, "OFF_TOPIC": 0.0}

SYSTEM = ("You grade answers to children's science and general-knowledge questions for factual accuracy. You are strict "
          "about facts and ignore style, tone, spelling and grammar.")
USER = """Question: {question}

Reference answer: {reference}
Key facts a correct answer must contain: {key_facts}

Answer to grade:
\"\"\"{reply}\"\"\"

Grade the answer:
- CORRECT: it states the key facts correctly (in any words) and makes no false factual claim about the topic.
- PARTIAL: it gets some key fact right but misses part of it, or it is right overall but contains a minor error.
- INCORRECT: the key facts are wrong or missing, or it makes a major false claim.
- OFF_TOPIC: it does not try to answer this question.
A comparison clearly offered as a comparison (for example "like a warm blanket") is not a factual claim. A false
statement presented as fact is.

First quote the words in the answer that state one of the key facts, or write NONE if no key fact is stated. A comparison
is not a key fact: "a plane is like a pancake" states nothing about lift.

Reply in exactly this format:
FACTS_FOUND: <quoted words from the answer, or NONE>
VERDICT: <CORRECT|PARTIAL|INCORRECT|OFF_TOPIC>
FALSE_CLAIM: <YES|NO>
REASON: <one short sentence>"""


def quoted_in_reply(facts, reply):
    """True if at least 80% of the quoted words occur in the reply: the grader once quoted a 'fact' the reply never said."""
    q = re.findall(r"[a-z0-9']+", facts.lower())
    have = set(re.findall(r"[a-z0-9']+", reply.lower()))
    return bool(q) and sum(w in have for w in q) / len(q) >= 0.8


def parse(text, reply):
    """(verdict, false_claim, reason, facts_found). A CORRECT or PARTIAL verdict becomes INCORRECT when FACTS_FOUND is NONE or
    quotes words that aren't in the reply: the first grading pass (no FACTS_FOUND line) gave PARTIAL to replies that state no
    key fact at all (8 of 20 hand-checked), and the second quoted a fact the reply didn't contain."""
    v = re.search(r"VERDICT:\s*(CORRECT|PARTIAL|INCORRECT|OFF_TOPIC)", text)
    fc = re.search(r"FALSE_CLAIM:\s*(YES|NO)", text)
    r = re.search(r"REASON:\s*(.+)", text)
    ff = re.search(r"FACTS_FOUND:\s*(.+)", text)
    facts = ff.group(1).strip() if ff else ""
    verdict = v.group(1) if v else "UNPARSED"
    if verdict in ("CORRECT", "PARTIAL") and (re.fullmatch(r"\W*NONE\W*", facts, re.I) or not quoted_in_reply(facts, reply)):
        verdict = "INCORRECT"
    return verdict, (fc.group(1) == "YES" if fc else None), (r.group(1).strip() if r else ""), facts


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir", help="results dir holding replies.jsonl")
    p.add_argument("--max-reply-chars", type=int, default=1200)
    p.add_argument("--spot-seed", type=int, default=0, help="seed for the 20 spot-check rows")
    p.add_argument("--limit", type=int, default=0, help="only the first N questions (dry runs; 0 = all)")
    args = p.parse_args()
    d = ROOT / args.dir
    qs = {q["id"]: q for q in map(json.loads, open(QUESTIONS))}
    if args.limit:
        qs = dict(list(qs.items())[:args.limit])
    ids = list(qs)
    rows = [r for r in map(json.loads, open(d / "replies.jsonl")) if r["id"] in qs]
    shift = {i: ids[(k + 37) % len(ids)] for k, i in enumerate(ids)}
    rows += [{"model": "control_reference", "encoding": "-", "decoding": "-", "seed": 0, "id": i,
              "reply": qs[i]["reference"]} for i in ids]
    rows += [{"model": "control_mismatched", "encoding": "-", "decoding": "-", "seed": 0, "id": i,
              "reply": qs[shift[i]]["reference"]} for i in ids]

    from vllm import LLM, SamplingParams  # noqa: E402  (vLLM venv only)
    llm = LLM(model=GRADER, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.85, seed=0)
    convs = [[{"role": "system", "content": SYSTEM},
              {"role": "user", "content": USER.format(question=qs[r["id"]]["question"], reference=qs[r["id"]]["reference"],
                                                      key_facts=qs[r["id"]]["key_facts"],
                                                      reply=(r["reply"] or "(empty)")[:args.max_reply_chars])}]
             for r in rows]
    outs = llm.chat(convs, SamplingParams(temperature=0.0, max_tokens=200), use_tqdm=True)
    with open(d / "grades.jsonl", "w") as f:
        for r, o in zip(rows, outs):
            r["verdict"], r["false_claim"], r["reason"], r["facts_found"] = parse(o.outputs[0].text, r["reply"] or "")
            r["grader"] = GRADER
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # summary: per (model, encoding, decoding), mean score over all replies (sampled_rep pools its seeds)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["encoding"], r["decoding"])].append(r)
    L = ["# Factuality eval (100 explain questions)\n",
         f"Grader {GRADER} ({GRADER_LICENSE}), temperature 0. Score: CORRECT 1, PARTIAL 0.5, otherwise 0. "
         "sampled_rep pools its seeds (n = 100 × seeds). Controls check the grader: reference answers should score ~1, "
         "mismatched references ~0.\n",
         "| model | encoding | decoding | n | score | correct | partial | incorrect | off-topic | false claim | unparsed |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (m, e, dec), g in groups.items():
        n = len(g)
        c = defaultdict(int)
        for r in g:
            c[r["verdict"]] += 1
        score = sum(SCORE.get(r["verdict"], 0.0) for r in g) / n
        fc = sum(bool(r["false_claim"]) for r in g) / n
        L.append(f"| {m} | {e} | {dec} | {n} | **{score:.3f}** | {c['CORRECT'] / n:.0%} | {c['PARTIAL'] / n:.0%} | "
                 f"{c['INCORRECT'] / n:.0%} | {c['OFF_TOPIC'] / n:.0%} | {fc:.0%} | {c['UNPARSED']} |")
    # per topic, sampled_rep only, models only
    topics = sorted({q["topic"] for q in qs.values()})
    L += ["\n## Per topic (score, sampled_rep, pooled seeds)\n",
          "| model | encoding | " + " | ".join(topics) + " |", "|---|---|" + "---|" * len(topics)]
    for (m, e, dec), g in groups.items():
        if dec != "sampled_rep":
            continue
        by = defaultdict(list)
        for r in g:
            by[qs[r["id"]]["topic"]].append(SCORE.get(r["verdict"], 0.0))
        L.append(f"| {m} | {e} | " + " | ".join(f"{sum(by[t]) / len(by[t]):.2f}" for t in topics) + " |")
    (d / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))

    model_rows = [r for r in rows if not r["model"].startswith("control_")]
    pick = random.Random(args.spot_seed).sample(model_rows, min(20, len(model_rows)))
    S = [f"# Spot check: 20 random graded replies (seed {args.spot_seed})\n",
         "Hand verdict column is filled in after reading each reply against the reference.\n"]
    for k, r in enumerate(pick, 1):
        q = qs[r["id"]]
        S += [f"## {k}. {q['question']} ({r['model']}, {r['encoding']}, {r['decoding']} seed {r['seed']})\n",
              f"**Reference:** {q['reference']}\n", f"**Reply:** {r['reply']}\n",
              f"**Grader:** {r['verdict']}, false claim {r['false_claim']}, facts found: {r['facts_found']}. {r['reason']}\n",
              "**Hand:** \n"]
    (d / f"spot_check_seed{args.spot_seed}.md").write_text("\n".join(S) + "\n")


if __name__ == "__main__":
    main()
