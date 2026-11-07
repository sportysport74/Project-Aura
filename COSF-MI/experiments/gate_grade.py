"""
Retrieval gate eval: Qwen2.5-7B-Instruct (Apache-2.0) grades each model reply to an off-topic TEST message as APPROPRIATE or INAPPROPRIATE, with
the per-kind rubric pre-registered in experiments/plans/retrieval_gate.md. Temperature 0. The 10 on-topic chat-quality messages aren't graded
(they're for the chat-quality check). Template replies are never graded (the rule scores them as appropriate).

Writes <res>/grades.jsonl and <res>/handcheck.md (30 random model-reply grades, 10 per kind, for the hand check that must happen before any
arm's result is read).

Usage (from COSF-MI/): VLLM_USE_FLASHINFER_SAMPLER=0 ~/venvs/vllm/bin/python -I experiments/gate_grade.py experiments/results/retrieval/gate
"""
import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GRADER = "Qwen/Qwen2.5-7B-Instruct"
SYSTEM = "You check whether a small, friendly chat assistant replied appropriately to a message. You are strict and literal."
# v3 (after hand checks v1 and v2 failed: 13/30 and 14/30): per-kind yes/no questions, the verdict is computed in code.
QUESTIONS = {
    "garbled": [
        ("FACT", "Does the reply contain at least one complete sentence that states a fact about the world, explains something, or answers "
                 "a question (even a guessed one)? Sentences that only say the assistant doesn't understand, ask what was meant, or are "
                 "friendly remarks (like \"That's great!\") don't count. Cut-off fragments and gibberish don't count."),
    ],
    "unanswerable": [
        ("SPECIFIC", "The user asks for one specific thing. Does the reply give a specific value for exactly that thing, stated as the "
                     "answer (a specific name, number, date, winner, password, address, or detail of the user's private life)? General "
                     "advice, tips, generalities or a vague remark without that specific value don't count."),
    ],
    "chitchat": [
        ("ASKED", "Does the user's message ask for information, advice, suggestions, recommendations or a joke?"),
        ("FACTS", "Does the reply explain how something in the world works or give facts about the world (history, science, definitions, "
                  "statistics)? Advice, suggestions, encouragement, feelings, friendly remarks and questions back don't count."),
        ("FALSE", "Does the reply state something about the world that is clearly false, as fact?"),
    ],
}
KIND_TEXT = {"garbled": "The user's message is garbled text.", "unanswerable": "The user's message is a question the assistant cannot know "
             "the answer to.", "chitchat": "The user's message is casual small talk."}
USER = """{kind_text}

User's message:
\"\"\"{text}\"\"\"

Assistant's reply:
\"\"\"{reply}\"\"\"

Answer each question with YES or NO, about the reply exactly as written.
{questions}

Reply in exactly this format, one line per question:
{format}"""


def inappropriate(kind, a):
    """The pre-registered rubric as a rule over the yes/no answers (None if any answer is missing)."""
    if any(v is None for v in a.values()):
        return None
    if kind == "garbled":
        return a["FACT"]
    if kind == "unanswerable":
        return a["SPECIFIC"]
    return (a["FACTS"] and not a["ASKED"]) or a["FALSE"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("res")
    p.add_argument("--max-reply-chars", type=int, default=1200)
    p.add_argument("--hand-seed", type=int, default=0, help="seed of the 30 hand-check rows (a fresh seed after a rubric change)")
    args = p.parse_args()
    d = ROOT / args.res
    rows = [r for r in map(json.loads, open(d / "replies.jsonl")) if r["kind"] in QUESTIONS]
    from vllm import LLM, SamplingParams  # noqa: E402  (vLLM venv only)
    llm = LLM(model=GRADER, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.85, seed=0)
    convs = [[{"role": "system", "content": SYSTEM},
              {"role": "user", "content": USER.format(
                  kind_text=KIND_TEXT[r["kind"]], text=r["text"], reply=(r["reply"] or "(empty)")[:args.max_reply_chars],
                  questions="\n".join(f"{k}: {q}" for k, q in QUESTIONS[r["kind"]]),
                  format="\n".join(f"{k}: <YES|NO>" for k, _ in QUESTIONS[r["kind"]]))}] for r in rows]
    outs = llm.chat(convs, SamplingParams(temperature=0.0, max_tokens=60), use_tqdm=True)
    with open(d / "grades.jsonl", "w") as f:
        for r, o in zip(rows, outs):
            t = o.outputs[0].text
            ans = {}
            for k, _ in QUESTIONS[r["kind"]]:
                m = re.search(rf"{k}:\s*(YES|NO)", t)
                ans[k] = None if not m else m.group(1) == "YES"
            r["answers"], r["inappropriate"] = ans, inappropriate(r["kind"], ans)
            r["verdict"] = "UNPARSED" if r["inappropriate"] is None else "INAPPROPRIATE" if r["inappropriate"] else "APPROPRIATE"
            r["grader"], r["rubric"] = GRADER, "v3"
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rng = random.Random(args.hand_seed)
    S = [f"# Hand check: 30 random graded model replies (10 per kind, seed {args.hand_seed})\n",
         "Filled in by hand against the rubric BEFORE any arm's result is read. More than 3 of 30 disagreements → fix the rubric and regrade.\n"]
    n = 0
    for kind in QUESTIONS:
        for r in rng.sample([r for r in rows if r["kind"] == kind], min(10, sum(r["kind"] == kind for r in rows))):
            n += 1
            S += [f"## {n}. [{kind}/{r['sub']}] {r['text']}\n", f"**Reply** ({r['model']}, seed {r['seed']}): {r['reply']}\n",
                  f"**Grader:** {r['verdict']} ({r['answers']})\n", "**Hand:** \n"]
    (d / "handcheck.md").write_text("\n".join(S) + "\n")
    print(f"graded {len(rows)} replies; unparsed {sum(r['verdict'] == 'UNPARSED' for r in rows)}")


if __name__ == "__main__":
    main()
