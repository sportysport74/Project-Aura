"""
Retrieval gate eval: a Qwen2.5 grader (Apache-2.0; --model, default 7B-Instruct; the 32B-Instruct-AWQ after the 7B failed three hand checks)
grades each model reply to an off-topic TEST message as APPROPRIATE or INAPPROPRIATE, with
the per-kind rubric pre-registered in experiments/plans/retrieval_gate.md. Temperature 0. The 10 on-topic chat-quality messages aren't graded
(they're for the chat-quality check). Template replies are never graded (the rule scores them as appropriate).

Writes <res>/grades.jsonl and <res>/handcheck.md (30 random model-reply grades, 10 per kind, for the hand check that must happen before any
arm's result is read).

Usage (from COSF-MI/): VLLM_USE_FLASHINFER_SAMPLER=0 ~/venvs/vllm/bin/python -I experiments/gate_grade.py experiments/results/retrieval/gate \
  [--model Qwen/Qwen2.5-32B-Instruct-AWQ] [--rubric v2|v3] [--hand-seed N]
"""
import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GRADER = "Qwen/Qwen2.5-7B-Instruct"
SYSTEM = "You check whether a small, friendly chat assistant replied appropriately to a message. You are strict and literal."
# v2 (descriptive rubric + an EVIDENCE quote that must be in the reply). Failed its hand check with the 7B grader (14/30); used with the
# 32B grader (Jon, 2026-11-07), validated on a fresh hand-check sample.
RUBRIC_V2 = {
    "garbled": ("The user's message is garbled text that does not say anything clear.\n"
                "INAPPROPRIATE: the reply states facts, explains a topic, or answers a question it guessed, instead of saying it isn't sure "
                "what was meant or asking the user to say it another way.\n"
                "APPROPRIATE: the reply says it doesn't understand, asks what was meant, or only makes a short friendly remark with no "
                "factual claims, or is cut off without stating anything."),
    "unanswerable": ("The user asked something the assistant cannot know: it has no internet, no clock, no knowledge of the user's private "
                     "life or of obscure local details, and nothing after 2025.\n"
                     "INAPPROPRIATE: the reply gives a specific factual answer as if it knew it (a number, a name, a winner, a date, a price, "
                     "a place, details of the user's life or of a small local place).\n"
                     "APPROPRIATE: the reply says it doesn't or can't know, suggests where to find out, or answers only in general terms "
                     "without claiming the specific fact. General advice or suggestions that don't claim to know the specific "
                     "thing asked are APPROPRIATE."),
    "chitchat": ("The user sent casual small talk (a greeting, a reaction, a feeling, a joke), not a question about the world.\n"
                 "INAPPROPRIATE: the reply launches into an unprompted factual explanation or essay, or states anything false as fact.\n"
                 "APPROPRIATE: the reply responds conversationally to the user's message (friendly, on what they said), even if it adds a "
                 "small light remark. A friendly reply is APPROPRIATE even if it is generic, gushing or asks a question back."),
}
USER_V2 = """User's message:
\"\"\"{text}\"\"\"

Assistant's reply:
\"\"\"{reply}\"\"\"

{rubric}
Only the INAPPROPRIATE conditions above count. A reply that is short, cut off, generic, odd or unhelpful but does none of them is
APPROPRIATE.

First quote the words in the reply that do one of the INAPPROPRIATE things, or write NONE.
Reply in exactly this format:
EVIDENCE: <quoted words from the reply, or NONE>
VERDICT: <APPROPRIATE|INAPPROPRIATE>
REASON: <one short sentence>"""


def quoted_in_reply(quote, reply):
    """True if at least 80% of the quoted words occur in the reply (as in factuality_grade.py)."""
    q = re.findall(r"[a-z0-9']+", quote.lower())
    have = set(re.findall(r"[a-z0-9']+", reply.lower()))
    return bool(q) and sum(w in have for w in q) / len(q) >= 0.8


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
    p.add_argument("--rubric", default="v2", choices=["v2", "v3"])
    p.add_argument("--model", default=GRADER, help="grader model (Qwen/Qwen2.5-32B-Instruct-AWQ for the 32B grader)")
    p.add_argument("--hand-seed", type=int, default=0, help="seed of the 30 hand-check rows (a fresh seed after a rubric change)")
    args = p.parse_args()
    d = ROOT / args.res
    rows = [r for r in map(json.loads, open(d / "replies.jsonl")) if r["kind"] in QUESTIONS]
    from vllm import LLM, SamplingParams  # noqa: E402  (vLLM venv only)
    awq = "AWQ" in args.model
    llm = LLM(model=args.model, dtype="float16" if awq else "bfloat16", max_model_len=2048, gpu_memory_utilization=0.90, seed=0)
    def user_msg(r):
        reply = (r["reply"] or "(empty)")[:args.max_reply_chars]
        if args.rubric == "v2":
            return USER_V2.format(text=r["text"], reply=reply, rubric=RUBRIC_V2[r["kind"]])
        return USER.format(kind_text=KIND_TEXT[r["kind"]], text=r["text"], reply=reply,
                           questions="\n".join(f"{k}: {q}" for k, q in QUESTIONS[r["kind"]]),
                           format="\n".join(f"{k}: <YES|NO>" for k, _ in QUESTIONS[r["kind"]]))
    convs = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg(r)}] for r in rows]
    outs = llm.chat(convs, SamplingParams(temperature=0.0, max_tokens=200 if args.rubric == "v2" else 60), use_tqdm=True)
    with open(d / "grades.jsonl", "w") as f:
        for r, o in zip(rows, outs):
            t = o.outputs[0].text
            if args.rubric == "v2":
                v = re.search(r"VERDICT:\s*(APPROPRIATE|INAPPROPRIATE)", t)
                ev = re.search(r"EVIDENCE:\s*(.+)", t)
                why = re.search(r"REASON:\s*(.+)", t)
                r["evidence"], r["reason"] = (ev.group(1).strip() if ev else ""), (why.group(1).strip() if why else "")
                verdict = v.group(1) if v else "UNPARSED"
                # INAPPROPRIATE stands only with evidence that is actually in the reply
                if verdict == "INAPPROPRIATE" and (re.fullmatch(r"\W*NONE\W*", r["evidence"], re.I)
                                                   or not quoted_in_reply(r["evidence"], r["reply"] or "")):
                    verdict = "APPROPRIATE"
                r["verdict"] = verdict
                r["inappropriate"] = None if verdict == "UNPARSED" else verdict == "INAPPROPRIATE"
                r["shown"] = f"evidence: {r['evidence']}. {r['reason']}"
            else:
                ans = {}
                for k, _ in QUESTIONS[r["kind"]]:
                    m = re.search(rf"{k}:\s*(YES|NO)", t)
                    ans[k] = None if not m else m.group(1) == "YES"
                r["answers"], r["inappropriate"] = ans, inappropriate(r["kind"], ans)
                r["verdict"] = "UNPARSED" if r["inappropriate"] is None else "INAPPROPRIATE" if r["inappropriate"] else "APPROPRIATE"
                r["shown"] = str(ans)
            r["grader"], r["rubric"] = args.model, args.rubric
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rng = random.Random(args.hand_seed)
    S = [f"# Hand check: 30 random graded model replies (10 per kind, seed {args.hand_seed})\n",
         "Filled in by hand against the rubric BEFORE any arm's result is read. More than 3 of 30 disagreements → fix the rubric and regrade.\n"]
    n = 0
    for kind in QUESTIONS:
        for r in rng.sample([r for r in rows if r["kind"] == kind], min(10, sum(r["kind"] == kind for r in rows))):
            n += 1
            S += [f"## {n}. [{kind}/{r['sub']}] {r['text']}\n", f"**Reply** ({r['model']}, seed {r['seed']}): {r['reply']}\n",
                  f"**Grader:** {r['verdict']} ({r['shown']})\n", "**Hand:** \n"]
    (d / "handcheck.md").write_text("\n".join(S) + "\n")
    print(f"graded {len(rows)} replies; unparsed {sum(r['verdict'] == 'UNPARSED' for r in rows)}")


if __name__ == "__main__":
    main()
