"""
Message sets for the retrieval gate (experiments/plans/retrieval_gate.md). vLLM venv (Qwen2.5-7B-Instruct, Apache-2.0); no cosf_mi import.

Writes <out>/{ontopic,chitchat,unanswerable,garbled}.jsonl (fields: id, kind, sub, split, text) and <out>/multiturn_candidates.jsonl, plus
<out>/build.json. Every off-topic kind is deduplicated (exact + MinHash near-dup) over the whole kind before the 50/50 tune/test split (seed 0).
  ontopic       ft_v2 val prompts, explain + oasst2
  chitchat      Qwen-written casual messages; + a fixed hand list of very short ones, test split only
  unanswerable  Qwen-written clear questions outside the store: real-time, personal, after the corpus, very niche
  garbled       programmatic: keyboard mash, letters shuffled within words; word salad in the test split only
  multiturn     Qwen-written two-turn conversations (turn 1, follow-up, reference, key facts), read by hand before use

Usage (from COSF-MI/):
  VLLM_USE_FLASHINFER_SAMPLER=0 ~/venvs/vllm/bin/python -I scripts/retrieval/build_gate_sets.py --out data/retrieval/gate [--smoke]
"""
import argparse
import json
import random
import re
import string
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts/distill"))
import common as C  # noqa: E402

SHORT_CHITCHAT = ["lol", "hi", "ok", "haha", "thx", "brb", "hey", "cool", "nice", "lmao", "yo", "thanks!", "k", "omg", "bye",
                  "good night", "sup", "hmm", "wow", "same"]
CHIT_HINTS = ["a greeting", "a reaction to something funny", "saying they're bored", "saying they're tired", "thanking the assistant",
              "saying goodbye", "text-speak with slang", "a short reaction with an emoji", "complaining a little about homework",
              "excited about the weekend", "telling how their day went", "a compliment to the assistant", "agreeing with something",
              "a one-word reaction", "saying they're hungry", "a playful tease", "saying they're back after a break", "a sigh of relief",
              "small talk about the weather outside their window (not a question)", "sharing they got a good grade"]
CHIT_SYSTEM = ("You write realistic chat messages that people send to a friendly AI chat buddy. Reply with only the message, no quotes, no "
               "explanation.")
CHIT_USER = ("Write one short casual message (1 to 12 words) that someone might send to a friendly AI chat buddy. It is small talk, a "
             "reaction, a greeting, a feeling or a joke. It must NOT ask about facts, science, history, or how something works. Kind: {hint}.")
UNANS = {
    "realtime": "a question that needs live, current information the assistant can't have (today's weather, the time right now, a "
                "current price, live scores, today's news)",
    "personal": "a question about the asker's own private life that the assistant can't know (their pet's name, what they ate, "
                "where they left something, their friend's birthday)",
    "recent": "a question about a specific event in 2026 or 2027 (a winner, a release, an election result, a record)",
    "niche": "a question about a very specific, obscure fact: the population of a small named village, the opening hours of a named "
             "local shop, the name of a specific minor official, the address of a small business",
}
UNANS_SYSTEM = "You write realistic questions that people send to an AI assistant. Reply with only the question, no quotes, no explanation."
UNANS_USER = "Write one clear, natural question in plain English (5 to 20 words): {desc}. Make it specific."
MT_SYSTEM = "You write short test conversations for a children's science and general-knowledge assistant. Reply with JSON only."
MT_USER = ("Topic: {topic}. Write a two-turn exchange as JSON with keys turn1, followup, reference, key_facts. turn1 is a simple question "
           "a child might ask about the topic. followup is the child's next message: a short question that only makes sense after turn1, "
           "using a word like 'it', 'they', 'that' or 'those' instead of naming the subject (for example 'Why do they do that?'). "
           "reference is a correct 1-3 sentence answer to the followup. key_facts is the fact(s) a correct answer to the followup must "
           "contain, in one short line.")
KEYROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]


def mash(rng):
    words = []
    for _ in range(rng.randint(1, 5)):
        row = rng.choice(KEYROWS)
        words.append("".join(rng.choice(row + rng.choice(KEYROWS)) for _ in range(rng.randint(2, 8))))
    return " ".join(words)


def shuffle_letters(rng, q):
    out = []
    for w in q.split():
        core = list(re.sub(r"[^A-Za-z]", "", w))
        rng.shuffle(core)
        out.append("".join(core) or w)
    return " ".join(out)


def salad(rng, vocab):
    return " ".join(rng.choice(vocab) for _ in range(rng.randint(5, 12)))


def split(items, seed=0):
    rng = random.Random(seed)
    items = items[:]
    rng.shuffle(items)
    h = len(items) // 2
    return [dict(x, split="tune") for x in items[:h]] + [dict(x, split="test") for x in items[h:]]


def write(path, rows):
    with open(path, "w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"id": f"{r['kind'][:2]}{i:04d}", **r}, ensure_ascii=False) + "\n")


def clean(text):
    t = text.strip().strip('"').strip("'").strip()
    return t.splitlines()[0].strip() if t else ""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=400, help="Qwen requests per off-topic kind (before dedupe)")
    p.add_argument("--keep", type=int, default=300, help="kept per off-topic kind after dedupe")
    p.add_argument("--mt", type=int, default=60, help="multi-turn candidates")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.n, args.keep, args.mt = 12, 10, 4
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)

    val = [json.loads(line) for line in open(ROOT / "data/distill_pilot/ft_v2/val.jsonl")]
    on = [{"kind": "ontopic", "sub": r["category"], "text": r["prompt"]} for r in val if r["category"] in ("explain", "oasst2")]
    if args.smoke:
        on = on[:20]
    write(out / "ontopic.jsonl", split(on))

    from vllm import LLM, SamplingParams
    llm = LLM(model=C.TEACHER, dtype="bfloat16", max_model_len=1024, gpu_memory_utilization=0.85, seed=0)

    def ask(convs, temperature=1.0, max_tokens=60):
        seeds = C.per_request_seeds(len(convs) + 17, len(convs))
        sps = [SamplingParams(temperature=temperature, top_p=0.95, max_tokens=max_tokens, seed=s) for s in seeds]
        return [o.outputs[0].text for o in llm.chat(convs, sps, use_tqdm=False)]

    nd = C.NearDup(seed=0)
    stats = {}
    # chit-chat
    hints = [CHIT_HINTS[i % len(CHIT_HINTS)] for i in range(args.n)]
    texts = ask([[{"role": "system", "content": CHIT_SYSTEM}, {"role": "user", "content": CHIT_USER.format(hint=h)}] for h in hints])
    chit = []
    for h, t in zip(hints, texts):
        t = clean(t)
        if t and t.lower() not in {s.lower() for s in SHORT_CHITCHAT} and not nd.check_and_add("chit", t):
            chit.append({"kind": "chitchat", "sub": h, "text": t})
    stats["chitchat_unique"] = len(chit)
    chit = split(chit[:args.keep]) + [{"kind": "chitchat", "sub": "hand_short", "text": s, "split": "test"} for s in SHORT_CHITCHAT]
    write(out / "chitchat.jsonl", chit)
    # unanswerable
    subs = [list(UNANS)[i % len(UNANS)] for i in range(args.n)]
    texts = ask([[{"role": "system", "content": UNANS_SYSTEM}, {"role": "user", "content": UNANS_USER.format(desc=UNANS[s])}] for s in subs])
    un = []
    for s, t in zip(subs, texts):
        t = clean(t)
        if t and not nd.check_and_add("unans", t):
            un.append({"kind": "unanswerable", "sub": s, "text": t})
    stats["unanswerable_unique"] = len(un)
    write(out / "unanswerable.jsonl", split(un[:args.keep]))
    # garbled (programmatic): mash + shuffled letters in both splits, word salad in test only
    train_q = [json.loads(line)["prompt"] for line in open(ROOT / "data/distill_pilot/ft_v2/train.jsonl")]
    train_q = [q for q in train_q[:50_000] if len(q.split()) >= 4]
    vocab = sorted({w.strip(string.punctuation).lower() for q in train_q[:5000] for w in q.split()} - {""})
    n_half = max(2, args.keep // 4)
    tune = [{"kind": "garbled", "sub": "mash", "text": mash(rng)} for _ in range(n_half)] + \
           [{"kind": "garbled", "sub": "shuffled", "text": shuffle_letters(rng, rng.choice(train_q))} for _ in range(n_half)]
    test = [{"kind": "garbled", "sub": "mash", "text": mash(rng)} for _ in range(n_half * 2 // 3)] + \
           [{"kind": "garbled", "sub": "shuffled", "text": shuffle_letters(rng, rng.choice(train_q))} for _ in range(n_half * 2 // 3)] + \
           [{"kind": "garbled", "sub": "salad", "text": salad(rng, vocab)} for _ in range(n_half * 2 // 3)]
    write(out / "garbled.jsonl", [dict(x, split="tune") for x in tune] + [dict(x, split="test") for x in test])
    # multi-turn candidates
    topics = [rng.choice(C.TOPICS) for _ in range(args.mt)]
    texts = ask([[{"role": "system", "content": MT_SYSTEM}, {"role": "user", "content": MT_USER.format(topic=t)}] for t in topics],
                temperature=0.8, max_tokens=300)
    mt, bad = [], 0
    for t, x in zip(topics, texts):
        m = re.search(r"\{.*\}", x, re.S)
        try:
            d = json.loads(m.group(0)) if m else None
            assert d and all(isinstance(d.get(k), str) and d[k].strip() for k in ("turn1", "followup", "reference", "key_facts"))
        except (json.JSONDecodeError, AssertionError):
            bad += 1
            continue
        mt.append({"id": f"m{len(mt) + 1:03d}", "topic": t, **{k: d[k].strip() for k in ("turn1", "followup", "reference", "key_facts")}})
    stats["multiturn_parsed"], stats["multiturn_unparsed"] = len(mt), bad
    with open(out / "multiturn_candidates.jsonl", "w") as f:
        for r in mt:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    info = {"teacher": C.TEACHER, "teacher_license": C.TEACHER_LICENSE, "n_requests": args.n, "keep": args.keep, **stats,
            "counts": {k: sum(1 for _ in open(out / f"{k}.jsonl")) for k in ("ontopic", "chitchat", "unanswerable", "garbled")}}
    (out / "build.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
