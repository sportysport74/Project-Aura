"""
Distillation pilot: sequence-level teacher data for AURA from Qwen2.5-7B-Instruct via vLLM.

Teacher: Qwen/Qwen2.5-7B-Instruct, Apache-2.0 (Copyright 2024 Alibaba Cloud). Outputs may be used to
train other models; credit the teacher in run configs and any dataset or model card.

Categories (prompt source, licence):
  arith    programmatic arithmetic and word problems with computed answers (ours). One in ten
           (op, a, b) triples, chosen by hash, is reserved for the eval battery and never generated.
  gsm8k    GSM8K train questions (MIT); kept only if 'Answer: N' matches the gold answer.
  oasst2   OASST2 English root prompts by humans (Apache-2.0).
  explain  questions written by the teacher itself from a topic list (Qwen outputs, Apache-2.0).
  chat     casual chat messages written by the teacher itself (Qwen outputs, Apache-2.0).
  story    templated story requests: character x situation x setting (ours).

Every reply is written as "Aura" (per-category system prompt with the Aura persona) and goes through:
  truncated (hit max tokens) -> identity (mentions Qwen / Alibaba / "language model" / "as an AI") -> latex -> code
  -> format ([INST] markers) -> answer (math: 'Answer: N' must match; others: no 'Answer:' line)
  -> too_long (conversation > --max-aura-tokens, encoded exactly as finetune_chat.py does)
  -> exact_dup / near_dup (MinHash-LSH over reply word 5-grams, estimated Jaccard >= 0.7, per category).
The near-duplicate rate (exact + near, over replies that passed the earlier filters) is tracked per
category; if any category goes above --max-dup-rate (after --dup-min-count candidates), generation stops.

Output (data/distill_pilot/<run>/): kept.jsonl (with a finetune_chat.py "line" field), rejected.jsonl,
stats.json and run_config.json.

Usage (vLLM venv; WSL has no nvcc, so FlashInfer's sampler must be off):
  VLLM_USE_FLASHINFER_SAMPLER=0 ~/venvs/vllm/bin/python -I scripts/distill/generate_pilot.py --sample 2000
  VLLM_USE_FLASHINFER_SAMPLER=0 ~/venvs/vllm/bin/python -I scripts/distill/generate_pilot.py --target-tokens 50e6
"""
import argparse
import hashlib
import json
import random
import re
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

TEACHER = "Qwen/Qwen2.5-7B-Instruct"
TEACHER_LICENSE = "Apache-2.0 (Copyright 2024 Alibaba Cloud)"

PERSONA = ("You are Aura, a small, friendly AI who lives on a home computer and is still learning. "
           "Use short sentences and everyday words. No jargon: if you need a special word, explain it simply. "
           "Write plain text only: no LaTeX, no code, no markdown headings. "
           "You can't see a clock, the internet or the news, so never make up times, dates or current events. "
           "Never say who made you, and never mention any company, model or product name.")
FINAL = "then finish with a last line 'Answer: <number>' with just the number."
SYSTEM = {
    "arith": f"{PERSONA} Solve the problem. Write the calculation as an equation (for example: 252 - 59 = 193), {FINAL}",
    "gsm8k": f"{PERSONA} Solve the problem in a few short steps, writing each calculation as a plain equation, {FINAL}",
    "story": f"{PERSONA} Tell a short, gentle story for a young child in at most 100 words. No title, no 'Answer:' line.",
    "explain": f"{PERSONA} Answer helpfully and simply in at most 100 words. No 'Answer:' line.",
    "oasst2": f"{PERSONA} Answer helpfully and simply in at most 100 words. No 'Answer:' line.",
    "chat": f"{PERSONA} Reply like a warm, friendly companion in one to three short sentences. No 'Answer:' line.",
}
MATH = {"arith", "gsm8k"}
LATEX = re.compile(r"\\\(|\\\[|\\frac|\\times|\\cdot|\$\$")
CODE = re.compile(r"```|^\s*\$ |^\s*sudo\s", re.M)  # code fences and shell prompts, not every line starting "cd"
# OASST2 prompts that need code, a terminal or real-time facts are skipped (out of scope for AURA)
TECH_PROMPT = re.compile(r"\b(code|coding|program|python|javascript|java|c\+\+|rust|sql|regex|html|css|linux|ubuntu|arch|"
                         r"windows|terminal|command|install|compile|api|docker|git|excel|function|script|bash|server|"
                         r"database|json|current time|today'?s date|what time is it|latest news|stock price|weather today)\b",
                         re.I)

# Share of AURA tokens per category in a full run (GSM8K and OASST2 are capped by their pool sizes:
# one kept reply per prompt, since several replies to one prompt are near-duplicates of each other).
SHARES = {"arith": 0.25, "explain": 0.33, "chat": 0.16, "story": 0.16, "gsm8k": 0.06, "oasst2": 0.04}
AVG_AURA_TOKENS = 150  # measured on the vLLM benchmark (prompt + reply)

IDENTITY = re.compile(r"\b(qwen|alibaba)\b|language model|\bas an ai\b", re.I)
ANSWER = re.compile(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)")

# --------------------------------------------------------------------------- prompt pools

NAMES = ["Mia", "Leo", "Ava", "Sam", "Zoe", "Ben", "Lily", "Max", "Ella", "Noah", "Ruby", "Omar", "Ivy", "Jack", "Nora",
         "Eli", "Maya", "Finn", "Rosa", "Theo", "Lena", "Kai", "Iris", "Jude", "Anna", "Ravi", "Sofia", "Tom", "Yuki", "Ada"]
THINGS = ["apples", "marbles", "stickers", "cookies", "books", "shells", "pencils", "coins", "balloons", "stamps",
          "cards", "flowers", "eggs", "blocks", "toy cars", "beads", "candles", "buttons", "leaves", "rocks"]


def heldout(op, a, b):
    """One in ten (op, a, b) triples is reserved for the arithmetic eval battery."""
    return int(hashlib.sha1(f"{op}|{a}|{b}".encode()).hexdigest(), 16) % 10 == 0


def rand_num(rng):
    d = rng.choices([1, 2, 3], weights=[2, 5, 3])[0]
    return rng.randint(1 if d == 1 else 10 ** (d - 1), 10 ** d - 1)


def arith_prompt(rng):
    while True:
        op = rng.choice("+-*/")
        a, b = rand_num(rng), rand_num(rng)
        if op == "-" and b > a:
            a, b = b, a
        if op == "*":
            b = rng.randint(2, 12) if a > 99 else b
        if op == "/":
            b = rng.randint(2, 12)
            a = b * rng.randint(2, 99)
        if heldout(op, a, b):
            continue
        gold = {"+": a + b, "-": a - b, "*": a * b, "/": a // b}[op]
        name, thing = rng.choice(NAMES), rng.choice(THINGS)
        sym = {"+": "+", "-": "-", "*": "×", "/": "÷"}[op]
        if rng.random() < 0.5:
            text = rng.choice([f"What is {a} {sym} {b}?", f"Can you work out {a} {sym} {b}?", f"{a} {sym} {b} = ?"])
        else:
            text = {
                "+": f"{name} has {a} {thing} and gets {b} more. How many {thing} does {name} have now?",
                "-": f"{name} had {a} {thing} and gave away {b}. How many {thing} are left?",
                "*": f"{name} has {a} boxes with {b} {thing} in each box. How many {thing} is that in total?",
                "/": f"{name} shares {a} {thing} equally among {b} friends. How many {thing} does each friend get?",
            }[op]
        return {"category": "arith", "source": "programmatic", "license": "ours", "prompt": text,
                "gold": str(gold), "meta": f"{op}|{a}|{b}"}


CHARACTERS = ["a shy little fox", "a curious robot", "a brave mouse", "a sleepy bear", "a tiny dragon", "a kind old owl",
              "a lost kitten", "a clumsy penguin", "a cheerful snail", "a young girl named Pia", "a boy named Tomas",
              "a talking tree", "a lonely cloud", "a busy bee", "a little red boat", "a grumpy goat", "a happy puppy",
              "a wise turtle", "a small star", "a playful otter", "a baby elephant", "a friendly ghost", "a lighthouse keeper",
              "a baker's cat", "a paper kite", "a rabbit who loves carrots", "a frog who cannot jump", "a snowman",
              "a little ladybug", "a quiet giraffe", "a hedgehog with a cold", "a squirrel who hides acorns",
              "a duckling", "a toy soldier", "a moth who loves lamps", "a whale who sings", "a farmer's dog",
              "a seagull", "a caterpillar", "a pair of old shoes"]
SITUATIONS = ["wants to make a new friend", "is afraid of the dark", "loses something important", "learns to share",
              "finds a mysterious map", "tries something new", "helps someone in trouble", "has a bad day",
              "goes on a big adventure", "wants to win a race", "learns to be patient", "makes a mistake and fixes it",
              "plans a surprise party", "gets lost on the way home", "discovers a hidden talent", "waits for spring",
              "cannot fall asleep", "builds something special", "says sorry to a friend", "learns to ask for help",
              "finds a lost baby animal", "is nervous about the first day of school", "wants to touch the moon",
              "learns why rain is good", "keeps a secret", "learns to tell the truth", "visits grandma",
              "goes camping for the first time", "plants a seed", "teaches a friend a game"]
SETTINGS = ["in a deep forest", "by the sea", "in a busy city", "on a snowy mountain", "in a small village",
            "on a farm", "under the sea", "in a garden", "on a rainy afternoon", "at night under the stars",
            "in a library", "at a school", "in the desert", "on a little island", "in a castle", "by a river",
            "in a toy shop", "in a meadow full of flowers", "on a train", "in the jungle", "in a cozy kitchen",
            "on a windy beach", "in a park", "in the clouds", "at a fair"]
STORY_ASKS = ["Tell me a story about {c} who {s} {p}.", "Can you tell a short story about {c} who {s} {p}?",
              "Write a bedtime story about {c} who {s} {p}.", "I'd love a story about {c} who {s} {p}."]

TOPICS = [
    # nature and science
    "why the sky is blue", "rainbows", "volcanoes", "earthquakes", "the moon", "the sun", "stars", "planets", "gravity",
    "magnets", "electricity", "light", "sound", "rain", "snow", "clouds", "wind", "thunder and lightning", "oceans",
    "rivers", "mountains", "deserts", "rainforests", "seasons", "plants", "seeds", "trees", "flowers", "bees",
    "ants", "butterflies", "birds", "fish", "whales", "dinosaurs", "frogs", "cats", "dogs", "horses", "insects",
    "the human body", "the heart", "bones", "teeth", "sleep", "germs", "food and energy", "water", "air", "fire",
    "rocks", "fossils", "the water cycle", "recycling", "the weather", "ice", "shadows", "mirrors", "bubbles",
    # everyday life and how-to
    "cooking pasta", "baking bread", "making a sandwich", "tying shoelaces", "brushing teeth", "saving money",
    "planting a garden", "caring for a pet", "riding a bike", "learning to swim", "keeping a room tidy",
    "packing for a trip", "making friends", "being kind", "saying sorry", "handling anger", "feeling nervous",
    "staying healthy", "drinking enough water", "reading books", "writing a letter", "telling time", "using a map",
    "crossing the road safely", "first aid for a small cut", "staying warm in winter", "keeping cool in summer",
    # school subjects and ideas
    "fractions", "multiplication", "telling odd and even numbers apart", "shapes", "measuring length", "money and change",
    "clocks and hours", "maps and directions", "the alphabet", "rhyming words", "verbs and nouns", "spelling tips",
    "the history of writing", "how books are made", "inventions", "the wheel", "computers", "the internet",
    "how phones work", "robots", "trains", "airplanes", "boats", "cars", "bridges", "castles", "pyramids",
    "music", "drawing", "colors", "painting", "dancing", "football", "chess", "puzzles", "riddles", "jokes",
    "holidays around the world", "different languages", "farms", "bakeries", "libraries", "hospitals", "firefighters",
]
ASK_KINDS = ["a simple question", "a 'how does it work' question", "a 'why' question", "a request for a short explanation",
             "a request for practical tips"]
CHAT_KINDS = ["a casual message about their day", "a message sharing how they feel", "a friendly greeting with a small question",
              "a message asking for advice about a small everyday problem", "a message telling something they are excited about"]
ASKERS = ["a curious eight-year-old", "a teenager", "a busy parent", "a grandparent", "a student", "an adult learning something new"]
PROMPT_WRITER = ("You write realistic messages that people send to a friendly AI helper named Aura. "
                 "Output only the message itself, with no quotes and no explanation.")


def story_prompt(rng):
    c, s, p = rng.choice(CHARACTERS), rng.choice(SITUATIONS), rng.choice(SETTINGS)
    return {"category": "story", "source": "template", "license": "ours",
            "prompt": rng.choice(STORY_ASKS).format(c=c, s=s, p=p), "meta": f"{c}|{s}|{p}"}


def writer_request(rng, kind):
    """Stage-1 request: the teacher writes a user message for the 'explain' or 'chat' category."""
    asker, topic = rng.choice(ASKERS), rng.choice(TOPICS)
    style = rng.choice(ASK_KINDS if kind == "explain" else CHAT_KINDS)
    msg = f"Write {style} that {asker} might send to Aura, about {topic}. Keep it under 30 words."
    return {"kind": kind, "topic": topic, "conv": [{"role": "system", "content": PROMPT_WRITER},
                                                   {"role": "user", "content": msg}]}


class AuraLength:
    """Token count of a one-turn conversation exactly as finetune_chat.encode_chat builds it:
    'You:' + user + 'Aura:' + reply + 'You:', each piece encoded separately. Uses SentencePiece directly so
    this script doesn't import the cosf_mi package (its __init__ needs scipy, which the vLLM venv lacks)."""

    def __init__(self, model_path):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))
        self.user, self.aura = len(self.sp.encode("You:")), len(self.sp.encode("Aura:"))

    def __call__(self, prompt, reply):
        return 2 * self.user + self.aura + len(self.sp.encode(prompt)) + len(self.sp.encode(reply))


def clean_written_prompt(text):
    t = text.strip().split("\n")[0].strip().strip('"“”').strip()
    t = re.sub(r"^(message|user|question)\s*:\s*", "", t, flags=re.I)
    return t if 8 <= len(t) <= 220 and "aura:" not in t.lower() else None


# --------------------------------------------------------------------------- near-duplicate detection

class NearDup:
    """MinHash-LSH over word 5-gram shingles; per category. Near-dup = estimated Jaccard >= threshold."""
    P = (1 << 31) - 1

    def __init__(self, n_perm=128, bands=16, threshold=0.7, seed=0):
        r = np.random.default_rng(seed)
        self.a = r.integers(1, self.P, n_perm, dtype=np.uint64)
        self.b = r.integers(0, self.P, n_perm, dtype=np.uint64)
        self.bands, self.rows, self.threshold = bands, n_perm // bands, threshold
        self.buckets = defaultdict(lambda: defaultdict(list))  # category -> band key -> [signature index]
        self.sigs = defaultdict(list)
        self.exact = defaultdict(set)

    @staticmethod
    def norm(text):
        return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split()

    def signature(self, words):
        grams = {" ".join(words[i:i + 5]) for i in range(max(1, len(words) - 4))}
        h = np.array([int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "little") % self.P
                      for g in grams], dtype=np.uint64)
        return ((np.outer(h, self.a) + self.b) % self.P).min(axis=0)

    def check_and_add(self, category, text):
        """Return 'exact_dup', 'near_dup' or None (and remember the text if it's new)."""
        words = self.norm(text)
        key = " ".join(words)
        if key in self.exact[category]:
            return "exact_dup"
        sig = self.signature(words)
        keys = [(i, sig[i * self.rows:(i + 1) * self.rows].tobytes()) for i in range(self.bands)]
        seen = set()
        for k in keys:
            for j in self.buckets[category].get(k, ()):
                if j not in seen:
                    seen.add(j)
                    if (self.sigs[category][j] == sig).mean() >= self.threshold:
                        return "near_dup"
        idx = len(self.sigs[category])
        self.sigs[category].append(sig)
        self.exact[category].add(key)
        for k in keys:
            self.buckets[category][k].append(idx)
        return None


# --------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", type=int, default=0, help="small run with ~N candidates in total; prints 20 random kept samples")
    p.add_argument("--target-tokens", type=float, default=50e6, help="full run: AURA tokens to keep (prompt + reply)")
    p.add_argument("--out", default=None, help="output folder (default data/distill_pilot/sample or .../full)")
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
    out = Path(args.out) if args.out else ROOT / "data/distill_pilot" / ("sample" if args.sample else "full")
    out.mkdir(parents=True, exist_ok=True)
    aura_len = AuraLength(ROOT / "models/tokenizer_mix16k.model")

    # How many candidates per category
    total = args.sample if args.sample else int(args.target_tokens / AVG_AURA_TOKENS * 1.25)  # +25% for rejects
    gsm = [json.loads(line) for line in open(ROOT / "data/distill_pilot/src_gsm8k.jsonl")]
    oas_all = [json.loads(line) for line in open(ROOT / "data/distill_pilot/src_oasst2.jsonl")]
    oas = [o for o in oas_all if not TECH_PROMPT.search(o["prompt"])]
    print(f"OASST2: {len(oas):,} of {len(oas_all):,} prompts kept after dropping code/tech/real-time ones", flush=True)
    want = {c: max(1, round(total * s)) for c, s in SHARES.items()}
    want["gsm8k"] = min(want["gsm8k"], len(gsm))
    want["oasst2"] = min(want["oasst2"], len(oas))
    rng.shuffle(gsm)
    rng.shuffle(oas)

    run_config = {
        "teacher": TEACHER, "teacher_license": TEACHER_LICENSE, "method": "sequence-level (text) distillation",
        "system_prompts": SYSTEM, "prompt_writer": PROMPT_WRITER, "temperature": args.temperature, "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens, "max_aura_tokens": args.max_aura_tokens, "seed": args.seed,
        "candidates_per_category": want, "max_dup_rate": args.max_dup_rate,
        "prompt_licences": {"arith": "ours", "story": "ours", "explain": "Qwen output (Apache-2.0)",
                            "chat": "Qwen output (Apache-2.0)", "gsm8k": "MIT", "oasst2": "Apache-2.0"},
        "git_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                     text=True).stdout.strip(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / "run_config.json").write_text(json.dumps(run_config, indent=1))
    print(f"Candidates per category: {want} (total {sum(want.values()):,})", flush=True)

    llm = LLM(model=TEACHER, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.85, seed=args.seed)
    t_start = time.time()

    # Stage 1: the teacher writes the 'explain' and 'chat' prompts (over-generate 15% for duplicates and junk)
    pools = {"explain": [], "chat": []}
    seen_prompts = set()
    sp1 = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=60, seed=args.seed)
    for kind in ("explain", "chat"):
        reqs = [writer_request(rng, kind) for _ in range(int(want[kind] * 1.15) + 10)]
        for i in range(0, len(reqs), args.chunk):
            part = reqs[i:i + args.chunk]
            for r, o in zip(part, llm.chat([r["conv"] for r in part], sp1, use_tqdm=False)):
                t = clean_written_prompt(o.outputs[0].text)
                k = " ".join(NearDup.norm(t)) if t else None
                if t and k not in seen_prompts:
                    seen_prompts.add(k)
                    pools[kind].append({"category": kind, "source": "qwen-written", "license": "Apache-2.0 (Qwen output)",
                                        "prompt": t, "meta": r["topic"]})
        print(f"stage 1 {kind}: {len(pools[kind]):,} unique prompts from {len(reqs):,} requests", flush=True)

    # All stage-2 requests, interleaved so every chunk mixes categories
    requests = ([arith_prompt(rng) for _ in range(want["arith"])] + [story_prompt(rng) for _ in range(want["story"])]
                + pools["explain"][:want["explain"]] + pools["chat"][:want["chat"]]
                + [dict(g, category="gsm8k") for g in gsm[:want["gsm8k"]]]
                + [dict(o, category="oasst2") for o in oas[:want["oasst2"]]])
    rng.shuffle(requests)

    nd = NearDup(seed=args.seed)
    stats = defaultdict(Counter)
    kept_n = Counter()
    kept_tokens = Counter()
    sp2 = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_new_tokens, seed=args.seed)
    stopped = None
    with open(out / "kept.jsonl", "w") as fk, open(out / "rejected.jsonl", "w") as fr:
        for i in range(0, len(requests), args.chunk):
            part = requests[i:i + args.chunk]
            convs = [[{"role": "system", "content": SYSTEM[r["category"]]}, {"role": "user", "content": r["prompt"]}]
                     for r in part]
            for r, o in zip(part, llm.chat(convs, sp2, use_tqdm=False)):
                c, out0 = r["category"], o.outputs[0]
                reply = out0.text.strip()
                stats[c]["generated"] += 1
                reason = None
                if out0.finish_reason != "stop":
                    reason = "truncated"
                elif IDENTITY.search(reply):
                    reason = "identity"
                elif "INST]" in reply or "INST]" in r["prompt"]:
                    reason = "format"
                elif LATEX.search(reply):
                    reason = "latex"
                elif CODE.search(reply):
                    reason = "code"
                elif c in MATH:
                    m = ANSWER.findall(reply)
                    try:
                        ok = bool(m) and float(m[-1].replace(",", "").rstrip(".")) == float(r["gold"])
                    except ValueError:
                        ok = False
                    reason = None if ok else "wrong_answer"
                elif "Answer:" in reply:
                    reason = "answer_leak"
                if reason is None:
                    n_tok = aura_len(r["prompt"], reply)
                    if n_tok > args.max_aura_tokens:
                        reason = "too_long"
                if reason is None:
                    stats[c]["dup_checked"] += 1
                    reason = nd.check_and_add(c, reply)
                if reason:
                    stats[c][reason] += 1
                    fr.write(json.dumps({"category": c, "reason": reason, "prompt": r["prompt"], "reply": reply}) + "\n")
                    continue
                kept_n[c] += 1
                kept_tokens[c] += n_tok
                fk.write(json.dumps({"category": c, "source": r["source"], "prompt_license": r["license"],
                                     "teacher": TEACHER, "teacher_license": TEACHER_LICENSE, "aura_tokens": n_tok,
                                     "prompt": r["prompt"], "reply": reply,
                                     "line": f"[INST] {r['prompt']} [/INST] {reply}"}) + "\n")
            fk.flush()
            fr.flush()
            done = min(i + args.chunk, len(requests))
            el = time.time() - t_start
            dup = {c: (s["exact_dup"] + s["near_dup"]) / s["dup_checked"] for c, s in stats.items() if s["dup_checked"]}
            print(f"[{done:,}/{len(requests):,} requests, {el / 60:.1f} min] kept {sum(kept_n.values()):,} samples, "
                  f"{sum(kept_tokens.values()) / 1e6:.2f}M AURA tokens; dup rate "
                  + " ".join(f"{c} {v:.1%}" for c, v in sorted(dup.items())), flush=True)
            over = [c for c, v in dup.items() if v > args.max_dup_rate and stats[c]["dup_checked"] >= args.dup_min_count]
            if over:
                stopped = f"near-duplicate rate above {args.max_dup_rate:.0%} in {over} after {done:,} requests"
                print("STOPPING: " + stopped, flush=True)
                break

    summary = {"stopped_early": stopped, "minutes": round((time.time() - t_start) / 60, 1),
               "kept_samples": dict(kept_n), "kept_aura_tokens": dict(kept_tokens),
               "per_category": {c: dict(s) for c, s in stats.items()}}
    for c, s in stats.items():
        g = s["generated"]
        summary["per_category"][c].update({
            "identity_reject_rate": round(s["identity"] / g, 4) if g else 0,
            "dup_rate": round((s["exact_dup"] + s["near_dup"]) / s["dup_checked"], 4) if s["dup_checked"] else 0,
            "keep_rate": round(kept_n[c] / g, 4) if g else 0})
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
