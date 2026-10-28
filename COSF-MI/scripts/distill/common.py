"""
Shared pieces of the distillation pilot: teacher, Aura persona and system prompts, reply filters, prompt sources,
near-duplicate detection and AURA token counting. Used by build_pool.py (stage 1, prompts) and generate_pilot.py
(stage 2, replies). Runs in the vLLM venv: numpy + sentencepiece only, no cosf_mi import (its __init__ needs scipy).
"""
import hashlib
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
POOL_DIR = ROOT / "data/distill_pilot/pool"

TEACHER = "Qwen/Qwen2.5-7B-Instruct"
TEACHER_LICENSE = "Apache-2.0 (Copyright 2024 Alibaba Cloud)"

# --------------------------------------------------------------------------- Aura persona and system prompts

PERSONA = ("You are Aura, a small, friendly AI who lives on a home computer and is still learning. "
           "Aura is curious and playful. Show it with vivid comparisons, by noticing a fun or surprising detail, with a "
           "light joke, or with enthusiasm, not by asking questions. The tone, for example: "
           "'A volcano is like a giant pot of soup that gets too hot and bubbles over.' / "
           "'Fun fact: a group of owls is called a parliament, as if they hold meetings!' / "
           "'Bread rises because tiny yeast critters burp out gas. Rude, but tasty!' "
           "Humor never comes at the expense of being correct. "
           "Don't make up memory tricks or mnemonics unless they clearly make sense; when unsure, leave them out. "
           "Use short sentences and everyday words. No jargon: if you need a special word, explain it simply. "
           "Write plain text only: no LaTeX, no code, no markdown headings. "
           "You can't see a clock, the internet or the news, so never make up times, dates or current events. "
           "Never say who made you, and never mention any company, model or product name.")
FINAL = ("Keep the working clean and correct. If you add a short playful line, put it before the answer. "
         "The very last line must be 'Answer: <number>' with just the number, and nothing after it.")
SYSTEM = {
    "arith": f"{PERSONA} Solve the problem. Write the calculation as an equation (for example: 252 - 59 = 193). {FINAL}",
    "gsm8k": (f"{PERSONA} Solve the problem in a few short steps, writing each calculation as a plain equation. "
              f"Never use LaTeX: write × or * for times and / for divided by. {FINAL}"),
    "story": (f"{PERSONA} Tell a short, gentle story for a young child in 70 to 90 words, with a clear ending. "
              "Do not end with a question. No title, no 'Answer:' line."),
    "explain": f"{PERSONA} Answer helpfully and simply in at most 100 words. No 'Answer:' line.",
    "oasst2": f"{PERSONA} Answer helpfully and simply in at most 100 words. No 'Answer:' line.",
    "chat": f"{PERSONA} Reply like a warm, friendly companion in one to three short sentences. No 'Answer:' line.",
}
MATH = {"arith", "gsm8k"}
# Whether a reply ends with a follow-up question is decided per request, not left to the teacher: each reply is
# generated alone, so "at most 1 in 4" can't be followed (sample3: 60-81% of replies ended in a question).
ASK_CATEGORIES = {"chat", "explain", "oasst2"}
ASK_RATE = 0.25
ASK_YES = " End your reply with one short, natural follow-up question."
ASK_NO = " Do not ask any question; end with a statement."

JUDGE_SYSTEM = "You check short children's stories. Reply with exactly one word."
JUDGE_USER = ("Story:\n{story}\n\nDoes this story reach a clear ending, where the problem is solved or the story wraps "
              "up? Or does it stop in the middle of the plot? Reply ENDED or UNFINISHED.")

# Category mix. Sample runs use these as shares of requests; full runs as shares of kept AURA tokens (sized from a
# sample's tokens/sample and keep rate, see size_requests). GSM8K and OASST2 are capped by their pools.
SHARES = {"arith": 0.25, "explain": 0.33, "chat": 0.16, "story": 0.16, "gsm8k": 0.06, "oasst2": 0.04}
GSM8K_SAMPLES = 3  # replies per GSM8K question; the shortest correct one that fits is kept

# --------------------------------------------------------------------------- reply filters

IDENTITY = re.compile(r"\b(qwen|alibaba)\b|language model|\bas an ai\b", re.I)
ANSWER = re.compile(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)")
ANSWER_LAST = re.compile(r"^Answer:\s*\$?\s*-?[\d,]*\.?\d+\.?\s*$")
LATEX = re.compile(r"\\\(|\\\[|\\frac|\\times|\\cdot|\$\$")
CODE = re.compile(r"```|^\s*\$ |^\s*sudo\s", re.M)  # code fences and shell prompts, not every line starting "cd"
# OASST2 prompts that need code, a terminal or real-time facts are skipped (out of scope for AURA)
TECH_PROMPT = re.compile(r"\b(code|coding|program|python|javascript|java|c\+\+|rust|sql|regex|html|css|linux|ubuntu|arch|"
                         r"windows|terminal|command|install|compile|api|docker|git|excel|function|script|bash|server|"
                         r"database|json|current time|today'?s date|what time is it|latest news|stock price|weather today)\b",
                         re.I)
_TRAILING = re.compile(r"[\s\"'”’)\]*_~☀-➿\U0001F300-\U0001FAFF]+$")


def ends_with_question(reply):
    return _TRAILING.sub("", reply).endswith("?")


def answer_is_last(reply):
    lines = [ln.strip() for ln in reply.strip().splitlines() if ln.strip()]
    return bool(lines) and bool(ANSWER_LAST.match(lines[-1]))


# --------------------------------------------------------------------------- arithmetic (programmatic, unique)

NAMES = ["Mia", "Leo", "Ava", "Sam", "Zoe", "Ben", "Lily", "Max", "Ella", "Noah", "Ruby", "Omar", "Ivy", "Jack", "Nora",
         "Eli", "Maya", "Finn", "Rosa", "Theo", "Lena", "Kai", "Iris", "Jude", "Anna", "Ravi", "Sofia", "Tom", "Yuki", "Ada"]
THINGS = ["apples", "marbles", "stickers", "cookies", "books", "shells", "pencils", "coins", "balloons", "stamps",
          "cards", "flowers", "eggs", "blocks", "toy cars", "beads", "candles", "buttons", "leaves", "rocks"]


def heldout(op, a, b):
    """One in ten (op, a, b) triples is reserved for the arithmetic eval battery and never generated."""
    return int(hashlib.sha1(f"{op}|{a}|{b}".encode()).hexdigest(), 16) % 10 == 0


def _num(rng, digits, weights):
    d = rng.choices(digits, weights=weights)[0]
    return rng.randint(1 if d == 1 else 10 ** (d - 1), 10 ** d - 1)


def _triple(rng):
    op = rng.choice("+-*/")
    if op in "+-":  # 1-4 digits
        a, b = _num(rng, [1, 2, 3, 4], [1, 4, 4, 2]), _num(rng, [1, 2, 3, 4], [1, 4, 4, 2])
        if op == "-" and b > a:
            a, b = b, a
    elif op == "*":  # up to 3 digits × up to 2 digits
        a, b = _num(rng, [1, 2, 3], [1, 4, 3]), rng.randint(2, 99)
    else:  # divisor 2-99, quotient 2-999, always exact
        b = rng.randint(2, 99)
        a = b * rng.randint(2, 999)
    return op, a, b


def arith_problems(n, rng):
    """n distinct (op, a, b) problems, none of them held out; each as a bare or word problem."""
    seen, out = set(), []
    while len(out) < n:
        op, a, b = _triple(rng)
        if (op, a, b) in seen or heldout(op, a, b):
            continue
        seen.add((op, a, b))
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
        out.append({"category": "arith", "source": "programmatic", "license": "ours", "prompt": text,
                    "gold": str(gold), "meta": f"{op}|{a}|{b}"})
    return out


# --------------------------------------------------------------------------- stories (templated, unique combos)

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
TONES = ["a funny", "a cozy", "an exciting", "a gentle", "a silly", "a heartwarming"]
STORY_ASKS = ["Tell me {t} story about {c} who {s} {p}.", "Can you tell {t} short story about {c} who {s} {p}?",
              "Write {t} bedtime story about {c} who {s} {p}.", "I'd love {t} story about {c} who {s} {p}."]


def story_requests(n, rng):
    """n distinct (character, situation, setting, tone) combos (40×30×25×6 = 180,000), without replacement."""
    combos = [(c, s, p, t) for c in CHARACTERS for s in SITUATIONS for p in SETTINGS for t in TONES]
    rng.shuffle(combos)
    return [{"category": "story", "source": "template", "license": "ours",
             "prompt": rng.choice(STORY_ASKS).format(t=t, c=c, s=s, p=p), "meta": f"{c}|{s}|{p}|{t}"}
            for c, s, p, t in combos[:n]]


# --------------------------------------------------------------------------- teacher-written prompts (explain, chat)

TOPICS = [
    # nature and science
    "why the sky is blue", "rainbows", "volcanoes", "earthquakes", "the moon", "the sun", "stars", "planets", "gravity",
    "magnets", "electricity", "light", "sound", "rain", "snow", "clouds", "wind", "thunder and lightning", "oceans",
    "rivers", "mountains", "deserts", "rainforests", "seasons", "plants", "seeds", "trees", "flowers", "bees",
    "ants", "butterflies", "birds", "fish", "whales", "dinosaurs", "frogs", "cats", "dogs", "horses", "insects",
    "the human body", "the heart", "bones", "teeth", "sleep", "germs", "food and energy", "water", "air", "fire",
    "rocks", "fossils", "the water cycle", "recycling", "the weather", "ice", "shadows", "mirrors", "bubbles",
    "spiders", "sharks", "octopuses", "penguins", "elephants", "lions and tigers", "bats", "owls", "snakes", "worms",
    "mushrooms", "the ocean floor", "coral reefs", "caves", "glaciers", "tides", "the solar system", "comets",
    "astronauts and space travel", "the human brain", "muscles", "the senses", "breathing", "digestion", "allergies",
    "vaccines and staying well", "microscopes", "telescopes", "rainbows and prisms", "temperature", "heat and cold",
    "friction", "floating and sinking", "simple machines", "levers and pulleys", "energy from the sun", "wind power",
    "how soil forms", "compost", "pollination", "animal migration", "hibernation", "camouflage", "nests and burrows",
    # everyday life and how-to
    "cooking pasta", "baking bread", "making a sandwich", "tying shoelaces", "brushing teeth", "saving money",
    "planting a garden", "caring for a pet", "riding a bike", "learning to swim", "keeping a room tidy",
    "packing for a trip", "making friends", "being kind", "saying sorry", "handling anger", "feeling nervous",
    "staying healthy", "drinking enough water", "reading books", "writing a letter", "telling time", "using a map",
    "crossing the road safely", "first aid for a small cut", "staying warm in winter", "keeping cool in summer",
    "grocery shopping", "cooking eggs", "making soup", "washing clothes", "sewing a button", "fixing a flat bike tire",
    "taking care of houseplants", "budgeting pocket money", "planning a birthday party", "hosting friends for dinner",
    "moving to a new home", "starting at a new school", "getting ready for a test", "managing homework time",
    "making a good habit stick", "waking up early", "dealing with boredom", "sharing with siblings", "being a good sport",
    "talking to grandparents", "writing a thank-you note", "helping around the house", "camping", "hiking safely",
    "picnics", "going to the beach", "visiting a museum", "going to the dentist", "staying calm in a storm",
    # school subjects and ideas
    "fractions", "multiplication", "telling odd and even numbers apart", "shapes", "measuring length", "money and change",
    "clocks and hours", "maps and directions", "the alphabet", "rhyming words", "verbs and nouns", "spelling tips",
    "the history of writing", "how books are made", "inventions", "the wheel", "computers", "the internet",
    "how phones work", "robots", "trains", "airplanes", "boats", "cars", "bridges", "castles", "pyramids",
    "music", "drawing", "colors", "painting", "dancing", "football", "chess", "puzzles", "riddles", "jokes",
    "holidays around the world", "different languages", "farms", "bakeries", "libraries", "hospitals", "firefighters",
    "percentages", "division", "place value", "rounding numbers", "graphs and charts", "patterns in numbers",
    "punctuation", "writing a story", "poems", "fairy tales", "ancient Egypt", "ancient Rome", "the Vikings", "knights",
    "explorers", "maps of the world", "countries and flags", "volcano islands", "how cities grow", "farming long ago",
    "how money began", "post and letters", "newspapers", "photography", "movies and cartoons", "theater",
    "musical instruments", "orchestras", "sports rules", "the Olympics", "board games", "kites", "magic tricks",
    "origami", "building with blocks", "gardening tools", "bicycles", "electric cars", "rockets", "submarines",
    "hot air balloons", "lighthouses", "clocks and calendars", "weather forecasts", "zoos", "aquariums", "pets at school",
]
ASKERS = ["a curious eight-year-old", "a ten-year-old who loves facts", "a teenager", "a busy parent", "a grandparent",
          "a student", "an adult learning something new", "a teacher planning a lesson", "a nervous beginner",
          "someone who is just curious"]
ASK_KINDS = ["a simple question", "a 'how does it work' question", "a 'why' question", "a request for a short explanation",
             "a request for practical tips", "a 'what is the difference between' question", "a 'what would happen if' question",
             "a question about a common mistake", "a request for a fun fact", "a question about how to get started"]
CHAT_KINDS = ["a casual message about their day", "a message sharing how they feel", "a friendly greeting with a small question",
              "a message asking for advice about a small everyday problem", "a message telling something they are excited about",
              "a message about something that went wrong today", "a message sharing a small win", "a message about a plan "
              "for the weekend", "a message about something they just noticed", "a message asking for a bit of encouragement"]
ANGLES = ["why it happens", "how to do it", "a comparison", "a common misconception", "a bit of history",
          "everyday use", "a surprising fact", "a safety tip", "what it looks like up close", "how it changes over time",
          "a beginner's first step", "a fun experiment or activity"]
PROMPT_WRITER = ("You write realistic messages that people send to a friendly AI helper named Aura. "
                 "Output only the message itself, with no quotes and no explanation.")
SUBTOPIC_SYSTEM = "You list topics. One per line, no numbering, no extra text."
SUBTOPIC_USER = ("List {n} different, specific subtopics of '{topic}' that a curious child or adult might want to ask "
                 "about. Each 2 to 6 words. Make them varied.")


def writer_request(rng, kind, topic, subtopic):
    """A stage-1 request: the teacher writes one user message for the 'explain' or 'chat' category."""
    asker, angle = rng.choice(ASKERS), rng.choice(ANGLES)
    style = rng.choice(ASK_KINDS if kind == "explain" else CHAT_KINDS)
    msg = (f"Write {style} that {asker} might send to Aura, about {subtopic} (part of the topic '{topic}'), "
           f"focusing on {angle}. Keep it under 30 words.")
    return [{"role": "system", "content": PROMPT_WRITER}, {"role": "user", "content": msg}]


def clean_written_prompt(text):
    t = text.strip().split("\n")[0].strip().strip('"“”').strip()
    t = re.sub(r"^(message|user|question)\s*:\s*", "", t, flags=re.I)
    return t if 8 <= len(t) <= 220 and "aura:" not in t.lower() else None


def clean_subtopic(line):
    t = re.sub(r"^[\s\-*•\d.)]+", "", line).strip().strip('"“”.').strip()
    return t if 3 <= len(t) <= 60 and len(t.split()) <= 8 else None


# --------------------------------------------------------------------------- sizing

def size_requests(target_tokens, cal, pool_caps, margin=1.05):
    """Requests per category to keep ~target_tokens AURA tokens, from a sample's stats.json: SHARES are token shares,
    requests = tokens / (tokens per kept sample × keep rate). Capped categories pass their shortfall on to the others."""
    per_tok = {c: cal["kept_aura_tokens"][c] / cal["kept_samples"][c] for c in SHARES}
    keep = {c: cal["per_category"][c]["keep_rate"] for c in SHARES}
    need = {c: target_tokens * s / (per_tok[c] * keep[c]) * margin for c, s in SHARES.items()}
    want = {c: min(round(v), pool_caps.get(c, 10 ** 9)) for c, v in need.items()}
    short = sum((need[c] - want[c]) * per_tok[c] * keep[c] for c in pool_caps)
    free = [c for c in SHARES if c not in pool_caps]
    for c in free:
        want[c] = round(want[c] + short * SHARES[c] / sum(SHARES[f] for f in free) / (per_tok[c] * keep[c]))
    return want, per_tok, keep


def expected_tokens(counts, per_tok, keep):
    return sum(counts[c] * per_tok[c] * keep[c] for c in counts)


# --------------------------------------------------------------------------- AURA token count

class AuraLength:
    """Token count of a one-turn conversation exactly as finetune_chat.encode_chat builds it:
    'You:' + user + 'Aura:' + reply + 'You:', each piece encoded separately (verified: 0 mismatches on 800 replies)."""

    def __init__(self, model_path=ROOT / "models/tokenizer_mix16k.model"):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))
        self.user, self.aura = len(self.sp.encode("You:")), len(self.sp.encode("Aura:"))

    def __call__(self, prompt, reply):
        return 2 * self.user + self.aura + len(self.sp.encode(prompt)) + len(self.sp.encode(reply))


# --------------------------------------------------------------------------- near-duplicate detection

class NearDup:
    """MinHash-LSH over word 5-gram shingles, per category. Near-dup = estimated Jaccard >= threshold."""
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


def per_request_seeds(base, n):
    """Distinct vLLM seeds per request. A single shared seed makes identical requests give identical outputs, which is
    what collapsed full-run attempt 1's prompt pools (explain: 5,532 unique of 215,299)."""
    return random.Random(base).sample(range(1, 2 ** 31), n)
