"""
Generate verified chat data for AURA from a local teacher model (Ollama).

Two kinds of samples:
  - numeric (60%): a math / numerics question, a short direct answer, and a
    Python block with asserts that checks the answer. The block is run in a
    sandbox and the sample is kept only if every assert passes.
  - dialogue (40%): a short direct conversation about AURA's own world
    (training runs, the GPU, sleep cycles, everyday tasks). Kept only if the
    replies have no assistant filler ("Great question!", "As an AI...").

Every kept sample fits AURA's 256-token window: the whole conversation, encoded
exactly as finetune_chat.py does it, must be at most --max-tokens (240). The
Python block is only used for checking and is not part of the training text.

Kept samples go to data/cosf_distill_verified.jsonl. Each record has a "line"
field in the TinyChat "[INST] user [/INST] reply" format that finetune_chat.py
parses. Rejects (with the reason) go to data/cosf_distill_rejected.log.
The run resumes: it counts the records already in the output file.

Usage:
    ollama pull qwen2.5:32b
    python scripts/generate_cosf_distill.py --n 5000
"""

import argparse
import ast
import asyncio
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import psutil

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------

ALLOWED_IMPORTS = {"math", "cmath", "fractions", "decimal", "statistics",
                   "itertools", "functools", "random", "numpy"}
BANNED_CALLS = {"open", "exec", "eval", "compile", "__import__", "input",
                "globals", "locals", "getattr", "setattr", "delattr", "breakpoint"}


@dataclass
class SandboxResult:
    ok: bool
    reason: str          # "ok", "syntax", "unsafe", "no_assert", "timeout", "memory", "error"
    detail: str = ""
    seconds: float = 0.0
    peak_mb: float = 0.0


def check_code(code):
    """Static check before running: parses, only math-ish imports, no file/exec access,
    and at least one assert. Returns (reason, detail); reason is "ok" when it passes."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return "syntax", f"line {e.lineno}: {e.msg}"
    has_assert = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            has_assert = True
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for name in names:
                if name.split(".")[0] not in ALLOWED_IMPORTS:
                    return "unsafe", f"import {name}"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in BANNED_CALLS:
            return "unsafe", f"call {node.func.id}()"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return "unsafe", f"dunder attribute {node.attr}"
    if not has_assert:
        return "no_assert", "verification block has no assert"
    return "ok", ""


def _tree_rss_mb(proc):
    try:
        procs = [proc] + proc.children(recursive=True)
        return sum(p.memory_info().rss for p in procs) / 2**20
    except psutil.Error:
        return 0.0


def _kill_tree(proc):
    try:
        for p in proc.children(recursive=True) + [proc]:
            p.kill()
    except psutil.Error:
        pass


def run_sandboxed(code, timeout=5.0, mem_limit_mb=512, poll=0.02):
    """Run `code` in a fresh isolated Python (-I) inside an empty temp dir.
    A psutil watchdog kills the process tree past `timeout` seconds or
    `mem_limit_mb` of resident memory. Output goes to files, not pipes, so a
    chatty script can't fill a pipe buffer and hang."""
    reason, detail = check_code(code)
    if reason != "ok":
        return SandboxResult(False, reason, detail)

    with tempfile.TemporaryDirectory(prefix="cosf_sandbox_") as tmp:
        script = Path(tmp) / "check.py"
        script.write_text(code, encoding="utf-8")
        env = {k: os.environ[k] for k in ("SYSTEMROOT", "PATH", "TEMP", "TMP") if k in os.environ}
        env["PYTHONIOENCODING"] = "utf-8"
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        with open(Path(tmp) / "out.txt", "wb") as out, open(Path(tmp) / "err.txt", "wb") as err:
            popen = subprocess.Popen([sys.executable, "-I", str(script)], cwd=tmp, env=env,
                                     stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                     creationflags=flags)
            proc = psutil.Process(popen.pid)
            start, peak, killed = time.monotonic(), 0.0, None
            while popen.poll() is None:
                peak = max(peak, _tree_rss_mb(proc))
                elapsed = time.monotonic() - start
                if elapsed > timeout:
                    killed = "timeout"
                elif peak > mem_limit_mb:
                    killed = "memory"
                if killed:
                    _kill_tree(proc)
                    popen.wait()
                    break
                time.sleep(poll)
            seconds = time.monotonic() - start
        stderr = (Path(tmp) / "err.txt").read_text(encoding="utf-8", errors="replace")

    if killed == "timeout":
        return SandboxResult(False, "timeout", f"> {timeout:.1f}s", seconds, peak)
    if killed == "memory":
        return SandboxResult(False, "memory", f"> {mem_limit_mb} MB", seconds, peak)
    if popen.returncode != 0:
        last = stderr.strip().splitlines()[-1] if stderr.strip() else f"exit code {popen.returncode}"
        return SandboxResult(False, "error", last[:300], seconds, peak)
    return SandboxResult(True, "ok", "", seconds, peak)


# ---------------------------------------------------------------------------
# Parsing teacher output
# ---------------------------------------------------------------------------

CODE_BLOCK = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
THINK = re.compile(r"<think>.*?</think>", re.DOTALL)  # deepseek-r1 reasoning
FILLER = re.compile(
    r"\b(great question|good question|i'?d be (happy|glad) to|happy to help|"
    r"as an ai|as a language model|certainly!|absolutely!|of course!|"
    r"i hope this helps|feel free to|let me know if)\b", re.IGNORECASE)


def extract_code_blocks(text):
    return [b.strip() for b in CODE_BLOCK.findall(text)]


def clean(text):
    """One line of plain text: no code, no markdown emphasis, collapsed spaces."""
    text = CODE_BLOCK.sub("", text)
    text = re.sub(r"[*_`#]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_numeric(text):
    """QUESTION: / ```python``` / ANSWER: (either order) -> (question, answer, code) or None."""
    text = THINK.sub("", text)
    q = re.search(r"QUESTION:\s*(.*?)\s*(?:```|ANSWER:)", text, re.DOTALL | re.IGNORECASE)
    a = re.search(r"ANSWER:\s*(.*?)(?:```|$)", text, re.DOTALL | re.IGNORECASE)
    blocks = extract_code_blocks(text)
    if not (q and a and blocks):
        return None
    return clean(q.group(1)), clean(a.group(1)), "\n\n".join(blocks)


def parse_dialogue(text):
    """'USER: ... AURA: ...' lines -> [(user, aura), ...]"""
    text = THINK.sub("", text)
    parts = re.split(r"^\s*(USER|AURA)\s*:\s*", text, flags=re.MULTILINE | re.IGNORECASE)
    turns, pending = [], None
    for speaker, said in zip(parts[1::2], parts[2::2]):
        said = clean(said)
        if speaker.upper() == "USER":
            pending = said
        elif pending:
            turns.append((pending, said))
            pending = None
    return [(u, a) for u, a in turns if u and a]


def norm_key(turns):
    """Dedup key: the first user message, lowercased, punctuation dropped."""
    return re.sub(r"[^a-z0-9 ]", "", turns[0][0].lower()).strip()


def to_line(turns):
    return " ".join(f"[INST] {u} [/INST] {a}" for u, a in turns)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# AURA's context is 256 tokens, so everything is kept short and simple.
NUMERIC_TOPICS = [
    ("arithmetic", "single-digit or two-digit addition, like 'what is 5+5?'", 4),
    ("arithmetic", "two-digit subtraction or multiplication", 3),
    ("arithmetic", "a sum of three or four small numbers, like 6+6+6", 2),
    ("word problem", "a one-step everyday word problem (apples, coins, minutes)", 3),
    ("word problem", "a two-step everyday word problem", 2),
    ("fractions", "simple fractions or percentages", 2),
    ("sequences", "the next number in a simple pattern, or counting", 1),
    ("angles", "angles and wrapping a phase into [0, 2*pi)", 1),
    ("oscillators", "the period, frequency or value of sin(w*t + phi) at a time", 1),
    ("kuramoto", "the Kuramoto order parameter R = |mean(exp(i*theta))| for a few given phases", 1),
    ("complex", "one step of the Julia map z -> z*z + c, or |z| of a complex number", 1),
    ("numerics", "one Euler step of a simple equation like dx/dt = -x", 1),
    ("cuda", "CUDA thread index math: idx = blockIdx.x * blockDim.x + threadIdx.x, or how many blocks cover n items", 1),
]

DIALOGUE_TOPICS = [
    "checking on an overnight training run and its validation loss",
    "deciding whether the GPU is free for a game or a training job",
    "what a sleep cycle does for a model that learns new data",
    "planning the day: chores, a workout and a coding session",
    "fixing a script that crashed after a computer restart",
    "picking a good name for a new experiment folder",
    "explaining what a heartbeat in a neural network means, simply",
    "a friend asking how the AURA project is going",
    "deciding what to cook with rice, eggs and onions",
    "the user is tired after work and wants a short plan for the evening",
    "choosing a video game to play tonight",
    "a reminder to drink water and take a break from the screen",
    "how to keep a PC cool during a long job",
    "what to do when a download is stuck",
    "deciding whether to train a bigger model or get better data",
    "the user asks about the weather or the news (AURA can't see them and says so)",
    "making a shopping list for the week",
    "explaining what validation loss means, simply",
    "the user wants to learn a little Python",
    "talking about a fish screensaver that shows running jobs",
]

NUMERIC_PROMPT = """Write one {topic} question about {detail}.
Build it around these numbers (use the ones that fit): {nums}.

Reply in exactly this format:
QUESTION: <the question, one sentence, plain words>
```python
# work the answer out step by step in code, then assert the final value
```
ANSWER: <the direct answer in 1-2 short sentences, stating the value the code computed>

Work it out in the code FIRST, carefully, then copy the computed value into the ANSWER.
Rules: the QUESTION is under 30 words and the ANSWER under 35 words.
The ANSWER has no code, no LaTeX and no markdown. Round decimals to 2 places.
The Python block is under 15 lines, uses only the standard library (math, cmath,
fractions), and must assert the final value, e.g. assert round(result, 2) == 24.7"""

# Dialogue isn't checked by code, so the teacher gets the real facts to stay inside.
AURA_FACTS = """AURA is a small language model the user is training at home on an RTX 5090 GPU.
It is built on COSF, where the neurons are phase oscillators instead of plain weights.
AURA has no internet, camera or sensors: it never states live facts like the weather,
the time, news or prices, and says it can't see them.
Its "heartbeat" gently pulls the oscillator phases toward loose synchrony.
Its "sleep" replays old training examples so it doesn't forget them while learning new ones.
Training runs overnight, saves a checkpoint every 3000 steps, and resumes after a crash.
Lower validation loss means a better model. AURA is still learning basic math."""

OPENERS = ["a question", "a statement of a problem", "a short greeting and a request",
           "a complaint", "a quick yes/no question", "a request for a plan", "a joke or a casual remark"]

DIALOGUE_PROMPT = """Write a short conversation between USER and AURA, a direct, friendly
assistant that lives on the user's PC. Topic: {topic} (variation #{seed}).
USER opens with {opener}, in their own words.

Facts about AURA (follow them; when the topic touches AURA, invent no others):
{facts}

Format, {turns} exchanges, each on its own line:
USER: <message>
AURA: <reply>

Rules: every message is under 25 words. AURA's replies are 1-2 short sentences,
concrete, and answer straight away. The whole conversation is under 120 words.
No filler: never "Great question", "I'd be happy to", "As an AI", "Certainly",
"Feel free to" or "Let me know if". No markdown, no lists, no emojis."""

SYSTEM = "You write short, correct training examples. Follow the format exactly."


def pick_numbers(rng):
    """Random numbers for the teacher: left to itself it reuses 34, 3/8, 3,6,9... over and over."""
    nums = [rng.randint(1, 9), rng.randint(2, 12), rng.randint(10, 99), rng.randint(10, 99), rng.randint(100, 999)]
    rng.shuffle(nums)
    return ", ".join(map(str, nums))


def make_task(rng, numeric_share):
    """numeric_share here is the chance of asking for a numeric sample."""
    seed = rng.randrange(10**6)
    if rng.random() < numeric_share:
        topics = NUMERIC_TOPICS
        topic, detail, _ = rng.choices(topics, weights=[w for *_, w in topics])[0]
        return {"category": "numeric", "topic": topic,
                "prompt": NUMERIC_PROMPT.format(topic=topic, detail=detail, nums=pick_numbers(rng))}
    topic = rng.choice(DIALOGUE_TOPICS)
    return {"category": "dialogue", "topic": topic,
            "prompt": DIALOGUE_PROMPT.format(topic=topic, seed=seed, turns=rng.choice([2, 3, 4]),
                                             opener=rng.choice(OPENERS), facts=AURA_FACTS)}


def make_token_counter():
    """Token count of a conversation, encoded exactly as finetune_chat.py trains on it."""
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src" / "python"))
    from finetune_chat import encode_chat
    from cosf_mi.data.tokenizer import BPETokenizer
    tok = BPETokenizer(str(ROOT / "models/tokenizer_mix16k.model"))
    return lambda turns: len(encode_chat(tok, turns)[0])


def verify(task, text, timeout, mem_limit_mb, count_tokens=None, max_tokens=240):
    """Teacher text -> (record or None, reject reason, detail).
    count_tokens(turns) -> int enables the context-window check."""
    if task["category"] == "numeric":
        parsed = parse_numeric(text)
        if not parsed:
            return None, "format", "missing QUESTION/ANSWER/python block"
        question, answer, code = parsed
        if len(answer.split()) > 60:
            return None, "too_long", f"{len(answer.split())} words"
        result = run_sandboxed(code, timeout=timeout, mem_limit_mb=mem_limit_mb)
        if not result.ok:
            return None, result.reason, result.detail
        turns = [(question, answer)]
        if count_tokens and count_tokens(turns) > max_tokens:
            return None, "too_many_tokens", f"{count_tokens(turns)} > {max_tokens}"
        return {"category": "numeric", "topic": task["topic"], "turns": turns,
                "code": code, "line": to_line(turns)}, "ok", ""

    turns = parse_dialogue(text)
    if not turns:
        return None, "format", "no USER/AURA turns"
    for _, reply in turns:
        if FILLER.search(reply):
            return None, "filler", FILLER.search(reply).group(0)
        if len(reply.split()) > 45:
            return None, "too_long", f"{len(reply.split())} words"
    if count_tokens and count_tokens(turns) > max_tokens:
        return None, "too_many_tokens", f"{count_tokens(turns)} > {max_tokens}"
    return {"category": "dialogue", "topic": task["topic"], "turns": turns,
            "line": to_line(turns)}, "ok", ""


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def ask_teacher(client, args, prompt):
    r = await client.post(f"{args.base_url}/chat/completions", json={
        "model": args.model, "temperature": args.temperature, "max_tokens": 700,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
    })
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


async def run(args):
    out_path, rej_path = ROOT / args.out, ROOT / args.rejected
    out_path.parent.mkdir(parents=True, exist_ok=True)
    old = [json.loads(l) for l in open(out_path, encoding="utf-8")] if out_path.exists() else []
    have = len(old)
    if have >= args.n:
        print(f"{out_path} already has {have} samples (target {args.n})")
        return
    print(f"Teacher {args.model} at {args.base_url}; {have} samples so far, target {args.n}")

    count_tokens = make_token_counter()
    rng = random.Random(args.seed + have)
    stats = {"kept": have, "tried": 0}
    quota = {"numeric": round(args.n * args.numeric_share)}
    quota["dialogue"] = args.n - quota["numeric"]
    kept = {c: sum(r["category"] == c for r in old) for c in quota}
    seen = {norm_key(r["turns"]) for r in old}
    reasons = {}
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(args.concurrency)
    started = time.time()

    async def one(client, out, rej):
        left = {c: max(0, quota[c] - kept[c]) for c in quota}
        task = make_task(rng, left["numeric"] / max(1, left["numeric"] + left["dialogue"]))
        async with sem:
            try:
                text = await ask_teacher(client, args, task["prompt"])
            except (httpx.HTTPError, KeyError, ValueError) as e:
                record, reason, detail, text = None, "teacher", str(e)[:200], ""
            else:
                # the sandbox blocks while it polls, so run it off the event loop
                record, reason, detail = await asyncio.to_thread(
                    verify, task, text, args.timeout, args.mem_limit_mb, count_tokens, args.max_tokens)
        async with lock:
            stats["tried"] += 1
            if record and norm_key(record["turns"]) in seen:
                record, reason, detail = None, "duplicate", record["turns"][0][0][:80]
            elif record and kept[record["category"]] >= quota[record["category"]]:
                record = None
                reason = "quota_full"
            if record and stats["kept"] < args.n:
                seen.add(norm_key(record["turns"]))
                kept[record["category"]] += 1
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                stats["kept"] += 1
            elif not record and reason != "quota_full":
                reasons[reason] = reasons.get(reason, 0) + 1
                rej.write(json.dumps({"category": task["category"], "topic": task["topic"],
                                      "reason": reason, "detail": detail,
                                      "teacher": text[:1500]}, ensure_ascii=False) + "\n")
                rej.flush()
            if stats["tried"] % 10 == 0:
                secs = max(time.time() - started, 1)
                rate = (stats["kept"] - have) / secs
                left = (args.n - stats["kept"]) / max(rate, 1e-9)
                print(f"  kept {stats['kept']}/{args.n} {kept}  tried {stats['tried']}  "
                      f"rejects {reasons}  ~{rate * 3600:.0f}/h", flush=True)
                # tqdm-style line so fish.bat can follow the run; loss= is the reject rate
                print(f"Epoch 1/1: {100 * stats['kept'] // args.n}%|| {stats['kept']}/{args.n} "
                      f"[{secs / 60:.0f}m<{int(left // 3600)}h{int(left % 3600 // 60):02d}m, {rate:.2f}it/s, "
                      f"loss={sum(reasons.values()) / stats['tried']:.4f}]", flush=True)

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        with open(out_path, "a", encoding="utf-8") as out, open(rej_path, "a", encoding="utf-8") as rej:
            while stats["kept"] < args.n:
                batch = min(args.concurrency * 4, (args.n - stats["kept"]) * 2)
                await asyncio.gather(*(one(client, out, rej) for _ in range(batch)))

    print(f"Done. {stats['kept']} samples in {out_path} ({kept}); rejects {reasons}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:11434/v1")
    p.add_argument("--model", default="qwen2.5:32b")
    p.add_argument("--n", type=int, default=1000, help="target total samples in --out")
    p.add_argument("--numeric-share", type=float, default=0.6)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--timeout", type=float, default=5.0, help="sandbox seconds per check")
    p.add_argument("--mem-limit-mb", type=int, default=512, help="sandbox memory cap")
    p.add_argument("--max-tokens", type=int, default=240, help="max tokens per conversation (AURA's window is 256)")
    p.add_argument("--out", default="data/cosf_distill_verified.jsonl")
    p.add_argument("--rejected", default="data/cosf_distill_rejected.log")
    p.add_argument("--seed", type=int, default=0)
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
