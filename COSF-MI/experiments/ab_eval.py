"""
Distill A/B evaluation: every checkpoint is re-scored with this same code.

1. Frozen val sets (reply-only loss, fp32): TinyChat chat val (2,252, as aura_chat_distill), old distill val
   (500, split seed 0), story val (512 styled; forgetting), pilot val by category (<pilot>/val.jsonl).
2. Arithmetic battery (experiments/arith_battery.py): greedy, exact match on the last number, by op × digits,
   plus the share with a final "Answer: N" line. Also the 120-question math_eval exam (its own decoding).
3. The fixed 40-prompt readable set (experiments/readable_40.json): greedy + one sampled reply.
4. Format and degeneration on ~370 prompts (readable + TinyChat val + pilot val), greedy and sampled:
   question endings, turn ended ("You:" written before the cap), distinct-2/3, repeated 4-grams, length.

Generation here is plain: no repetition penalty or n-gram ban (chat_with_aura.py's anti-repeat settings
would hide the repetition this measures and penalise repeated digits). Batched with right padding,
which is exact: the model's eval forward has no cross-position ops besides causal attention.

Usage (from COSF-MI/):
  .venv/bin/python experiments/ab_eval.py --out experiments/results/ab_distill NAME=CKPT [NAME=CKPT ...]
  .venv/bin/python experiments/ab_eval.py --out experiments/results/ab_distill --report-only
"""
import argparse
import collections
import contextlib
import io
import json
import random
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "python"))
sys.path.insert(0, str(ROOT / "experiments"))
import finetune_chat as ft  # noqa: E402
import math_eval  # noqa: E402
from arith_battery import battery  # noqa: E402
from cosf_mi.data.tokenizer import BPETokenizer  # noqa: E402
from cosf_mi.data.token_dataset_fast import create_datasets_fast  # noqa: E402
from cosf_mi.models.cosf_oscillator import COSFOscillatorTransformer  # noqa: E402

SEQ = ft.SEQ
READABLE = json.loads((ROOT / "experiments/readable_40.json").read_text())
RCATS = ("chat", "explain", "story", "maths")


# --------------------------------------------------------------------------- data

def frozen_sets(tok, pilot_dir):
    """{name: (x, y)} for every frozen val set."""
    sets = {}
    lines = Path(ROOT / "data/tinychat/tinychat_80mb.txt").read_text(encoding="utf-8").splitlines()[:-1]
    random.Random(1234).shuffle(lines)
    n_val = max(500, len(lines) // 50)  # all lines, as aura_chat_distill: 2,252
    sets["chat (TinyChat)"] = ft.build_chat_tensors(tok, lines[:n_val])
    _, d_val, _, _ = ft.load_distill(ROOT / "data/cosf_distill_verified.jsonl", 0.3, 1, 4, 0)
    sets["old distill"] = ft.build_chat_tensors(tok, d_val)
    with contextlib.redirect_stdout(io.StringIO()):
        _, story_val = create_datasets_fast(
            train_path=str(ROOT / "data/mix/styled_train.txt"), val_path=str(ROOT / "data/mix/styled_val.txt"),
            tokenizer=tok, seq_len=SEQ, cache_dir=str(ROOT / "cache/tokenized_styled"))
    g = torch.Generator().manual_seed(1234)  # same draw as finetune_chat.py
    vidx = torch.randperm(len(story_val), generator=g)[:512]
    sv = [story_val[int(i)] for i in vidx]
    sets["story (forgetting)"] = (torch.stack([s[0] for s in sv]), torch.stack([s[1] for s in sv]))
    by_cat = collections.defaultdict(list)
    for line in open(Path(pilot_dir) / "val.jsonl", encoding="utf-8"):
        r = json.loads(line)
        by_cat[r["category"]].append(r)
    for c in sorted(by_cat):
        sets[f"pilot {c}"] = ft.build_chat_tensors(tok, [r["line"] for r in by_cat[c]])
    sets["pilot all"] = ft.build_chat_tensors(tok, [r["line"] for rs in by_cat.values() for r in rs])
    return sets, lines[:n_val], by_cat


def rate_prompts(chat_val_lines, pilot_by_cat):
    """(kind, prompt) for the format/degeneration stats. kind: 'convo' (a question ending is a choice), 'story', 'maths'."""
    out = [("story" if c == "story" else "maths" if c == "maths" else "convo", p) for c in RCATS for p in READABLE[c]]
    for line in chat_val_lines[:100]:
        turns = ft.parse_conversation(line)
        if turns:
            out.append(("convo", turns[0][0]))
    rng = random.Random(7)
    for c, n in (("chat", 60), ("explain", 60), ("oasst2", 60), ("story", 40), ("arith", 20), ("gsm8k", 20)):
        for r in rng.sample(pilot_by_cat[c], n):
            out.append(("story" if c == "story" else "maths" if c in ("arith", "gsm8k") else "convo", r["prompt"]))
    return out


# --------------------------------------------------------------------------- generation

@torch.no_grad()
def generate(model, tok, prompts, greedy, max_new=200, temperature=0.7, top_k=40, seed=0, bs=64):
    """[(reply, ended)] per prompt; ended = the model wrote "You:" (its turn is over) before the cap."""
    dev = next(model.parameters()).device
    user_ids, aura_ids = tok.encode(ft.USER), tok.encode(ft.AURA)
    k = len(user_ids)
    out = []
    for s in range(0, len(prompts), bs):
        ctx = [(user_ids + tok.encode(p) + aura_ids)[-(SEQ - 8):] for p in prompts[s:s + bs]]
        B = len(ctx)
        buf = torch.zeros(B, SEQ, dtype=torch.long, device=dev)
        for i, c in enumerate(ctx):
            buf[i, :len(c)] = torch.tensor(c)
        pos = torch.tensor([len(c) for c in ctx], device=dev)
        gen, done, ended = [[] for _ in ctx], [False] * B, [False] * B
        g = torch.Generator(device=dev).manual_seed(seed * 100_003 + s)
        for _ in range(max_new):
            T = int(pos.max())
            if all(done) or T >= SEQ:
                break
            logits = model(buf[:, :T])[torch.arange(B, device=dev), pos - 1].float()
            if greedy:
                nxt = logits.argmax(-1)
            else:
                logits = logits / temperature
                kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
                logits[logits < kth] = -float("inf")
                nxt = torch.multinomial(torch.softmax(logits, -1), 1, generator=g)[:, 0]
            nxt_l = nxt.tolist()
            for i in range(B):
                if done[i]:
                    continue
                gen[i].append(nxt_l[i])
                buf[i, pos[i]] = nxt_l[i]
                pos[i] += 1
                if gen[i][-k:] == user_ids:
                    gen[i], done[i], ended[i] = gen[i][:-k], True, True
                elif int(pos[i]) >= SEQ:
                    done[i] = True
        out += [(tok.decode(x).strip(), e) for x, e in zip(gen, ended)]
    return out


# --------------------------------------------------------------------------- metrics

def last_number(text):
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(nums[-1]) if nums else None


def words(t):
    return re.findall(r"[a-z0-9']+", t.lower())


def ngrams(w, n):
    return [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]


def text_stats(rows):
    """rows: [(kind, reply, ended)] -> question endings, turn ended, distinct-2/3, repeated 4-grams, length, top opener."""
    convo = [r for k, r, _ in rows if k == "convo"]
    d = {"n": len(rows), "ends_with_question_convo": sum(r.rstrip().endswith("?") for r in convo) / max(1, len(convo)),
         "ends_with_question_story_maths": sum(r.rstrip().endswith("?") for k, r, _ in rows if k != "convo")
         / max(1, sum(k != "convo" for k, _, _ in rows)),
         "turn_ended": sum(e for _, _, e in rows) / len(rows)}
    ws = [words(r) for _, r, _ in rows]
    for n in (2, 3):
        per = [len(set(ngrams(w, n))) / len(ngrams(w, n)) for w in ws if len(w) > n]
        d[f"distinct{n}_mean"] = sum(per) / max(1, len(per))
        allg = [g for w in ws for g in ngrams(w, n)]
        d[f"distinct{n}_corpus"] = len(set(allg)) / max(1, len(allg))
    d["repeated_4gram"] = sum(any(c > 1 for c in collections.Counter(ngrams(w, 4)).values()) for w in ws) / len(ws)
    d["loop_4gram_3x"] = sum(any(c > 2 for c in collections.Counter(ngrams(w, 4)).values()) for w in ws) / len(ws)
    d["mean_words"] = sum(map(len, ws)) / len(ws)
    openers = collections.Counter(" ".join(w[:3]) for w in ws if len(w) >= 3)
    top, k = openers.most_common(1)[0] if openers else ("", 0)
    d["top_opener"], d["top_opener_share"] = top, k / len(ws)
    return d


@torch.no_grad()
def val_losses(model, sets):
    return {name: ft.val_loss(model, x, y, next(model.parameters()).device, bs=64) for name, (x, y) in sets.items()}


# --------------------------------------------------------------------------- per checkpoint

def evaluate(name, ckpt_path, tok, sets, rprompts, exam, out_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = COSFOscillatorTransformer(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    res = {"name": name, "checkpoint": str(ckpt_path),
           "run_config": json.loads((Path(ckpt_path).parent / "run_config.json").read_text())
           if (Path(ckpt_path).parent / "run_config.json").exists() else None}
    print(f"[{name}] val losses...", flush=True)
    res["val"] = val_losses(model, sets)

    print(f"[{name}] arithmetic battery...", flush=True)
    bat = battery()
    replies = generate(model, tok, [b["prompt"] for b in bat], greedy=True, max_new=120)
    res["battery"] = [dict(b, reply=r, ended=e, correct=last_number(r) == b["gold"],
                           answer_line=bool(re.search(rf"Answer:\s*{b['gold']}\s*$", r.replace(",", ""))))
                      for b, (r, e) in zip(bat, replies)]

    print(f"[{name}] readable set + format stats...", flush=True)
    rp = [p for _, p in rprompts]
    greedy = generate(model, tok, rp, greedy=True)
    sampled = generate(model, tok, rp, greedy=False, seed=1)
    res["rates"] = {"greedy": text_stats([(k, r, e) for (k, _), (r, e) in zip(rprompts, greedy)]),
                    "sampled": text_stats([(k, r, e) for (k, _), (r, e) in zip(rprompts, sampled)])}
    nr = sum(len(READABLE[c]) for c in RCATS)  # the readable set comes first in rprompts
    res["readable"] = [{"prompt": p, "greedy": g[0], "sampled": s[0]} for p, g, s in zip(rp[:nr], greedy[:nr], sampled[:nr])]
    gold = READABLE["maths_gold"]
    res["readable_maths_correct"] = {"greedy": sum(last_number(r["greedy"]) == a for r, a in zip(res["readable"][30:], gold)),
                                     "sampled": sum(last_number(r["sampled"]) == a for r, a in zip(res["readable"][30:], gold))}
    del model
    torch.cuda.empty_cache()

    print(f"[{name}] math_eval exam (120)...", flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        rows = math_eval.run(ckpt_path, exam, device)
    res["math_eval"] = rows
    (out_dir / f"{name}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    return res


# --------------------------------------------------------------------------- report

def group(names):
    """Arms by name prefix: 'new_s0', 'new_s1' -> 'new'."""
    arms = collections.defaultdict(list)
    for n in names:
        arms[re.sub(r"_s\d+$", "", n)].append(n)
    return arms


def fmt_arm(vals, f="{:.4f}"):
    if len(vals) == 1:
        return f.format(vals[0])
    m = sum(vals) / len(vals)
    return f.format(m) + " (" + ", ".join(f.format(v) for v in vals) + ")"


def report(out_dir, names):
    R = {n: json.loads((out_dir / f"{n}.json").read_text()) for n in names}
    arms = group(names)
    base = json.loads((out_dir / "base_val.json").read_text()) if (out_dir / "base_val.json").exists() else None
    L = ["# Distill A/B results\n", "Every checkpoint re-scored with `experiments/ab_eval.py`. Arms with 2 seeds show "
         "mean (seed values). The arms are not like for like: `new` trains on the licence-clean pilot set only "
         "(9,890 steps, no TinyChat), `old` on TinyChat + the old distill set (3,949 steps). So the comparison leads "
         "with the arithmetic battery, the readable set and story forgetting.\n"]
    L.append("\n## Arithmetic battery (greedy, exact match on the last number)\n")
    cells = list(dict.fromkeys(b["cell"] for b in next(iter(R.values()))["battery"]))
    L.append("| cell | n | " + " | ".join(arms) + " |")
    L.append("|---|---|" + "---|" * len(arms))
    pct = "{:.0f}%"
    for cell in cells + ["ALL"]:
        row = []
        for ns in arms.values():
            vals = []
            for n in ns:
                sel = [b for b in R[n]["battery"] if cell in ("ALL", b["cell"])]
                vals.append(100 * sum(b["correct"] for b in sel) / len(sel))
            row.append(fmt_arm(vals, pct))
        n_cell = sum(1 for b in next(iter(R.values()))["battery"] if cell in ("ALL", b["cell"]))
        L.append(f"| {cell} | {n_cell} | " + " | ".join(row) + " |")
    row = [fmt_arm([100 * sum(b["answer_line"] for b in R[n]["battery"]) / len(R[n]["battery"]) for n in ns], pct)
           for ns in arms.values()]
    L.append("| ends with `Answer: <gold>` | | " + " | ".join(row) + " |")

    L.append("\n## Readable set (40 prompts)\n")
    L.append("Side by side in `readable.md`. Maths prompts answered correctly (last number = gold, of 10):\n")
    L.append("| decoding | " + " | ".join(arms) + " |")
    L.append("|---|" + "---|" * len(arms))
    for mode in ("greedy", "sampled"):
        L.append(f"| {mode} | " + " | ".join(fmt_arm([R[n]["readable_maths_correct"][mode] for n in ns], "{:.0f}")
                                           for ns in arms.values()) + " |")

    L.append("\n## Story val (forgetting), before -> after\n")
    sv = "story (forgetting)"
    if base:
        L.append(f"Before = the base `{Path(base['checkpoint']).parent.name}` scored by this code: {base['val'][sv]:.4f}.\n")
    L.append("| arm | story val after | change |")
    L.append("|---|---|---|")
    for arm, ns in arms.items():
        after = [R[n]["val"][sv] for n in ns]
        delta = fmt_arm([a - base["val"][sv] for a in after], "{:+.4f}") if base else "-"
        L.append(f"| {arm} | {fmt_arm(after)} | {delta} |")

    L.append("\n## Other frozen val sets (reply-only loss, lower is better)\n")
    L.append("TinyChat val is **not comparable** between arms: `old` trains on TinyChat and `new` never sees it. "
             "Old distill val favours `old` the same way, and pilot val favours `new`.\n")
    L.append("| val set | " + ("base | " if base else "") + " | ".join(arms) + " |")
    L.append("|---|" + "---|" * (len(arms) + bool(base)))
    for vs in next(iter(R.values()))["val"]:
        if vs == sv:
            continue
        lab = vs + (" (not comparable)" if vs == "chat (TinyChat)" else "")
        L.append(f"| {lab} | " + (f"{base['val'][vs]:.4f} | " if base else "")
                 + " | ".join(fmt_arm([R[n]["val"][vs] for n in ns]) for ns in arms.values()) + " |")

    L.append("\n## math_eval exam (120 questions, its own near-greedy decoding with repetition penalty)\n")
    kinds = list(dict.fromkeys(r["kind"] for r in next(iter(R.values()))["math_eval"]))
    L.append("| kind | " + " | ".join(arms) + " |")
    L.append("|---|" + "---|" * len(arms))
    for kind in kinds + ["TOTAL"]:
        row = [fmt_arm([sum(r["correct"] for r in R[n]["math_eval"] if kind in ("TOTAL", r["kind"])) for n in ns], "{:.0f}")
               for ns in arms.values()]
        L.append(f"| {kind} | " + " | ".join(row) + " |")

    for mode in ("greedy", "sampled"):
        st = next(iter(R.values()))["rates"][mode]
        L.append(f"\n## Format and repetition, {mode} (n = {st['n']} prompts)\n")
        L.append("| metric | " + " | ".join(arms) + " |")
        L.append("|---|" + "---|" * len(arms))
        for m, lab, f in (("ends_with_question_convo", "ends in \"?\" (chat/explain prompts)", pct),
                          ("ends_with_question_story_maths", "ends in \"?\" (story/maths prompts)", pct),
                          ("turn_ended", "turn ended (wrote \"You:\")", pct),
                          ("distinct2_mean", "distinct-2 per reply", "{:.3f}"),
                          ("distinct3_mean", "distinct-3 per reply", "{:.3f}"),
                          ("distinct2_corpus", "distinct-2 across replies", "{:.3f}"),
                          ("repeated_4gram", "replies with a repeated 4-gram", pct),
                          ("loop_4gram_3x", "replies with a 4-gram 3+ times", pct),
                          ("mean_words", "mean words", "{:.0f}"),
                          ("top_opener_share", "most common 3-word opener share", pct)):
            mult = 100 if f == pct else 1
            L.append(f"| {lab} | " + " | ".join(fmt_arm([mult * R[n]["rates"][mode][m] for n in ns], f)
                                                 for ns in arms.values()) + " |")
        L.append("| most common opener | " + " | ".join(
            "; ".join(f"\"{R[n]['rates'][mode]['top_opener']}\"" for n in ns) for ns in arms.values()) + " |")
    (out_dir / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # readable set, side by side: one table per prompt, a row per checkpoint
    cut = lambda t: t.replace("|", "\\|").replace("\n", "<br>")  # noqa: E731
    M = ["# Readable set, side by side\n", "Greedy and one sampled reply (temperature 0.7, top-k 40, no repetition "
         "penalty) per checkpoint.\n"]
    first = next(iter(R.values()))["readable"]
    for i, row in enumerate(first):
        if i % 10 == 0:
            M.append(f"\n## {RCATS[i // 10]}\n")
        M.append(f"\n### {i + 1}. {row['prompt']}\n")
        M.append("| checkpoint | greedy | sampled |")
        M.append("|---|---|---|")
        for n in names:
            r = R[n]["readable"][i]
            M.append(f"| {n} | {cut(r['greedy'])} | {cut(r['sampled'])} |")
    (out_dir / "readable.md").write_text("\n".join(M) + "\n", encoding="utf-8")
    print((out_dir / "summary.md").read_text())


def base_val(path, sets, out_dir):
    """Frozen val losses of the base model every arm starts from (the "before" of story val)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = COSFOscillatorTransformer(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    res = {"checkpoint": str(path), "val": val_losses(model, sets)}
    (out_dir / "base_val.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    del model
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="*", help="NAME=path/to/best_model.pt")
    p.add_argument("--out", default="experiments/results/ab_distill")
    p.add_argument("--pilot", default="data/distill_pilot/ft_v1")
    p.add_argument("--base", default=None, help="base checkpoint: its val losses become the 'before' of story val")
    p.add_argument("--report-only", action="store_true", help="rebuild summary.md/readable.md from the saved JSON")
    p.add_argument("--names", default=None, help="comma-separated order for the report (default: every JSON in --out)")
    args = p.parse_args()
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        tok = BPETokenizer(str(ROOT / "models/tokenizer_mix16k.model"))
        sets, chat_val_lines, pilot_by_cat = frozen_sets(tok, ROOT / args.pilot)
        rprompts = rate_prompts(chat_val_lines, pilot_by_cat)
        skip = set()
        for line in open(ROOT / "data/cosf_distill_verified.jsonl", encoding="utf-8"):
            skip.add(json.loads(line)["turns"][0][0].lower())
        exam = math_eval.make_exam(skip=skip)
        if args.base:
            base_val(ROOT / args.base, sets, out_dir)
        for spec in args.ckpts:
            name, path = spec.split("=", 1)
            evaluate(name, ROOT / path, tok, sets, rprompts, exam, out_dir)
    names = args.names.split(",") if args.names else sorted(f.stem for f in out_dir.glob("*.json")
                                                                   if f.stem not in ("base_val", "ft_v1_build_report"))
    report(out_dir, names)


if __name__ == "__main__":
    main()
