"""
Arithmetic battery (410 problems, + − × ÷) on a pretrained base model, before any fine-tune.

The prompt is encoded the way scripts/corpus/build_corpus.py wrote the corpus's arithmetic slot: one SentencePiece encode of
"You: {question}\nAura:" (ab_eval.generate encodes the pieces separately, which differs at the joins). Greedy, the reply is
cut at the next "You:", and the answer is the number after "Answer:" (else the last number). 1-digit cells measure fact recall:
the corpus trains every 1-digit fact (CLAUDE.md, corpus arithmetic slot). Multi-digit battery triples were excluded from it.
Two of the corpus's four templates are scored: "What is a OP b?" and "a OP b = ?".

Usage (from COSF-MI/): .venv/bin/python experiments/base_arith_eval.py NAME=CKPT [...] [--out experiments/results/base_arith]
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import sentencepiece as spm
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "python"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "scripts" / "distill"))
from arith_battery import GOLD, SYM, battery  # noqa: E402
from cosf_mi.models.cosf_oscillator import COSFOscillatorTransformer  # noqa: E402

TEMPLATES = {"what_is": "What is {a} {op} {b}?", "equals": "{a} {op} {b} = ?"}


def answer(reply):
    m = re.search(r"Answer:\s*(-?\d+)", reply)
    if m:
        return int(m.group(1))
    nums = re.findall(r"-?\d+", reply)
    return int(nums[-1]) if nums else None


@torch.no_grad()
def greedy(model, sp, prompts, seq, max_new=48, bs=64):
    dev = next(model.parameters()).device
    out = []
    for s in range(0, len(prompts), bs):
        ctx = [sp.encode(f"You: {p}\nAura:") for p in prompts[s:s + bs]]
        B = len(ctx)
        buf = torch.zeros(B, seq, dtype=torch.long, device=dev)
        for i, c in enumerate(ctx):
            buf[i, :len(c)] = torch.tensor(c)
        pos = torch.tensor([len(c) for c in ctx], device=dev)
        gen, done = [[] for _ in ctx], [False] * B
        for _ in range(max_new):
            T = int(pos.max())
            if all(done) or T >= seq:
                break
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(buf[:, :T])[torch.arange(B, device=dev), pos - 1].float()
            nxt = logits.argmax(-1)
            for i in range(B):
                if done[i]:
                    continue
                t = int(nxt[i])
                if t == sp.eos_id():
                    done[i] = True
                    continue
                gen[i].append(t)
                buf[i, pos[i]] = t
                pos[i] += 1
                if "You:" in sp.decode(gen[i][-4:]):
                    done[i] = True
        out += [sp.decode(g).split("You:")[0].strip() for g in gen]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpts", nargs="+", help="NAME=path/to/checkpoint.pt (repo-relative)")
    p.add_argument("--tokenizer", default="models/tokenizer_clean24k.model")
    p.add_argument("--out", default="experiments/results/base_arith")
    args = p.parse_args()
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    sp = spm.SentencePieceProcessor(model_file=str(ROOT / args.tokenizer))
    bat = battery()
    L = ["# Arithmetic battery on base models (no fine-tune)\n",
         "Greedy exact match, 410 problems. Prompt `You: {q}\\nAura:` as in the corpus. 1-digit cells = fact recall.\n"]
    for spec in args.ckpts:
        name, path = spec.split("=", 1)
        ckpt = torch.load(ROOT / path, map_location="cpu", weights_only=False)
        model = COSFOscillatorTransformer(ckpt["config"]).to("cuda")
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        seq = ckpt["config"].max_seq_len
        res = {"name": name, "checkpoint": path, "templates": {}}
        for tname, tmpl in TEMPLATES.items():
            prompts = [tmpl.format(a=b["a"], op=SYM[b["op"]], b=b["b"]) for b in bat]
            replies = greedy(model, sp, prompts, seq)
            by, ex = defaultdict(list), defaultdict(list)
            for b, q, r in zip(bat, prompts, replies):
                ok = answer(r) == GOLD[b["op"]](b["a"], b["b"])
                by[b["cell"]].append(ok)
                if len(ex[b["cell"]]) < 3:
                    ex[b["cell"]].append([q, r, ok])
            n_ok = sum(sum(v) for v in by.values())
            res["templates"][tname] = {"cells": {c: sum(v) / len(v) for c, v in by.items()}, "ALL": n_ok / len(bat),
                                       "examples": ex}
            print(f"[{name}] {tname}: {n_ok}/{len(bat)}", flush=True)
        (out_dir / f"{name}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
        L.append(f"\n## {name} (`{path}`)\n\n| cell | " + " | ".join(TEMPLATES) + " |\n|---|" + "---|" * len(TEMPLATES))
        for c in res["templates"]["what_is"]["cells"]:
            L.append(f"| {c} | " + " | ".join(f"{res['templates'][t]['cells'][c]:.0%}" for t in TEMPLATES) + " |")
        L.append("| **ALL** | " + " | ".join(f"**{res['templates'][t]['ALL']:.0%}**" for t in TEMPLATES) + " |")
        del model
        torch.cuda.empty_cache()
    (out_dir / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
