"""
Format-neutral knowledge probe for base models (2026-10-31). The chat-format factuality eval is confounded for bases: what "Aura:"
triggers depends on which "You: … Aura: …" text the corpus held (the 10B extension had only arithmetic transcripts, so the 5B base
answers science questions with a sum). Here no chat format is used.

For each of the 100 factuality questions, in plain text "Question: {q}\nAnswer: {reference}":
  matched     = mean log-prob per token of the reference answer after its own question
  mismatched  = the same answer after 10 other questions (fixed random pick), averaged
  PMI         = matched − mismatched: how much the right question raises the probability of the right answer.
The answer text is identical on both sides, so general fluency (which improves with more training) cancels; what's left is
question → answer association, i.e. knowledge. Also reports raw matched NLL. Paired over questions, bootstrap 95% CI for B − A.

Usage (from COSF-MI/): .venv/bin/python experiments/knowledge_pmi.py A=CKPT B=CKPT [...] --out experiments/results/X/knowledge_pmi.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "python"))
from cosf_mi.data.tokenizer import BPETokenizer, tokenizer_for_vocab  # noqa: E402
from cosf_mi.models.cosf_oscillator import COSFOscillatorTransformer  # noqa: E402

QUESTIONS = ROOT / "experiments/factuality/questions.jsonl"


@torch.no_grad()
def answer_logprob(model, tok, pairs, device, bs=32):
    """Mean log-prob per answer token for each (question, answer) pair."""
    out = []
    for i in range(0, len(pairs), bs):
        seqs, spans = [], []
        for q, a in pairs[i:i + bs]:
            pre, ans = tok.encode(f"Question: {q}\nAnswer:"), tok.encode(" " + a)
            seqs.append(pre + ans)
            spans.append((len(pre), len(pre) + len(ans)))
        n = max(map(len, seqs))
        x = torch.zeros(len(seqs), n, dtype=torch.long)
        for j, s in enumerate(seqs):
            x[j, :len(s)] = torch.tensor(s)
        x = x.to(device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(x)
        lp = F.log_softmax(logits.float(), dim=-1)
        for j, (s, e) in enumerate(spans):
            tgt = x[j, s:e]
            out.append(lp[j, s - 1:e - 1].gather(-1, tgt.unsqueeze(-1)).mean().item())
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpts", nargs="+", help="NAME=path (repo-relative); the first is the reference for the paired differences")
    p.add_argument("--out", required=True)
    p.add_argument("--others", type=int, default=10)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    qs = [json.loads(line) for line in open(QUESTIONS)]
    rng = random.Random(0)
    others = {k: rng.sample([j for j in range(len(qs)) if j != k], args.others) for k in range(len(qs))}
    matched_pairs = [(q["question"], q["reference"]) for q in qs]
    mism_pairs = [(qs[j]["question"], q["reference"]) for k, q in enumerate(qs) for j in others[k]]
    res = {}
    for spec in args.ckpts:
        name, path = spec.split("=", 1)
        ck = torch.load(ROOT / path, map_location="cpu", weights_only=False)
        model = COSFOscillatorTransformer(ck["config"]).to(device).eval()
        model.load_state_dict(ck["model_state_dict"])
        tok = BPETokenizer(str(ROOT / tokenizer_for_vocab(ck["config"].vocab_size)))
        m = answer_logprob(model, tok, matched_pairs, device)
        mm = answer_logprob(model, tok, mism_pairs, device)
        mm = [sum(mm[k * args.others:(k + 1) * args.others]) / args.others for k in range(len(qs))]
        res[name] = {"checkpoint": path, "matched": m, "mismatched": mm, "pmi": [a - b for a, b in zip(m, mm)]}
        print(f"{name}: matched NLL/token {-sum(m) / len(m):.4f}, PMI {sum(res[name]['pmi']) / len(qs):+.4f} "
              f"(questions with PMI > 0: {sum(x > 0 for x in res[name]['pmi'])}/{len(qs)})", flush=True)
        del model
        torch.cuda.empty_cache()
    names = list(res)
    rng = random.Random(1)
    summary = {}
    for b in names[1:]:
        for key in ("pmi", "matched"):
            d = [y - x for x, y in zip(res[names[0]][key], res[b][key])]
            boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(10_000))
            summary[f"{b} − {names[0]}: {key}"] = {"mean": round(sum(d) / len(d), 4), "ci95": [round(boots[249], 4), round(boots[9749], 4)]}
            print(f"{b} − {names[0]} {key}: {sum(d) / len(d):+.4f}, 95% CI [{boots[249]:+.4f}, {boots[9749]:+.4f}]")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "per_model": res}, indent=1))


if __name__ == "__main__":
    main()
