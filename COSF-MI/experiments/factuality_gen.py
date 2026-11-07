"""
Factuality eval, step 1 of 2: AURA replies to the 100 explain questions in experiments/factuality/questions.jsonl.

Per checkpoint, both prompt encodings (ab_eval.generate: "pieces" as finetune_chat.py trains, "joined" as the pretraining
corpus writes its chat slot) and two decodings: greedy (1 reply) and sampled_rep (top-p 0.9, repetition penalty 1.15,
--samples replies with seeds 1..n, the readable-set setting). Replies are capped at 160 tokens. Writes
<out>/replies.jsonl; grade it with experiments/factuality_grade.py (vLLM venv).

Usage (from COSF-MI/):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u experiments/factuality_gen.py \
    base=checkpoints/pretrain/l100_on_s1/best_model.pt ft_s0=checkpoints/ft_l100_s0/best_model.pt --out experiments/results/factuality
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "python"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))
import ab_eval  # noqa: E402
import finetune_chat as ft  # noqa: E402
from cosf_mi.data.tokenizer import BPETokenizer, tokenizer_for_vocab  # noqa: E402
from cosf_mi.models.cosf_oscillator import COSFOscillatorTransformer  # noqa: E402

QUESTIONS = ROOT / "experiments/factuality/questions.jsonl"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpts", nargs="+", help="NAME=path/to/checkpoint.pt (repo-relative)")
    p.add_argument("--out", default="experiments/results/factuality")
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--max-new", type=int, default=160)
    p.add_argument("--limit", type=int, default=0, help="only the first N questions (dry runs; 0 = all)")
    p.add_argument("--store", default=None, help="retrieval store dir (data/retrieval/corpus or .../wiki): top-k sources in the prompt")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--oracle", action="store_true", help="the question's reference answer as source [1] (upper bound)")
    p.add_argument("--rep-scope", default="all", choices=["all", "no_sources"],
                   help="no_sources: the repetition penalty skips tokens that appear only in the retrieved sources "
                        "(experiments/plans/retrieval_rescore.md); all = the original whole-context penalty")
    p.add_argument("--append", action="store_true", help="append to replies.jsonl instead of overwriting (several arms, one grading run)")
    args = p.parse_args()
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    qs = [json.loads(line) for line in open(QUESTIONS)]
    qs = qs[:args.limit] if args.limit else qs
    sources = [[] for _ in qs]
    if args.store or args.oracle:
        from cosf_mi.retrieval import Store, format_prompt
        if args.store:
            st = Store(ROOT / args.store)
            sources = st.search([q["question"] for q in qs], k=args.k)
            del st
            torch.cuda.empty_cache()
        if args.oracle:
            sources = [[{"id": -1, "source": "oracle", "title": "", "text": q["reference"]}] for q in qs]
        prompts = [format_prompt(q["question"], ps) for q, ps in zip(qs, sources)]
    else:
        prompts = [q["question"] for q in qs]
    from cosf_mi.retrieval import tag_answer

    def copies(reply, ps, n=8):  # does the answer copy >= n consecutive words of one of its sources?
        w = re.findall(r"[a-z0-9']+", reply.lower())
        grams = {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}
        for x in ps:
            xw = re.findall(r"[a-z0-9']+", x["text"].lower())
            if grams & {" ".join(xw[i:i + n]) for i in range(len(xw) - n + 1)}:
                return True
        return False
    with open(out_dir / "replies.jsonl", "a" if args.append else "w") as f:
        for spec in args.ckpts:
            name, path = spec.split("=", 1)
            ckpt = torch.load(ROOT / path, map_location="cpu", weights_only=False)
            model = COSFOscillatorTransformer(ckpt["config"]).to("cuda")
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            ab_eval.SEQ = ft.SEQ = ckpt["config"].max_seq_len
            tok = BPETokenizer(str(ROOT / tokenizer_for_vocab(ckpt["config"].vocab_size)))
            for enc in ("pieces", "joined"):
                runs = [("greedy", 0, dict(greedy=True))] + [
                    ("sampled_rep", s, dict(greedy=False, seed=s, top_k=0, top_p=0.9, rep_penalty=1.15))
                    for s in range(1, args.samples + 1)]
                for mode, seed, kw in runs:
                    rep_ids = None
                    if args.rep_scope == "no_sources" and kw.get("rep_penalty", 1.0) != 1.0:
                        # penalize the template + question tokens (+ the reply), not tokens that appear only in the sources
                        rep_ids = [None if not ps else (tok.encode(f"{ft.USER} {q['question']}\n{ft.AURA}") if enc == "joined" else
                                                        tok.encode(ft.USER) + tok.encode(q["question"]) + tok.encode(ft.AURA))
                                   for q, ps in zip(qs, sources)]
                    replies = ab_eval.generate(model, tok, prompts, max_new=args.max_new, bs=20, joined=enc == "joined",
                                               rep_ids=rep_ids, **kw)
                    for q, ps, (reply, ended) in zip(qs, sources, replies):
                        # "reply" is what the grader sees; "reply_shown" carries the "(Wikipedia)" tag (pre-registered rule)
                        f.write(json.dumps({"model": name, "checkpoint": path, "encoding": enc, "decoding": mode,
                                            "seed": seed, "id": q["id"], "reply": reply, "ended": ended,
                                            "reply_shown": tag_answer(reply, q["question"], ps),
                                            "copies8": copies(reply, ps),
                                            "sources": [{"id": x["id"], "source": x["source"], "title": x.get("title", "")} for x in ps]},
                                           ensure_ascii=False) + "\n")
                    f.flush()
                    print(f"[{name}] {enc} {mode} seed {seed}: done", flush=True)
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
