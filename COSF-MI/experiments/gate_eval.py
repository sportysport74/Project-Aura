"""
Retrieval gate eval (pre-registered: experiments/plans/retrieval_gate.md). Project venv. Phases, in chain order:

  signals     g / c / s for every message in data/retrieval/gate (minus the dropped "realtime" generator) and the 100 factuality questions;
              saves the chit-chat tune embeddings (the c signal's reference set) → <gate>/chit_emb.pt, <res>/signals.jsonl
  thresholds  gate 0 AUROCs and the thresholds, from the TUNE split only → <gate>/thresholds.json (+ a copy in <res>)
  gen         replies on the off-topic TEST messages + 10 on-topic test messages (chat-quality set): RAG s0/s1 with the top-3 corpus
              sources, and b10_s1 without sources; joined encoding, sampled (top-p 0.9, T 0.7, penalty 1.15 excluding the sources), seeds 1-3
              → <res>/replies.jsonl
  multiturn   M0 (follow-up alone, top-3 sources) vs M1 (turn 1 + AURA's own turn-1 answer as history, query = turn 1 + follow-up, k=2 at
              60 words); RAG s0/s1, seeds 1-3 → <res>/multiturn/{replies,questions}.jsonl (graded by factuality_grade.py --questions)
  entropy     mean raw-token entropy of the re-score's answers (R2c_s0/s1 and R0, joined, sampled) by teacher forcing
              → <res>/entropy.jsonl (diagnostic only)

Usage (from COSF-MI/): PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u experiments/gate_eval.py <phase> [--smoke]
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "python"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))
import ab_eval  # noqa: E402
import finetune_chat as ft  # noqa: E402
from cosf_mi.chat_gate import answer_entropy, message_nll  # noqa: E402
from cosf_mi.data.tokenizer import BPETokenizer, tokenizer_for_vocab  # noqa: E402
from cosf_mi.models.cosf_oscillator import COSFOscillatorTransformer  # noqa: E402
from cosf_mi.retrieval import format_prompt  # noqa: E402

DATA = ROOT / "data/retrieval/gate"
STORE = ROOT / "data/retrieval/corpus"
B10 = "checkpoints/v2base/ft/b10_s1/best_model.pt"
RAG = {"RAG_s0": "checkpoints/rag/s0/best_model.pt", "RAG_s1": "checkpoints/rag/s1/best_model.pt"}
KINDS = ("ontopic", "chitchat", "unanswerable", "garbled")
DROP_SUBS = {"realtime"}  # dropped generator (data_review.md, pre-registered rule)
SAMPLED = dict(greedy=False, top_k=0, top_p=0.9, rep_penalty=1.15)
SEEDS = (1, 2, 3)


def load_messages():
    rows = []
    for k in KINDS:
        rows += [r for r in map(json.loads, open(DATA / f"{k}.jsonl")) if r["sub"] not in DROP_SUBS]
    return rows


def load_model(path):
    ckpt = torch.load(ROOT / path, map_location="cpu", weights_only=False)
    model = COSFOscillatorTransformer(ckpt["config"]).to("cuda")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    ab_eval.SEQ = ft.SEQ = ckpt["config"].max_seq_len
    return model, BPETokenizer(str(ROOT / tokenizer_for_vocab(ckpt["config"].vocab_size))), ckpt["config"].max_seq_len


class Passages:
    """Store passages by id, from disk (no embeddings loaded)."""
    def __init__(self, path):
        self.off = np.load(path / "offsets.npy")
        self.fh = open(path / "passages.jsonl", "rb")

    def get(self, i):
        self.fh.seek(int(self.off[i]))
        return json.loads(self.fh.readline())


def auroc(pos, neg):
    """P(score_pos > score_neg) + 0.5 P(tie)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1
    for v in np.unique(allv):  # average ranks for ties
        m = allv == v
        ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def order_stat(values, q):
    """The smallest v such that at least a fraction q of values are <= v."""
    v = sorted(x for x in values if not np.isnan(x))
    return v[max(0, int(np.ceil(q * len(v))) - 1)]


def cmd_signals(args):
    from cosf_mi.retrieval import Store
    msgs = load_messages()
    fq = [json.loads(line) for line in open(ROOT / "experiments/factuality/questions.jsonl")]
    items = msgs + [{"id": q["id"], "kind": "factuality", "sub": q["topic"], "split": "eval", "text": q["question"]} for q in fq]
    if args.smoke:
        rng = random.Random(0)
        by = {}
        for r in items:
            by.setdefault((r["kind"], r["split"]), []).append(r)
        items = [r for v in by.values() for r in rng.sample(v, min(6, len(v)))]
    texts = [r["text"] for r in items]
    model, tok, seq = load_model(B10)
    g = message_nll(model, tok, texts, seq)
    del model
    torch.cuda.empty_cache()
    st = Store(STORE)
    emb = st.embedder().encode(texts, query=True).float()
    chit_idx = [i for i, r in enumerate(items) if r["kind"] == "chitchat" and r["split"] == "tune"]
    chit = emb[chit_idx]
    sim = emb @ chit.t()
    for col, i in enumerate(chit_idx):  # leave one out: a tune chit-chat message isn't compared with itself
        sim[i, col] = -1.0
    c = sim.max(1).values.tolist()
    srcs = st.search(texts, k=3)
    args.gate.mkdir(parents=True, exist_ok=True)
    torch.save(chit.half(), args.gate / "chit_emb.pt")
    args.res.mkdir(parents=True, exist_ok=True)
    with open(args.res / "signals.jsonl", "w") as f:
        for r, gi, ci, ps in zip(items, g, c, srcs):
            f.write(json.dumps({**r, "g": round(gi, 4), "c": round(ci, 4), "s": ps[0]["score"],
                                "sources": [{"id": p["id"], "score": p["score"]} for p in ps]}, ensure_ascii=False) + "\n")
    print(f"signals for {len(items):,} messages -> {args.res / 'signals.jsonl'}")


def cmd_thresholds(args):
    rows = [r for r in map(json.loads, open(args.res / "signals.jsonl")) if r["split"] == "tune"]
    by = {k: [r for r in rows if r["kind"] == k] for k in KINDS}
    clear = by["ontopic"] + by["chitchat"] + by["unanswerable"]
    au = {"g": auroc([r["g"] for r in by["garbled"]], [r["g"] for r in clear]),
          "c": auroc([r["c"] for r in by["chitchat"]], [r["c"] for r in by["ontopic"]]),
          "s": auroc([-r["s"] for r in by["unanswerable"]], [-r["s"] for r in by["ontopic"]])}
    th = {"g": max(order_stat([r["g"] for r in by[k]], 0.95) for k in ("ontopic", "chitchat", "unanswerable")),
          "c": order_stat([r["c"] for r in by["ontopic"]], 0.95),
          "s": -order_stat([-r["s"] for r in by["ontopic"]], 0.95)}
    used = {k: (th[k] if au[k] >= 0.85 else None) for k in th}
    out = {"thresholds": used, "candidate_thresholds": th, "auroc_tune": au, "gate0_min_auroc": 0.85,
           "n_tune": {k: len(v) for k, v in by.items()},
           "rules": {"g": "lowest value passing >= 95% of each clear tune group (ontopic, chitchat, unanswerable)",
                     "c": "passes >= 95% of on-topic tune as not chit-chat", "s": "passes >= 95% of on-topic tune"}}
    args.gate.mkdir(parents=True, exist_ok=True)
    (args.gate / "thresholds.json").write_text(json.dumps(out, indent=1))
    (args.res / "thresholds.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def test_messages(args):
    sig = [r for r in map(json.loads, open(args.res / "signals.jsonl")) if r["split"] == "test"]
    off = [r for r in sig if r["kind"] != "ontopic"]
    ont = [r for r in sig if r["kind"] == "ontopic"]
    on = random.Random(0).sample(ont, min(10, len(ont)))  # the chat-quality set's on-topic part
    msgs = off + on
    if args.smoke:
        msgs = msgs[:4]
    return msgs


def cmd_gen(args):
    msgs = test_messages(args)
    pas = Passages(STORE)
    srcs = [[pas.get(s["id"]) for s in r["sources"]] for r in msgs]
    with open(args.res / "replies.jsonl", "w") as f:
        for name, path in [("B10", B10)] + list(RAG.items()):
            model, tok, _ = load_model(path)
            rag = name != "B10"
            prompts = [format_prompt(r["text"], ps) if rag else r["text"] for r, ps in zip(msgs, srcs)]
            rep_ids = [tok.encode(f"{ft.USER} {r['text']}\n{ft.AURA}") for r in msgs] if rag else None
            for seed in SEEDS[:1] if args.smoke else SEEDS:
                replies = ab_eval.generate(model, tok, prompts, max_new=160, bs=20, joined=True, seed=seed, rep_ids=rep_ids, **SAMPLED)
                for r, (reply, ended) in zip(msgs, replies):
                    f.write(json.dumps({"model": name, "id": r["id"], "kind": r["kind"], "sub": r["sub"], "seed": seed, "text": r["text"],
                                        "reply": reply, "ended": ended}, ensure_ascii=False) + "\n")
                print(f"[{name}] seed {seed}: {len(msgs)} replies", flush=True)
            del model
            torch.cuda.empty_cache()


def cmd_multiturn(args):
    from cosf_mi.retrieval import Store
    convs = [json.loads(line) for line in open(DATA / "multiturn.jsonl")]
    if args.smoke:
        convs = convs[:3]
    st = Store(STORE)
    s_f = st.search([c["followup"] for c in convs], k=3)            # M0: the follow-up alone
    s_1 = st.search([c["turn1"] for c in convs], k=3)               # turn 1 (to get AURA's own turn-1 answer)
    s_h = st.search([c["turn1"] + " " + c["followup"] for c in convs], k=2)  # M1: query = turn 1 + follow-up, k = 2
    del st
    torch.cuda.empty_cache()
    out = args.res / "multiturn"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "questions.jsonl", "w") as f:
        for c in convs:
            f.write(json.dumps({"id": c["id"], "topic": c["topic"], "question": f"Turn 1: {c['turn1']} / Follow-up: {c['followup']}",
                                "reference": c["reference"], "key_facts": c["key_facts"]}, ensure_ascii=False) + "\n")
    trunc = 0
    with open(out / "replies.jsonl", "w") as f:
        for name, path in RAG.items():
            model, tok, seq = load_model(path)
            for seed in SEEDS[:1] if args.smoke else SEEDS:
                p0 = [format_prompt(c["followup"], ps) for c, ps in zip(convs, s_f)]
                r0 = ab_eval.generate(model, tok, p0, max_new=160, bs=20, joined=True, seed=seed,
                                      rep_ids=[tok.encode(f"{ft.USER} {c['followup']}\n{ft.AURA}") for c in convs], **SAMPLED)
                p1 = [format_prompt(c["turn1"], ps) for c, ps in zip(convs, s_1)]
                a1 = ab_eval.generate(model, tok, p1, max_new=160, bs=20, joined=True, seed=seed,
                                      rep_ids=[tok.encode(f"{ft.USER} {c['turn1']}\n{ft.AURA}") for c in convs], **SAMPLED)
                ph = [f"{c['turn1']}\n{ft.AURA} {a}\n{ft.USER} {format_prompt(c['followup'], ps, max_words=60)}"
                      for c, (a, _), ps in zip(convs, a1, s_h)]
                trunc += sum(len(tok.encode(f"{ft.USER} {p}\n{ft.AURA}")) > seq - 8 for p in ph)
                rh = ab_eval.generate(model, tok, ph, max_new=160, bs=20, joined=True, seed=seed,
                                      rep_ids=[tok.encode(f"{ft.USER} {c['turn1']}\n{ft.AURA} {a}\n{ft.USER} {c['followup']}\n{ft.AURA}")
                                               for c, (a, _) in zip(convs, a1)], **SAMPLED)
                for arm, replies in (("M0", r0), ("M1", rh)):
                    for c, (a, _), (reply, ended) in zip(convs, a1, replies):
                        f.write(json.dumps({"model": f"{arm}_{name[-2:]}", "checkpoint": path, "encoding": "joined", "decoding": "sampled_rep",
                                            "seed": seed, "id": c["id"], "reply": reply, "ended": ended,
                                            "turn1_answer": a if arm == "M1" else None}, ensure_ascii=False) + "\n")
                print(f"[{name}] seed {seed}: done", flush=True)
            del model
            torch.cuda.empty_cache()
    (out / "notes.json").write_text(json.dumps({"m1_contexts_cut_at_the_window": trunc}, indent=1))
    print(f"M1 contexts longer than the window (cut from the left): {trunc}")


def cmd_entropy(args):
    rs = ROOT / "experiments/results/retrieval/factuality_rescore"
    fq = {q["id"]: q for q in map(json.loads, open(ROOT / "experiments/factuality/questions.jsonl"))}
    rows = [r for r in map(json.loads, open(rs / "replies.jsonl"))
            if r["model"] in ("R2c_s0", "R2c_s1", "R0") and r["encoding"] == "joined" and r["decoding"] == "sampled_rep"]
    if args.smoke:
        rows = rows[:10]
    pas = Passages(STORE)
    with open(args.res / "entropy.jsonl", "w") as f:
        for path, models in [(B10, ("R0",)), (RAG["RAG_s0"], ("R2c_s0",)), (RAG["RAG_s1"], ("R2c_s1",))]:
            sub = [r for r in rows if r["model"] in models]
            if not sub:
                continue
            model, tok, seq = load_model(path)
            ctx, ans = [], []
            for r in sub:
                q = fq[r["id"]]["question"]
                p = format_prompt(q, [pas.get(s["id"]) for s in r["sources"]]) if r["sources"] else q
                ctx.append(tok.encode(f"{ft.USER} {p}\n{ft.AURA}")[-(seq - 8):])
                ans.append(tok.encode(r["reply"]))
            ent = answer_entropy(model, ctx, ans, seq)
            for r, e in zip(sub, ent):
                f.write(json.dumps({k: r[k] for k in ("model", "encoding", "decoding", "seed", "id")} | {"entropy": round(e, 4)}) + "\n")
            print(f"entropy for {len(sub)} {models} answers", flush=True)
            del model
            torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("phase", choices=["signals", "thresholds", "gen", "multiturn", "entropy"])
    p.add_argument("--res", default="experiments/results/retrieval/gate")
    p.add_argument("--gate", default="data/retrieval/gate/model", help="where the chat gate reads thresholds.json + chit_emb.pt")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    args.res, args.gate = ROOT / args.res, ROOT / args.gate
    args.res.mkdir(parents=True, exist_ok=True)
    {"signals": cmd_signals, "thresholds": cmd_thresholds, "gen": cmd_gen, "multiturn": cmd_multiturn, "entropy": cmd_entropy}[args.phase](args)


if __name__ == "__main__":
    main()
