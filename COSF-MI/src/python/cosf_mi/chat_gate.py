"""
Retrieval gate for chat (pre-registered in experiments/plans/retrieval_gate.md, 2026-11-07).

Three signals per message, then a fixed routing order:
  g  mean per-token NLL of the message under the no-source chat model (b10_s1), in "You: {msg}" (message tokens only)  → high = garbled
  c  max bge cosine similarity to the chit-chat tune examples                                                         → high = chit-chat
  s  top-1 bge similarity of the retrieved store passage                                                              → low = no good sources
Routing: g > t_g → "unclear"; c > t_c → "chitchat"; s < t_s → "nosources"; else "rag". A signal whose threshold is None (it failed gate 0's
AUROC ≥ 0.85 on the tune split) is never used. Thresholds and the chit-chat examples' embeddings live in a thresholds directory written by
experiments/gate_eval.py (thresholds.json + chit_emb.pt).

Also `answer_entropy`: mean entropy of the raw next-token distribution over an answer (diagnostic only, never a gate).
"""
import json
from pathlib import Path

import torch

TEMPLATE_UNCLEAR = "I'm not sure what you mean, can you say it another way?"
TEMPLATE_NOSOURCES = "I don't have good sources on that, so I'd rather not guess."
USER, AURA = "You:", "Aura:"


@torch.no_grad()
def message_nll(model, tok, messages, seq, bs=64):
    """Mean NLL (nats/token) of each message's own tokens after the "You:" prefix (the joined chat encoding)."""
    dev = next(model.parameters()).device
    pre = len(tok.encode(USER))
    out = []
    for i in range(0, len(messages), bs):
        ids = [tok.encode(f"{USER} {m}")[:seq] for m in messages[i:i + bs]]
        T = max(len(x) for x in ids)
        buf = torch.zeros(len(ids), T, dtype=torch.long, device=dev)
        for j, x in enumerate(ids):
            buf[j, :len(x)] = torch.tensor(x)
        logp = torch.log_softmax(model(buf).float(), -1)
        for j, x in enumerate(ids):
            if len(x) <= pre:
                out.append(float("nan"))
                continue
            tgt = torch.tensor(x[pre:], device=dev)
            lp = logp[j, pre - 1:len(x) - 1].gather(1, tgt[:, None])[:, 0]
            out.append(float(-lp.mean()))
    return out


@torch.no_grad()
def answer_entropy(model, contexts, answers, seq, bs=32):
    """Mean entropy (nats) of the model's raw next-token distribution (T = 1, no penalty or truncation) over each answer's tokens, by
    teacher forcing: contexts and answers are token-id lists. Answers cut by the window are scored on the part that fits."""
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(contexts), bs):
        rows = [(c, a) for c, a in zip(contexts[i:i + bs], answers[i:i + bs])]
        seqs = [(c + a)[:seq] for c, a in rows]
        T = max(len(x) for x in seqs)
        buf = torch.zeros(len(seqs), T, dtype=torch.long, device=dev)
        for j, x in enumerate(seqs):
            buf[j, :len(x)] = torch.tensor(x)
        logits = model(buf).float()
        for j, (c, a) in enumerate(rows):
            lo, hi = len(c) - 1, min(len(c) + len(a), len(seqs[j])) - 1
            if hi <= lo:
                out.append(float("nan"))
                continue
            lp = torch.log_softmax(logits[j, lo:hi], -1)
            out.append(float(-(lp.exp() * lp).sum(-1).mean()))
    return out


class Gate:
    def __init__(self, gate_dir, store, nll_model, tok, seq):
        gate_dir = Path(gate_dir)
        self.th = json.loads((gate_dir / "thresholds.json").read_text())["thresholds"]
        self.chit = torch.load(gate_dir / "chit_emb.pt").to(store.device).float()
        self.store, self.nll_model, self.tok, self.seq = store, nll_model, tok, seq

    def signals(self, messages, k=3):
        """[{g, c, s, sources}] per message (sources = the store's top-k, reused by the RAG branch)."""
        g = message_nll(self.nll_model, self.tok, messages, self.seq)
        q = self.store.embedder().encode(messages, query=True).to(self.chit.device).float()
        c = (q @ self.chit.t()).max(1).values.tolist()
        srcs = self.store.search(messages, k=k)
        return [{"g": gi, "c": ci, "s": (ps[0]["score"] if ps else 0.0), "sources": ps} for gi, ci, ps in zip(g, c, srcs)]

    def route(self, sig):
        return route(sig, self.th)


def route(sig, th):
    if th.get("g") is not None and sig["g"] > th["g"]:
        return "unclear"
    if th.get("c") is not None and sig["c"] > th["c"]:
        return "chitchat"
    if th.get("s") is not None and sig["s"] < th["s"]:
        return "nosources"
    return "rag"
