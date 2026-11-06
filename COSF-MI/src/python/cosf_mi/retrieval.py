"""
Retrieval for AURA (2026-11-06; experiments/plans/retrieval_prereg.md).

A store is a directory with `passages.jsonl` (one JSON object per line: id, source, title, text) and `emb.npy` (fp16, [N, 384], L2-normalized
bge-small-en-v1.5 CLS embeddings, same row order). Search is a brute-force inner product in torch (GPU if available), in chunks.

- `Embedder`: BAAI/bge-small-en-v1.5 (MIT). Passages are embedded as-is; queries get the model's retrieval instruction.
- `Store.search(questions, k)`: top-k passages per question.
- `format_prompt(question, passages)`: the prompt format the RAG fine-tune and the eval use.
- `uses_wikipedia(answer, question, passages)`: the pre-registered "(Wikipedia)" rule: ≥ 3 distinct content words (≥ 4 letters, not
  stopwords, not in the question) of the answer appear in a retrieved Wikipedia passage.
"""
import json
import re
from pathlib import Path

import numpy as np
import torch

EMBEDDER = "BAAI/bge-small-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
WORD = re.compile(r"[a-z]+")
STOP = set("""about above after again against also among another because been before being below between both
could does doing down during each even every from further have having here into itself just like made make many
more most much must only other over same should some such than that their them then there these they this those
through under until very what when where which while will with would your yours were well really things thing""".split())


class Embedder:
    def __init__(self, device=None, max_len=128):
        from transformers import AutoModel, AutoTokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(EMBEDDER)
        self.model = AutoModel.from_pretrained(EMBEDDER).to(self.device).eval()
        if self.device == "cuda":
            self.model = self.model.half()
        self.max_len = max_len

    @torch.no_grad()
    def encode(self, texts, batch=512, query=False):
        out = []
        for i in range(0, len(texts), batch):
            chunk = [(QUERY_INSTRUCTION + t) if query else t for t in texts[i:i + batch]]
            enc = self.tok(chunk, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
            cls = self.model(**enc).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(cls.float(), dim=-1).half().cpu())
        return torch.cat(out) if out else torch.zeros(0, 384, dtype=torch.float16)


class Store:
    def __init__(self, path, device=None):
        self.path = Path(path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # passages are read from disk on demand through a byte-offset index (6M passages as Python dicts would take ~5 GB of RAM)
        off = self.path / "offsets.npy"
        if not off.exists() or off.stat().st_mtime < (self.path / "passages.jsonl").stat().st_mtime:
            offsets, pos = [], 0
            with open(self.path / "passages.jsonl", "rb") as f:
                for line in f:
                    offsets.append(pos)
                    pos += len(line)
            np.save(off, np.array(offsets, dtype=np.int64))
        self.offsets = np.load(off)
        self._fh = open(self.path / "passages.jsonl", "rb")
        emb = np.load(self.path / "emb.npy", mmap_mode="r")
        assert len(emb) == len(self.offsets), "emb.npy and passages.jsonl differ in length"
        self.emb = torch.from_numpy(np.ascontiguousarray(emb)).to(self.device)
        if self.device == "cpu":
            self.emb = self.emb.float()  # fp16 matmul is slow or missing on CPU
        self._embedder = None

    def __len__(self):
        return len(self.offsets)

    def get(self, i):
        self._fh.seek(int(self.offsets[i]))
        return json.loads(self._fh.readline())

    def embedder(self):
        if self._embedder is None:
            self._embedder = Embedder(self.device)
        return self._embedder

    @torch.no_grad()
    def search(self, questions, k=3, q_batch=1024, chunk=2_000_000):
        q = self.embedder().encode(questions, query=True).to(self.device, self.emb.dtype)
        results = []
        for i in range(0, len(q), q_batch):
            qb = q[i:i + q_batch]
            best_s, best_i = None, None
            for j in range(0, len(self.emb), chunk):
                s = qb @ self.emb[j:j + chunk].t()
                ts, ti = s.topk(min(k, s.shape[1]), dim=1)
                ti = ti + j
                if best_s is None:
                    best_s, best_i = ts, ti
                else:
                    cs, ci = torch.cat([best_s, ts], 1), torch.cat([best_i, ti], 1)
                    best_s, idx = cs.topk(k, dim=1)
                    best_i = ci.gather(1, idx)
            for row_s, row_i in zip(best_s.tolist(), best_i.tolist()):
                results.append([dict(self.get(ix), score=round(sc, 4)) for sc, ix in zip(row_s, row_i)])
        return results


def format_prompt(question, passages, max_words=75):
    """The RAG prompt (training and eval use the same one). Each passage is trimmed to max_words so 3 sources + question + answer fit in
    AURA's 512 tokens."""
    if not passages:
        return question
    return question + "\n\nSources:\n" + "\n".join(f"[{n}] {' '.join(p['text'].split()[:max_words])}" for n, p in enumerate(passages, 1))


def content_words(text):
    return {w for w in WORD.findall(text.lower()) if len(w) >= 4 and w not in STOP}


def uses_wikipedia(answer, question, passages, min_words=3):
    a = content_words(answer) - content_words(question)
    return any(p.get("source") == "wikipedia" and len(a & content_words(p["text"])) >= min_words for p in passages)


def tag_answer(answer, question, passages):
    """The shown answer: ' (Wikipedia)' appended when it uses a Wikipedia passage (pre-registered rule)."""
    return answer.rstrip() + " (Wikipedia)" if uses_wikipedia(answer, question, passages) else answer
