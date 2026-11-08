"""
Arithmetic-only token stream for the settling retest (2026-11-08, experiments/plans/settling_retest.md).

Reads a build_arith_plain.py output dir (train.jsonl, val.jsonl, report.json), writes each problem as the corpus's chat-format
transcript ("You: {prompt}\nAura: {reply}", the same conversion as build_corpus.py's aura_arith source) with </s> between problems,
and tokenizes it into the corpus shard format: {train,val}.bin (uint16) + meta.json, loadable with
cosf_mi.data.memmap_dataset.load_corpus (train_overnight.py --corpus DIR).

Usage (from COSF-MI/):
  .venv/bin/python scripts/corpus/build_arith_stream.py --src data/arith_settle/src --out data/arith_settle/tok24k
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm

ROOT = Path(__file__).resolve().parents[2]
_INST = re.compile(r"\[INST\]\s*(.*?)\s*\[/INST\]\s*(.*)", re.S)


def transcripts(path):
    for line in open(path, encoding="utf-8"):
        m = _INST.match(json.loads(line)["line"])
        assert m, line[:200]
        yield f"You: {m.group(1)}\nAura: {m.group(2)}"


def write(sp, path_in, path_out, chunk=200_000):
    n_tok = n_doc = 0
    with open(path_out, "wb") as f:
        buf = []
        for doc in transcripts(path_in):
            buf.append(doc)
            if len(buf) == chunk:
                n_tok, n_doc = _flush(sp, buf, f, n_tok, n_doc)
                buf = []
        if buf:
            n_tok, n_doc = _flush(sp, buf, f, n_tok, n_doc)
    return n_tok, n_doc


def _flush(sp, docs, f, n_tok, n_doc):
    ids = sp.encode(docs)
    flat = np.fromiter((t for d in ids for t in (*d, sp.eos_id())), dtype=np.uint16)
    flat.tofile(f)
    return n_tok + len(flat), n_doc + len(docs)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="build_arith_plain.py output dir (repo-relative)")
    p.add_argument("--out", required=True)
    p.add_argument("--tokenizer", default="models/tokenizer_clean24k.model")
    args = p.parse_args()
    t0 = time.time()
    sp = spm.SentencePieceProcessor(model_file=str(ROOT / args.tokenizer))
    assert sp.vocab_size() < 65536
    src, out = ROOT / args.src, ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    tr = write(sp, src / "train.jsonl", out / "train.bin")
    va = write(sp, src / "val.jsonl", out / "val.bin")
    meta = {"tokenizer": args.tokenizer, "vocab_size": sp.vocab_size(), "eos_id": sp.eos_id(), "dtype": "uint16",
            "train_tokens": tr[0], "val_tokens": va[0], "train_docs": tr[1], "val_docs": va[1],
            "sources": {"aura_arith_settle": {"src": args.src, "licence": "ours (programmatic arithmetic, no teacher)",
                                              "report": json.loads((src / "report.json").read_text())}},
            "format": "You: {prompt}\\nAura: {reply} </s>", "built": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(f"train {tr[0] / 1e6:.1f}M tokens / {tr[1]:,} problems; val {va[0] / 1e6:.2f}M / {va[1]:,} [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
