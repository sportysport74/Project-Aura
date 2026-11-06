"""
Embed a store's passages with BAAI/bge-small-en-v1.5 (MIT): CLS, L2-normalized, fp16 → <store>/emb.npy (same row order as passages.jsonl).
Streams the passages in blocks, so RAM stays small. GPU if available.

Usage (from COSF-MI/): .venv/bin/python scripts/retrieval/embed_store.py data/retrieval/corpus
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "python"))
from cosf_mi.retrieval import Embedder  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("store")
    p.add_argument("--block", type=int, default=200_000)
    args = p.parse_args()
    d = ROOT / args.store
    n = sum(1 for _ in open(d / "passages.jsonl"))
    emb = np.lib.format.open_memmap(d / "emb.npy.tmp", mode="w+", dtype=np.float16, shape=(n, 384))
    e = Embedder()
    t0, i, buf = time.time(), 0, []

    def flush(buf, i, emb=emb):
        v = e.encode(buf).numpy()
        emb[i:i + len(v)] = v
        return i + len(v)
    for line in open(d / "passages.jsonl"):
        buf.append(json.loads(line)["text"])
        if len(buf) == args.block:
            i = flush(buf, i)
            buf = []
            print(f"  {i:,}/{n:,} ({i / (time.time() - t0):,.0f} passages/s)", flush=True)
    if buf:
        i = flush(buf, i)
    emb.flush()
    (d / "emb.npy.tmp").rename(d / "emb.npy")
    print(f"embedded {i:,} passages in {time.time() - t0:.0f} s -> {d / 'emb.npy'}")


if __name__ == "__main__":
    main()
