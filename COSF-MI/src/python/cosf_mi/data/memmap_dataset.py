"""
Token dataset backed by a uint16 memmap on disk, for corpora too big for RAM.

token_dataset_fast.py keeps the whole token cache in memory as int64 (8 bytes/token): fine for the 118M-token styled
mix, but 2.2B tokens would need 17.6 GB of the 24 GB WSL limit. Here the tokens stay on disk as uint16 (2 bytes, vocab
must be < 65,536) and each item is read on demand. Files come from scripts/corpus/tokenize_corpus.py:
  <dir>/train.bin, <dir>/val.bin   flat uint16 token streams (documents joined with </s>)
  <dir>/meta.json                  tokenizer path, vocab size, token counts, sources
"""
import json
from pathlib import Path

import numpy as np
import torch


class TokenMemmapDataset(torch.utils.data.Dataset):
    """Non-overlapping windows of seq_len + 1 tokens: item i -> (x, y) with y = x shifted by one."""

    def __init__(self, path, seq_len):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.n = (len(self.tokens) - 1) // seq_len

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        start = i * self.seq_len
        chunk = torch.from_numpy(self.tokens[start:start + self.seq_len + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]


def load_corpus(corpus_dir, seq_len):
    """(train, val, meta) for a tokenized corpus directory, or several joined with commas ("a,b,c"; 2026-11-07, Aura v3 plan).
    Several corpora: the train sets are concatenated (windows never cross a file boundary), the val split is the first corpus's,
    and meta is the first corpus's with train_tokens summed and a "parts" list. All parts must share the tokenizer."""
    dirs = [Path(x) for x in str(corpus_dir).split(",") if x]
    metas = [json.loads((d / "meta.json").read_text()) for d in dirs]
    assert all(m["tokenizer"] == metas[0]["tokenizer"] for m in metas), "corpora with different tokenizers"
    trains = [TokenMemmapDataset(d / "train.bin", seq_len) for d in dirs]
    meta = dict(metas[0])
    if len(dirs) > 1:
        meta["train_tokens"] = sum(m["train_tokens"] for m in metas)
        meta["parts"] = [{"dir": str(d), "train_tokens": m["train_tokens"]} for d, m in zip(dirs, metas)]
    train = trains[0] if len(trains) == 1 else torch.utils.data.ConcatDataset(trains)
    return train, TokenMemmapDataset(dirs[0] / "val.bin", seq_len), meta
