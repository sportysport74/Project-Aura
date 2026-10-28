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
    """(train, val, meta) for a tokenized corpus directory."""
    d = Path(corpus_dir)
    meta = json.loads((d / "meta.json").read_text())
    return TokenMemmapDataset(d / "train.bin", seq_len), TokenMemmapDataset(d / "val.bin", seq_len), meta
