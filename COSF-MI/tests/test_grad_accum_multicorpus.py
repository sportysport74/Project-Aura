"""Gradient accumulation (train_overnight.py --grad-accum) and the multi-corpus memmap loader (--corpus a,b), added 2026-11-07 for the Aura v3
plan."""
import json

import numpy as np
import torch

from cosf_mi.data.memmap_dataset import load_corpus
from cosf_mi.models.cosf_oscillator import COSFOscillatorConfig, COSFOscillatorTransformer


def test_accumulated_grads_equal_full_batch():
    """Micro-batch means / grad_accum, summed over equal chunks, give the full-batch gradient (dropout off, float64)."""
    torch.manual_seed(0)
    cfg = COSFOscillatorConfig(vocab_size=500, d_model=64, n_layers=2, n_heads=4, d_ff=256, max_seq_len=32, projection="linear",
                               emb_scale="sqrt", ffn_activation="gelu", linear_init="gpt")
    m = COSFOscillatorTransformer(cfg).double().eval()
    x, y = torch.randint(0, 500, (8, 32)), torch.randint(0, 500, (8, 32))

    def grads(k):
        m.zero_grad()
        for xi, yi in zip(x.chunk(k), y.chunk(k)):
            _, tl, sl, _ = m(xi, yi, return_coherence_loss=True)
            ((tl + sl) / k).backward()
        return torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None])

    g1 = grads(1)
    for k in (2, 4):
        assert torch.allclose(grads(k), g1, rtol=0, atol=1e-12 * g1.abs().max().item())


def _corpus(d, tokens, tok="tok.model"):
    d.mkdir()
    np.asarray(tokens, dtype=np.uint16).tofile(d / "train.bin")
    np.asarray(tokens[:9], dtype=np.uint16).tofile(d / "val.bin")
    (d / "meta.json").write_text(json.dumps({"tokenizer": tok, "train_tokens": len(tokens), "val_tokens": 9}))


def test_multi_corpus_concatenates_without_crossing_files(tmp_path):
    _corpus(tmp_path / "a", list(range(0, 17)))       # (17 - 1) // 4 = 4 windows
    _corpus(tmp_path / "b", list(range(100, 109)))    # (9 - 1) // 4 = 2 windows
    train, val, meta = load_corpus(f"{tmp_path / 'a'},{tmp_path / 'b'}", 4)
    assert len(train) == 6 and meta["train_tokens"] == 26 and len(meta["parts"]) == 2
    x4, y4 = train[4]  # first window of b: starts at b's first token, never mixes in a's tail
    assert x4.tolist() == [100, 101, 102, 103] and y4.tolist() == [101, 102, 103, 104]
    assert val[0][0].tolist() == [0, 1, 2, 3]  # val comes from the first corpus
    single, _, meta1 = load_corpus(str(tmp_path / "a"), 4)
    assert len(single) == 4 and "parts" not in meta1
