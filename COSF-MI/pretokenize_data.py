"""
Pre-tokenize TinyStories data for faster training startup.

Run this once to tokenize and cache the data, then training will start instantly.

Usage:
    python pretokenize_data.py
"""

import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from src.python.cosf_mi.data.token_dataset_fast import FastTokenDataset
from src.python.cosf_mi.data.tokenizer import Tokenizer, BPETokenizer


def pretokenize():
    """Pre-tokenize TinyStories data."""
    print("=" * 70)
    print("PRE-TOKENIZING TINYSTORIES DATA")
    print("=" * 70)
    print()
    print("This will tokenize train.txt and val.txt and cache them to disk.")
    print("After this completes, training will start instantly!")
    print()

    # Load tokenizer
    print("Loading BPE tokenizer...")
    tokenizer_path = "models/tokenizer_8k.model"
    tokenizer = BPETokenizer(tokenizer_path)
    print(f"Loaded tokenizer: {tokenizer.vocab_size} vocab size")
    print()

    # Tokenize training data
    print("=" * 70)
    print("TOKENIZING TRAINING DATA")
    print("=" * 70)
    train_dataset = FastTokenDataset(
        text_path="data/tinystories/train.txt",
        tokenizer=tokenizer,
        seq_len=512,
        cache_dir="cache/tokenized",
    )
    print(f"Train dataset ready: {len(train_dataset):,} samples")
    print()

    # Tokenize validation data
    print("=" * 70)
    print("TOKENIZING VALIDATION DATA")
    print("=" * 70)
    val_dataset = FastTokenDataset(
        text_path="data/tinystories/val.txt",
        tokenizer=tokenizer,
        seq_len=512,
        cache_dir="cache/tokenized",
    )
    print(f"Val dataset ready: {len(val_dataset):,} samples")
    print()

    print("=" * 70)
    print("PRE-TOKENIZATION COMPLETE!")
    print("=" * 70)
    print()
    print("Cached files created:")
    cache_dir = Path("cache/tokenized")
    for cache_file in cache_dir.glob("*.pt"):
        size_mb = cache_file.stat().st_size / (1024 * 1024)
        print(f"  {cache_file.name} ({size_mb:.1f} MB)")
    print()
    print("Now when you run training, it will load instantly from cache!")
    print()


if __name__ == "__main__":
    pretokenize()
