"""
Fast Token Dataset with Caching and Progress Bars

Improves tokenization speed by:
1. Chunked tokenization with progress bars
2. Disk caching to avoid re-tokenization
3. Memory-mapped loading for large datasets
"""

from pathlib import Path
from typing import Optional, Tuple
import pickle
import hashlib
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class FastTokenDataset(Dataset):
    """Token dataset with caching and chunked tokenization."""

    def __init__(
        self,
        text_path: str,
        tokenizer,
        seq_len: int,
        stride: Optional[int] = None,
        cache_dir: str = "cache/tokenized",
    ):
        """Initialize fast token dataset.

        Args:
            text_path: Path to text file
            tokenizer: Tokenizer with encode() method
            seq_len: Sequence length for each sample
            stride: Stride for sliding window (default: seq_len)
            cache_dir: Directory for cached tokenized data
        """
        self.text_path = Path(text_path)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride or seq_len
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Generate cache key based on file and tokenizer
        cache_key = self._get_cache_key()
        self.cache_path = self.cache_dir / f"{cache_key}.pt"

        # Load or create tokenized data
        if self.cache_path.exists():
            print(f"Loading cached tokens from {self.cache_path}")
            self.tokens = torch.load(self.cache_path, weights_only=True)
            print(f"Loaded {len(self.tokens):,} cached tokens")
        else:
            print(f"Cache not found, tokenizing {self.text_path}...")
            self.tokens = self._tokenize_with_progress()
            print(f"Saving tokens to cache: {self.cache_path}")
            torch.save(self.tokens, self.cache_path)
            print("Cache saved!")

        # Calculate number of samples
        self.n_samples = max(0, (len(self.tokens) - seq_len - 1) // self.stride)
        print(f"Dataset ready: {self.n_samples:,} samples")

    def _get_cache_key(self) -> str:
        """Generate unique cache key for this dataset."""
        # Hash file path, file mtime, and tokenizer model path
        hash_input = f"{self.text_path}_{self.text_path.stat().st_mtime}"

        # Add tokenizer info if available
        if hasattr(self.tokenizer, 'model_path'):
            hash_input += f"_{self.tokenizer.model_path}"
        elif hasattr(self.tokenizer, 'vocab_size'):
            hash_input += f"_{self.tokenizer.vocab_size}"

        cache_key = hashlib.md5(hash_input.encode()).hexdigest()[:16]
        return f"{self.text_path.stem}_{cache_key}"

    def _tokenize_with_progress(self, chunk_size: int = 1024 * 1024) -> torch.Tensor:
        """Tokenize text in chunks with progress bar.

        Args:
            chunk_size: Characters per chunk (default: 1MB)

        Returns:
            Tokenized tensor
        """
        # Read file
        print(f"Reading {self.text_path}...")
        with open(self.text_path, 'r', encoding='utf-8') as f:
            text = f.read()

        total_chars = len(text)
        print(f"Read {total_chars:,} characters")

        # Tokenize in chunks, many at a time (sentencepiece is multithreaded),
        # into int32 arrays - a Python list of ~300M ints would need ~10 GB of RAM
        import numpy as np
        num_chunks = (total_chars + chunk_size - 1) // chunk_size
        print(f"Tokenizing in {num_chunks} chunks...")

        parts = []
        sp = getattr(self.tokenizer, 'sp', None)
        batch = 64
        with tqdm(total=total_chars, desc="Tokenizing", unit="char", unit_scale=True) as pbar:
            for i in range(0, total_chars, chunk_size * batch):
                chunks = [text[j:j + chunk_size]
                          for j in range(i, min(i + chunk_size * batch, total_chars), chunk_size)]
                if sp is not None:
                    encoded = sp.encode(chunks, num_threads=16)
                else:
                    encoded = [self.tokenizer.encode(c) for c in chunks]
                for ids in encoded:
                    parts.append(np.asarray(ids, dtype=np.int32))
                pbar.update(sum(len(c) for c in chunks))

        tokens = torch.from_numpy(np.concatenate(parts))
        print(f"Tokenized to {len(tokens):,} tokens")
        return tokens

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a training sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (input_tokens, target_tokens)
        """
        start_idx = idx * self.stride
        end_idx = start_idx + self.seq_len

        # Input and target (shifted by 1)
        input_tokens = self.tokens[start_idx:end_idx].long()
        target_tokens = self.tokens[start_idx + 1:end_idx + 1].long()

        # Pad if necessary
        if len(input_tokens) < self.seq_len:
            pad_len = self.seq_len - len(input_tokens)
            input_tokens = torch.cat([input_tokens, torch.zeros(pad_len, dtype=torch.long)])
            target_tokens = torch.cat([target_tokens, torch.zeros(pad_len, dtype=torch.long)])

        return input_tokens, target_tokens


def create_datasets_fast(
    train_path: str,
    val_path: str,
    tokenizer,
    seq_len: int,
    stride: Optional[int] = None,
    cache_dir: str = "cache/tokenized",
) -> Tuple[FastTokenDataset, FastTokenDataset]:
    """Create train and validation datasets with caching.

    Args:
        train_path: Path to training text file
        val_path: Path to validation text file
        tokenizer: Tokenizer instance
        seq_len: Sequence length
        stride: Optional stride for sliding window
        cache_dir: Cache directory

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    print("=" * 70)
    print("CREATING DATASETS WITH FAST CACHING")
    print("=" * 70)
    print()

    print("Creating training dataset...")
    train_dataset = FastTokenDataset(
        train_path, tokenizer, seq_len, stride, cache_dir
    )

    print("\nCreating validation dataset...")
    val_dataset = FastTokenDataset(
        val_path, tokenizer, seq_len, stride, cache_dir
    )

    print()
    print("=" * 70)
    print("DATASETS READY")
    print("=" * 70)
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")
    print()

    return train_dataset, val_dataset
