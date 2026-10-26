"""
Token-based Dataset for COSF Transformer

Supports BPE tokenization for efficient training on real text data.
"""

from pathlib import Path
from typing import Optional, Tuple
import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    """Token-based dataset for BPE or other tokenizers.

    Loads pre-tokenized data or tokenizes on-the-fly.
    """

    def __init__(
        self,
        text: str,
        tokenizer,
        seq_len: int,
        stride: Optional[int] = None,
    ):
        """Initialize token dataset.

        Args:
            text: Raw text string to tokenize
            tokenizer: Tokenizer with encode() method
            seq_len: Sequence length for each sample
            stride: Stride for sliding window (default: seq_len for no overlap)
        """
        self.text = text
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride or seq_len

        # Tokenize all text
        print(f"Tokenizing {len(text):,} characters...")
        self.tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        print(f"Tokenized to {len(self.tokens):,} tokens")

        # Calculate number of samples
        # Each sample needs seq_len + 1 tokens (input + target)
        self.n_samples = max(0, (len(self.tokens) - seq_len - 1) // self.stride)

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
        input_tokens = self.tokens[start_idx:end_idx]
        target_tokens = self.tokens[start_idx + 1:end_idx + 1]

        # Pad if necessary (shouldn't happen with proper n_samples calculation)
        if len(input_tokens) < self.seq_len:
            pad_len = self.seq_len - len(input_tokens)
            input_tokens = torch.cat([input_tokens, torch.zeros(pad_len, dtype=torch.long)])
            target_tokens = torch.cat([target_tokens, torch.zeros(pad_len, dtype=torch.long)])

        return input_tokens, target_tokens


def load_text_file(path: str) -> str:
    """Load text file with proper encoding.

    Args:
        path: Path to text file

    Returns:
        Text content as string
    """
    path = Path(path)
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def create_datasets(
    train_path: str,
    val_path: str,
    tokenizer,
    seq_len: int,
    stride: Optional[int] = None,
) -> Tuple[TokenDataset, TokenDataset]:
    """Create train and validation datasets.

    Args:
        train_path: Path to training text file
        val_path: Path to validation text file
        tokenizer: Tokenizer instance
        seq_len: Sequence length
        stride: Optional stride for sliding window

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    print("Loading training data...")
    train_text = load_text_file(train_path)
    print(f"Loaded {len(train_text):,} characters")

    print("\nLoading validation data...")
    val_text = load_text_file(val_path)
    print(f"Loaded {len(val_text):,} characters")

    print("\nCreating training dataset...")
    train_dataset = TokenDataset(train_text, tokenizer, seq_len, stride)

    print("\nCreating validation dataset...")
    val_dataset = TokenDataset(val_text, tokenizer, seq_len, stride)

    return train_dataset, val_dataset
