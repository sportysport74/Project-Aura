"""
Tokenizer for COSF Transformer

Supports both character-level and BPE tokenization.
Uses SentencePiece for efficient BPE implementation.

Usage:
    # Train a new tokenizer
    python -m cosf_mi.data.tokenizer --train data/tinystories/train.txt --vocab-size 8000

    # Use existing tokenizer
    from cosf_mi.data.tokenizer import Tokenizer
    tokenizer = Tokenizer.from_pretrained("models/tokenizer_8k.model")
"""

import argparse
from pathlib import Path
from typing import List, Optional, Union
import json


class CharTokenizer:
    """Simple character-level tokenizer."""

    def __init__(self, vocab_size: int = 256):
        """Initialize character tokenizer.

        Args:
            vocab_size: Maximum character code (default: 256 for extended ASCII)
        """
        self.vocab_size = vocab_size

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return [min(ord(c), self.vocab_size - 1) for c in text]

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        return ''.join(chr(max(0, min(id, self.vocab_size - 1))) for id in ids)

    def save(self, path: str):
        """Save tokenizer config."""
        config = {
            'type': 'char',
            'vocab_size': self.vocab_size,
        }
        with open(path, 'w') as f:
            json.dump(config, f)

    @classmethod
    def from_pretrained(cls, path: str):
        """Load tokenizer from config."""
        with open(path) as f:
            config = json.load(f)
        return cls(vocab_size=config['vocab_size'])


class BPETokenizer:
    """BPE tokenizer using SentencePiece."""

    def __init__(self, model_path: str):
        """Initialize BPE tokenizer.

        Args:
            model_path: Path to trained SentencePiece model
        """
        try:
            import sentencepiece as spm
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(model_path)
            self.vocab_size = self.sp.vocab_size()
            self.model_path = model_path
        except ImportError:
            raise ImportError(
                "SentencePiece not installed. Install with: pip install sentencepiece"
            )

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self.sp.decode(ids)

    def save(self, path: str):
        """Save tokenizer reference (model is already saved)."""
        config = {
            'type': 'bpe',
            'model_path': self.model_path,
            'vocab_size': self.vocab_size,
        }
        with open(path, 'w') as f:
            json.dump(config, f)

    @classmethod
    def from_pretrained(cls, path: str):
        """Load tokenizer from saved config."""
        with open(path) as f:
            config = json.dump(config, f)
        return cls(model_path=config['model_path'])

    @classmethod
    def train(
        cls,
        input_file: str,
        model_prefix: str,
        vocab_size: int = 8000,
        character_coverage: float = 1.0,
        model_type: str = 'bpe',
    ):
        """Train a new SentencePiece model.

        Args:
            input_file: Path to training text file
            model_prefix: Output model prefix (will create .model and .vocab files)
            vocab_size: Vocabulary size (default: 8000)
            character_coverage: Character coverage (default: 1.0 for full coverage)
            model_type: Model type ('bpe' or 'unigram')

        Returns:
            Trained BPETokenizer instance
        """
        try:
            import sentencepiece as spm

            print(f"Training SentencePiece {model_type} tokenizer...")
            print(f"  Input: {input_file}")
            print(f"  Vocab size: {vocab_size}")
            print(f"  Model prefix: {model_prefix}")

            spm.SentencePieceTrainer.train(
                input=input_file,
                model_prefix=model_prefix,
                vocab_size=vocab_size,
                character_coverage=character_coverage,
                model_type=model_type,
                pad_id=0,
                unk_id=1,
                bos_id=2,
                eos_id=3,
            )

            print(f"Tokenizer trained successfully!")
            print(f"  Model: {model_prefix}.model")
            print(f"  Vocab: {model_prefix}.vocab")

            return cls(model_path=f"{model_prefix}.model")

        except ImportError:
            raise ImportError(
                "SentencePiece not installed. Install with: pip install sentencepiece"
            )


class Tokenizer:
    """Unified tokenizer interface supporting both char and BPE."""

    def __init__(self, tokenizer: Union[CharTokenizer, BPETokenizer]):
        """Initialize unified tokenizer.

        Args:
            tokenizer: Underlying tokenizer (CharTokenizer or BPETokenizer)
        """
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self.tokenizer.encode(text)

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(ids)

    def save(self, path: str):
        """Save tokenizer config."""
        self.tokenizer.save(path)

    @classmethod
    def from_pretrained(cls, path: str):
        """Load tokenizer from config."""
        with open(path) as f:
            config = json.load(f)

        if config['type'] == 'char':
            tokenizer = CharTokenizer.from_pretrained(path)
        elif config['type'] == 'bpe':
            tokenizer = BPETokenizer.from_pretrained(path)
        else:
            raise ValueError(f"Unknown tokenizer type: {config['type']}")

        return cls(tokenizer)

    @classmethod
    def train_bpe(
        cls,
        input_file: str,
        model_prefix: str,
        vocab_size: int = 8000,
    ):
        """Train a new BPE tokenizer.

        Args:
            input_file: Path to training text file
            model_prefix: Output model prefix
            vocab_size: Vocabulary size

        Returns:
            Trained Tokenizer instance
        """
        bpe_tokenizer = BPETokenizer.train(
            input_file=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
        )
        return cls(bpe_tokenizer)

    @classmethod
    def char(cls, vocab_size: int = 256):
        """Create a character-level tokenizer.

        Args:
            vocab_size: Vocabulary size (default: 256)

        Returns:
            Character-level Tokenizer instance
        """
        return cls(CharTokenizer(vocab_size=vocab_size))


def main():
    parser = argparse.ArgumentParser(description="Train COSF tokenizer")
    parser.add_argument(
        "--train",
        type=str,
        required=True,
        help="Input text file for training",
    )
    parser.add_argument(
        "--model-prefix",
        type=str,
        default="models/tokenizer",
        help="Output model prefix (default: models/tokenizer)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=8000,
        help="Vocabulary size (default: 8000)",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="bpe",
        choices=['bpe', 'unigram'],
        help="Tokenizer type (default: bpe)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.model_prefix).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train tokenizer
    print("=" * 70)
    print("COSF TOKENIZER TRAINING")
    print("=" * 70)
    print()

    tokenizer = Tokenizer.train_bpe(
        input_file=args.train,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
    )

    # Save config
    config_path = f"{args.model_prefix}.json"
    tokenizer.save(config_path)
    print(f"\nTokenizer config saved to: {config_path}")

    # Test tokenizer
    test_text = "Once upon a time, there was a little girl who loved to read stories."
    print(f"\nTest encoding:")
    print(f"  Text: {test_text}")
    encoded = tokenizer.encode(test_text)
    print(f"  Tokens: {encoded[:20]}... ({len(encoded)} total)")
    decoded = tokenizer.decode(encoded)
    print(f"  Decoded: {decoded}")

    print("\n" + "=" * 70)
    print("TOKENIZER TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
