"""
Download and prepare TinyStories dataset for COSF Transformer training.

TinyStories is a dataset of simple, short stories designed for training
small language models. It's perfect for our 50-100M parameter models.

Dataset: https://huggingface.co/datasets/roneneldan/TinyStories

Usage:
    python -m cosf_mi.data.prepare_tinystories --output data/tinystories
"""

import argparse
import json
from pathlib import Path
from typing import Optional
import urllib.request
import gzip
import shutil

def download_file(url: str, output_path: Path, show_progress: bool = True):
    """Download a file with progress bar."""
    print(f"Downloading {url}...")

    def report_hook(block_num, block_size, total_size):
        if show_progress and total_size > 0:
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size)
            print(f"\rProgress: {percent:.1f}% ({downloaded / 1024 / 1024:.1f}MB / {total_size / 1024 / 1024:.1f}MB)", end="")

    urllib.request.urlretrieve(url, output_path, reporthook=report_hook if show_progress else None)
    if show_progress:
        print()  # New line after progress


def prepare_tinystories(
    output_dir: str = "data/tinystories",
    train_size_mb: Optional[int] = None,
    val_split: float = 0.05,
):
    """Download and prepare TinyStories dataset.

    Args:
        output_dir: Output directory for prepared data
        train_size_mb: Optional size limit for training data in MB
        val_split: Validation split fraction (default: 5%)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TINYSTORIES DATASET PREPARATION")
    print("=" * 70)
    print()

    # TinyStories is available as a single file from HuggingFace
    # For simplicity, we'll download a preprocessed version or use the HF datasets library

    try:
        from datasets import load_dataset
        print("Using HuggingFace datasets library...")

        # Load TinyStories dataset
        print("\nLoading TinyStories dataset...")
        dataset = load_dataset("roneneldan/TinyStories", split="train")

        print(f"Loaded {len(dataset)} stories")

        # Extract text
        all_text = []
        total_chars = 0

        print("\nExtracting stories...")
        for i, example in enumerate(dataset):
            story = example['text']
            all_text.append(story)
            total_chars += len(story)

            # Limit size if specified
            if train_size_mb and total_chars > train_size_mb * 1024 * 1024:
                print(f"Reached size limit of {train_size_mb}MB")
                break

            if (i + 1) % 10000 == 0:
                print(f"  Processed {i + 1} stories ({total_chars / 1024 / 1024:.1f}MB)")

        # Join all stories
        combined_text = "\n\n".join(all_text)

        print(f"\nTotal: {len(all_text)} stories, {len(combined_text):,} characters ({len(combined_text) / 1024 / 1024:.1f}MB)")

        # Split into train/val
        split_idx = int(len(combined_text) * (1 - val_split))
        train_text = combined_text[:split_idx]
        val_text = combined_text[split_idx:]

        # Save
        train_path = output_dir / "train.txt"
        val_path = output_dir / "val.txt"

        print(f"\nSaving training data to {train_path}...")
        with open(train_path, 'w', encoding='utf-8') as f:
            f.write(train_text)

        print(f"Saving validation data to {val_path}...")
        with open(val_path, 'w', encoding='utf-8') as f:
            f.write(val_text)

        # Save metadata
        metadata = {
            'dataset': 'TinyStories',
            'num_stories': len(all_text),
            'total_chars': len(combined_text),
            'total_mb': len(combined_text) / 1024 / 1024,
            'train_chars': len(train_text),
            'val_chars': len(val_text),
            'val_split': val_split,
        }

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print("\n" + "=" * 70)
        print("DATASET PREPARATION COMPLETE")
        print("=" * 70)
        print(f"Training data:   {len(train_text):,} chars ({len(train_text) / 1024 / 1024:.1f}MB)")
        print(f"Validation data: {len(val_text):,} chars ({len(val_text) / 1024 / 1024:.1f}MB)")
        print(f"Output directory: {output_dir.absolute()}")
        print()

        return train_path, val_path, metadata

    except ImportError:
        print("\nERROR: HuggingFace datasets library not installed.")
        print("Install with: pip install datasets")
        print("\nAlternatively, download TinyStories manually from:")
        print("https://huggingface.co/datasets/roneneldan/TinyStories")
        return None, None, None


def main():
    parser = argparse.ArgumentParser(description="Prepare TinyStories dataset")
    parser.add_argument(
        "--output",
        type=str,
        default="data/tinystories",
        help="Output directory for prepared data",
    )
    parser.add_argument(
        "--size-mb",
        type=int,
        default=None,
        help="Optional size limit in MB (default: download all)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.05,
        help="Validation split fraction (default: 0.05 = 5%%)",
    )

    args = parser.parse_args()

    prepare_tinystories(
        output_dir=args.output,
        train_size_mb=args.size_mb,
        val_split=args.val_split,
    )


if __name__ == "__main__":
    main()
