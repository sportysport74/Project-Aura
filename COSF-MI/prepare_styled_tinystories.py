"""
Prepare Styled TinyStories Dataset

Adds style token conditioning to TinyStories while keeping ALL COSF features.

What this does:
  - Reads data/tinystories/train.txt and val.txt
  - Prepends "[STORY] " to each story
  - Saves to data/tinystories/styled_train.txt and styled_val.txt

What this does NOT change:
  - COSF oscillator architecture (unchanged)
  - Chirality (unchanged)
  - Homeostatic sleep (unchanged)
  - Julia bounds (unchanged)
  - Phase coherence (unchanged)

This is DATA PREPROCESSING only, not model architecture changes.

Usage:
  python prepare_styled_tinystories.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "python"))


def prepare_styled_data(
    input_file: Path,
    output_file: Path,
    style_token: str = "[STORY]",
    verbose: bool = True
):
    """
    Prepend style token to each story in the dataset.

    Args:
        input_file: Input file (e.g., train.txt)
        output_file: Output file (e.g., styled_train.txt)
        style_token: Token to prepend (default: "[STORY]")
        verbose: Print progress
    """
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    if verbose:
        print(f"Reading from: {input_file}")
        print(f"Writing to: {output_file}")
        print(f"Style token: {style_token}")
        print()

    # Read all stories
    with open(input_file, 'r', encoding='utf-8') as f:
        stories = f.read().split('<|endoftext|>')

    # Remove empty stories
    stories = [s.strip() for s in stories if s.strip()]

    if verbose:
        print(f"Found {len(stories)} stories")
        print()

    # Prepend style token to each story
    styled_stories = []
    for i, story in enumerate(stories):
        # Add style token at the beginning
        styled_story = f"{style_token} {story}"
        styled_stories.append(styled_story)

        # Show progress
        if verbose and (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1}/{len(stories)} stories...")

    if verbose:
        print(f"  Processed {len(stories)}/{len(stories)} stories")
        print()

    # Write styled stories
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('<|endoftext|>'.join(styled_stories))

    if verbose:
        print(f"[OK] Styled data saved to: {output_file}")
        print()

        # Show example
        print("Example styled story (first 200 chars):")
        print("-" * 70)
        print(styled_stories[0][:200] + "...")
        print("-" * 70)
        print()


def main():
    """Process both train and val splits."""
    print("=" * 70)
    print("PREPARE STYLED TINYSTORIES")
    print("=" * 70)
    print()
    print("Adding [STORY] token to TinyStories dataset")
    print("This is DATA PREPROCESSING only - model architecture unchanged")
    print()
    print("COSF features preserved:")
    print("  [OK] Oscillators")
    print("  [OK] Chirality")
    print("  [OK] Homeostatic sleep")
    print("  [OK] Julia bounds")
    print("  [OK] Phase coherence")
    print()
    print("=" * 70)
    print()

    # Define paths
    data_dir = Path("data/tinystories")

    # Process training data
    print("PROCESSING TRAINING DATA")
    print("=" * 70)
    train_input = data_dir / "train.txt"
    train_output = data_dir / "styled_train.txt"
    prepare_styled_data(train_input, train_output, style_token="[STORY]")

    # Process validation data
    print()
    print("PROCESSING VALIDATION DATA")
    print("=" * 70)
    val_input = data_dir / "val.txt"
    val_output = data_dir / "styled_val.txt"
    prepare_styled_data(val_input, val_output, style_token="[STORY]")

    print()
    print("=" * 70)
    print("[OK] STYLED DATASET READY")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Update training script to use styled_train.txt and styled_val.txt")
    print("  2. Retrain with same COSF architecture (no model changes)")
    print("  3. Model will learn to condition on [STORY] token")
    print()
    print("Files created:")
    print(f"  - {train_output}")
    print(f"  - {val_output}")
    print()


if __name__ == "__main__":
    main()
