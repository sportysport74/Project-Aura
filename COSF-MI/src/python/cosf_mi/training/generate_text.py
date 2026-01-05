"""
Generate Text from Trained COSF Transformer

Loads a trained checkpoint and generates text from various prompts.

Usage:
    python -m cosf_mi.training.generate_text
    python -m cosf_mi.training.generate_text --checkpoint path/to/checkpoint.pt
    python -m cosf_mi.training.generate_text --length 1000 --temperature 0.9
"""

import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from ..models.cosf_transformer import COSFTransformer


def load_model(checkpoint_path: str, device: str = "cuda") -> COSFTransformer:
    """Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract config and create model
    config = checkpoint['config']
    model = COSFTransformer(config).to(device)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Print checkpoint info
    if 'val_loss' in checkpoint:
        print(f"Validation loss: {checkpoint['val_loss']:.4f}")
    if 'val_ppl' in checkpoint:
        print(f"Validation perplexity: {checkpoint['val_ppl']:.2f}")
    if 'epoch' in checkpoint:
        print(f"Trained for {checkpoint['epoch']} epochs")

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    return model


def generate_text(
    model: COSFTransformer,
    prompt: str,
    max_length: int = 500,
    temperature: float = 0.8,
    device: str = "cuda",
) -> str:
    """Generate text from a prompt.

    Args:
        model: Trained model
        prompt: Starting text
        max_length: Maximum characters to generate
        temperature: Sampling temperature (higher = more random)
        device: Device to run on

    Returns:
        Generated text (including prompt)
    """
    model.eval()
    device_obj = torch.device(device)

    # Encode prompt
    input_ids = torch.tensor([[ord(c) for c in prompt]], dtype=torch.long).to(device_obj)

    generated = prompt
    with torch.no_grad():
        for _ in range(max_length):
            # Get predictions
            logits = model(input_ids)

            # Take last token logits
            next_token_logits = logits[0, -1, :] / temperature

            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

            # Truncate if too long (keep last max_seq_len tokens)
            if input_ids.size(1) > model.config.max_seq_len:
                input_ids = input_ids[:, -model.config.max_seq_len:]

            # Decode and append
            try:
                char = chr(next_token.item())
                # Skip non-printable characters except newlines and tabs
                if char.isprintable() or char in '\n\t':
                    generated += char
            except (ValueError, OverflowError):
                # Skip invalid character codes
                continue

    return generated


def main():
    """Main generation function."""
    parser = argparse.ArgumentParser(description="Generate text from trained COSF Transformer")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/cosf_transformer/best_model.pt",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=500,
        help="Number of characters to generate per prompt",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (0.1-2.0, higher = more random)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=None,
        help="Custom prompts to use (default: predefined prompts)",
    )

    args = parser.parse_args()

    # Load model
    model = load_model(args.checkpoint, args.device)

    # Define prompts
    if args.prompts:
        prompts = args.prompts
    else:
        prompts = [
            "JULIET:\n",
            "KING:\n",
            "To be or not to be",
            "ROMEO:\n",
            "What light through yonder window breaks?",
            "Friends, Romans, countrymen,",
            "HAMLET:\n",
            "MACBETH:\n",
            "The quality of mercy",
            "All the world's a stage",
        ]

    # Generate from each prompt
    print("=" * 70)
    print("GENERATING TEXT FROM PROMPTS")
    print("=" * 70)
    print(f"Temperature: {args.temperature}")
    print(f"Max length: {args.length} characters")
    print()

    for i, prompt in enumerate(prompts, 1):
        print("-" * 70)
        print(f"Prompt {i}/{len(prompts)}: {repr(prompt)}")
        print("-" * 70)

        try:
            generated = generate_text(
                model,
                prompt,
                max_length=args.length,
                temperature=args.temperature,
                device=args.device,
            )

            # Print with error handling for Windows console
            try:
                print(generated)
            except UnicodeEncodeError:
                # Fallback to ASCII with replacement
                print(generated.encode('ascii', errors='replace').decode('ascii'))

            print()

        except Exception as e:
            print(f"Error generating from prompt: {e}")
            print()

    print("=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
