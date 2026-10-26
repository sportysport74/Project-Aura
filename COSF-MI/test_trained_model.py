"""
Test Trained COSF Model

Load the best trained model and generate text to see what it learned.

Usage:
    python test_trained_model.py
"""

import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "python"))

from cosf_mi.models.cosf_oscillator import COSFOscillatorTransformer
from cosf_mi.data.tokenizer import BPETokenizer


def load_best_model(checkpoint_path: str = "checkpoints/styled_cosf/best_model.pt"):
    """Load the best trained model."""
    print("=" * 70)
    print("LOADING BEST MODEL")
    print("=" * 70)
    print()

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    print(f"Checkpoint info:")
    print(f"  Epoch: {checkpoint['epoch'] + 1}")
    print(f"  Val loss: {checkpoint['val_loss']:.4f}")
    print(f"  Perplexity: {torch.exp(torch.tensor(checkpoint['val_loss'])):.2f}")
    print()

    # Create model with same config
    config = checkpoint['config']
    model = COSFOscillatorTransformer(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded successfully!")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Chirality: {'ENABLED' if config.enable_chirality else 'DISABLED'}")
    print()

    return model, config


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.8,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """Generate text from a prompt."""
    model = model.to(device)
    model.eval()

    # Encode prompt
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)

    print(f"Generating from prompt: '{prompt}'")
    print(f"Initial tokens: {len(tokens)}")
    print()
    print("-" * 70)
    print(prompt, end='', flush=True)

    # Generate
    with torch.no_grad():
        for _ in range(max_tokens):
            # Forward pass (disable coherence loss for inference)
            logits = model(input_ids, return_coherence_loss=False)

            # Get next token logits
            next_token_logits = logits[0, -1, :] / temperature

            # Sample from distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Decode and print
            next_text = tokenizer.decode([next_token.item()])
            print(next_text, end='', flush=True)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

            # Stop at end of text token (if tokenizer has one)
            if hasattr(tokenizer, 'eos_id') and next_token.item() == tokenizer.eos_id:
                break

    print()
    print("-" * 70)
    print()


def get_model_stats(model):
    """Get statistics about the trained model."""
    print("=" * 70)
    print("MODEL STATISTICS")
    print("=" * 70)
    print()

    # Get phase coherence metrics
    metrics = model.get_phase_coherence_metrics()

    print("Phase Metrics:")
    print(f"  Mean coherence: {metrics['mean_coherence']:.4f}")
    print(f"  Max phase velocity: {metrics['max_phase_velocity']:.4f}")
    print(f"  Homeostatic drift: {metrics['homeostatic_drift']:.4f}")
    print(f"  Julia violations: {metrics['julia_violations']}")
    print()

    if 'chirality_balance' in metrics:
        print("Chirality Metrics:")
        print(f"  Right-handed oscillators: {metrics['total_right_handed']}")
        print(f"  Left-handed oscillators: {metrics['total_left_handed']}")
        print(f"  Balance (right ratio): {metrics['chirality_balance']:.3f}")
        print(f"  Total chirality flips: {metrics['total_chirality_flips']}")
        print()


def main():
    print()
    print("=" * 70)
    print("TEST TRAINED COSF MODEL")
    print("=" * 70)
    print()

    # Load model
    model, config = load_best_model()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = BPETokenizer("models/tokenizer_8k.model")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print()

    # Get model statistics
    get_model_stats(model)

    # Test prompts
    test_prompts = [
        "[STORY] Once upon a time",
        "[STORY] A little girl named",
        "[STORY] The brave knight",
        "[STORY] In a magical forest",
    ]

    print("=" * 70)
    print("GENERATING TEXT")
    print("=" * 70)
    print()

    for i, prompt in enumerate(test_prompts, 1):
        print(f"Test {i}/{len(test_prompts)}:")
        print()
        generate_text(model, tokenizer, prompt, max_tokens=50, temperature=0.8)
        print()

    print("=" * 70)
    print("TESTING COMPLETE")
    print("=" * 70)
    print()
    print("The model has been tested!")
    print()
    print("Next steps:")
    print("  1. Try your own prompts by modifying test_prompts in the script")
    print("  2. Adjust temperature (0.5 = more focused, 1.0 = more random)")
    print("  3. Continue training for more epochs")
    print()


if __name__ == "__main__":
    main()
