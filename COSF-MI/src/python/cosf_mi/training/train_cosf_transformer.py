"""
Train COSF Transformer on Character-Level Shakespeare

A complete training script for the COSF-native transformer architecture.
Trains on tiny-shakespeare dataset with character-level tokenization.

Usage:
    python -m cosf_mi.training.train_cosf_transformer

Features:
- Character-level language modeling
- Checkpoint saving/loading
- Text generation sampling
- COSF-specific monitoring (φ-shell distributions, Julia bounds)
- Perplexity tracking
- Sample generation during training
"""

import os
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from ..models.cosf_transformer import COSFTransformer, COSFConfig
from ..core.constants import PHI, JULIA_STABILITY_BOUND


# ============================================================================
# Training Configuration
# ============================================================================

@dataclass
class TrainConfig:
    """Training hyperparameters."""
    # Data
    data_path: str = "data/shakespeare.txt"
    train_split: float = 0.9

    # Model (default to small for local training)
    vocab_size: int = 256
    d_model: int = 384
    n_layers: int = 7
    n_heads: int = 8
    max_seq_len: int = 512

    # Training
    batch_size: int = 64
    seq_len: int = 256  # Half of max for memory efficiency
    num_epochs: int = 10
    max_steps: Optional[int] = None  # Override epochs if set
    learning_rate: float = 1e-4  # Reduced from 3e-4 for stability with Julia activations
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 500

    # Checkpointing
    checkpoint_dir: str = "checkpoints/cosf_transformer"
    save_every_epochs: int = 1
    save_every_steps: Optional[int] = None

    # Monitoring
    log_every_steps: int = 100
    generate_every_steps: int = 500
    generate_length: int = 200

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # COSF monitoring
    track_phi_shells: bool = True
    track_julia_bounds: bool = True

    # Resume training
    resume_from: Optional[str] = None  # Path to checkpoint to resume from
    start_epoch: int = 1  # Starting epoch (auto-set when resuming)


# ============================================================================
# Data Loading
# ============================================================================

class CharDataset(Dataset):
    """Character-level text dataset."""

    def __init__(self, text: str, seq_len: int):
        """Initialize dataset.

        Args:
            text: Raw text string
            seq_len: Sequence length for each sample
        """
        self.text = text
        self.seq_len = seq_len
        self.data = torch.tensor([ord(c) for c in text], dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        """Get a single sequence.

        Returns:
            Tuple of (input_ids, target_ids)
            input_ids: [seq_len]
            target_ids: [seq_len] (shifted by 1)
        """
        chunk = self.data[idx:idx + self.seq_len + 1]
        x = chunk[:-1]  # Input
        y = chunk[1:]   # Target (next char)
        return x, y


def load_shakespeare(config: TrainConfig) -> Tuple[CharDataset, CharDataset]:
    """Load and split Shakespeare dataset.

    Args:
        config: Training configuration

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Try to load from file
    data_path = Path(config.data_path)

    if not data_path.exists():
        print(f"Data file not found at {data_path}")
        print("Downloading tiny-shakespeare...")

        # Download from GitHub
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

        data_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, data_path)
        print(f"Downloaded to {data_path}")

    # Load text
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"Loaded {len(text):,} characters")

    # Check character range
    chars = set(text)
    max_char = max(ord(c) for c in chars)
    if max_char >= config.vocab_size:
        print(f"Warning: max character code {max_char} >= vocab size {config.vocab_size}")
        print("Filtering to ASCII printable characters...")
        text = ''.join(c for c in text if ord(c) < config.vocab_size)
        print(f"Filtered to {len(text):,} characters")

    # Train/val split
    split_idx = int(len(text) * config.train_split)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    # Create datasets
    train_dataset = CharDataset(train_text, config.seq_len)
    val_dataset = CharDataset(val_text, config.seq_len)

    print(f"Train: {len(train_dataset):,} sequences")
    print(f"Val:   {len(val_dataset):,} sequences")

    return train_dataset, val_dataset


# ============================================================================
# Training Utilities
# ============================================================================

def get_lr_schedule(optimizer, warmup_steps: int):
    """Create learning rate schedule with warmup.

    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps

    Returns:
        LR scheduler
    """
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        # Cosine decay after warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (10000 - warmup_steps)))

    return LambdaLR(optimizer, lr_lambda)


def compute_phi_shell_distribution(activations: torch.Tensor, num_shells: int = 10) -> dict:
    """Compute distribution of activations across φ-shells.

    Args:
        activations: Tensor of activations [batch, seq, features]
        num_shells: Number of shells to track

    Returns:
        Dictionary with shell statistics
    """
    # Compute norms
    norms = torch.norm(activations, p=2, dim=-1).flatten()

    # Log space for φ-shells
    log_norms = torch.log(norms + 1e-8)
    log_phi = math.log(PHI)

    # Compute shell indices
    shell_indices = (log_norms / log_phi).long()
    shell_indices = torch.clamp(shell_indices, 0, num_shells - 1)

    # Count per shell
    counts = torch.bincount(shell_indices, minlength=num_shells)
    distribution = counts.float() / counts.sum()

    return {
        f'shell_{i}': distribution[i].item()
        for i in range(num_shells)
    }


def check_julia_bounds(activations: torch.Tensor, bound: float = JULIA_STABILITY_BOUND) -> dict:
    """Check what fraction of activations exceed Julia bounds.

    Args:
        activations: Tensor of activations [batch, seq, features]
        bound: Julia stability bound

    Returns:
        Dictionary with bound statistics
    """
    norms = torch.norm(activations, p=2, dim=-1).flatten()

    exceeding = (norms > bound).float()
    frac_exceeding = exceeding.mean().item()
    max_norm = norms.max().item()
    mean_norm = norms.mean().item()

    return {
        'frac_exceeding_bound': frac_exceeding,
        'max_norm': max_norm,
        'mean_norm': mean_norm,
    }


@torch.no_grad()
def generate_text(
    model: COSFTransformer,
    prompt: str,
    max_length: int = 200,
    temperature: float = 0.8,
    device: str = 'cuda',
) -> str:
    """Generate text continuation from prompt.

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

    # Encode prompt
    input_ids = torch.tensor([[ord(c) for c in prompt]], dtype=torch.long).to(device)

    generated = prompt
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
        char = chr(next_token.item())
        generated += char

        # Stop if we hit a newline after some generation
        if len(generated) > len(prompt) + 20 and char == '\n':
            break

    return generated


# ============================================================================
# Training Loop
# ============================================================================

def train(config: TrainConfig):
    """Main training function.

    Args:
        config: Training configuration
    """
    print("=" * 70)
    print("COSF Transformer Training")
    print("=" * 70)

    # Setup
    device = torch.device(config.device)
    print(f"\nDevice: {device}")

    # Create checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    train_dataset, val_dataset = load_shakespeare(config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Create model
    print("\nCreating model...")
    model_config = COSFConfig(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        max_seq_len=config.max_seq_len,
    )

    model = COSFTransformer(model_config).to(device)

    param_counts = model.count_parameters()
    print(f"Model parameters: {param_counts['total']:,}")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = get_lr_schedule(optimizer, config.warmup_steps)

    # Training state
    step = 0
    best_val_loss = float('inf')
    start_epoch = config.start_epoch

    # Resume from checkpoint if specified
    if config.resume_from:
        print(f"\nResuming from checkpoint: {config.resume_from}")
        checkpoint = torch.load(config.resume_from, map_location=device, weights_only=False)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'step' in checkpoint:
            step = checkpoint['step']

        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1  # Start from next epoch
            print(f"Resuming from epoch {checkpoint['epoch']}, will start at epoch {start_epoch}")

        if 'val_loss' in checkpoint:
            best_val_loss = checkpoint.get('best_val_loss', checkpoint['val_loss'])
            print(f"Previous best validation loss: {best_val_loss:.4f}")

        print(f"Resumed at step {step}")

    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)

    for epoch in range(start_epoch, start_epoch + config.num_epochs):
        model.train()
        epoch_loss = 0
        epoch_start = time.time()

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            # Forward pass
            logits = model(x)

            # Compute loss
            loss = F.cross_entropy(
                logits.reshape(-1, config.vocab_size),
                y.reshape(-1),
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            step += 1

            # Logging
            if step % config.log_every_steps == 0:
                ppl = math.exp(loss.item())
                lr = optimizer.param_groups[0]['lr']
                print(f"Step {step:5d} | Loss: {loss.item():.4f} | PPL: {ppl:7.2f} | LR: {lr:.6f}")

                # COSF monitoring
                if config.track_phi_shells or config.track_julia_bounds:
                    with torch.no_grad():
                        # Get activations from middle layer
                        mid_layer = model.blocks[len(model.blocks) // 2]
                        sample_x = x[:4]  # Small sample
                        activations, _ = mid_layer(model.token_embedding(sample_x))

                        if config.track_julia_bounds:
                            julia_stats = check_julia_bounds(activations)
                            if julia_stats['frac_exceeding_bound'] > 0:
                                print(f"  Julia: {julia_stats['frac_exceeding_bound']*100:.1f}% exceed bound (max: {julia_stats['max_norm']:.2f})")

            # Generation
            if step % config.generate_every_steps == 0:
                print("\n" + "-" * 70)
                prompt = "ROMEO:\n"
                generated = generate_text(
                    model,
                    prompt,
                    max_length=config.generate_length,
                    temperature=0.8,
                    device=device,
                )
                print("Generated sample:")
                # Use errors='replace' to handle non-ASCII characters on Windows
                try:
                    print(generated)
                except UnicodeEncodeError:
                    print(generated.encode('ascii', errors='replace').decode('ascii'))
                print("-" * 70 + "\n")
                model.train()

            # Step-based checkpoint
            if config.save_every_steps and step % config.save_every_steps == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'config': model_config,
                }, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")

            # Early stopping by steps
            if config.max_steps and step >= config.max_steps:
                break

        # Epoch statistics
        avg_train_loss = epoch_loss / len(train_loader)
        epoch_time = time.time() - epoch_start

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, config.vocab_size),
                    y.reshape(-1),
                )
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_ppl = math.exp(avg_val_loss)

        print(f"\nEpoch {epoch+1}/{config.num_epochs} ({epoch_time:.1f}s):")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  Val PPL:    {val_ppl:.2f}")

        # Save epoch checkpoint
        if (epoch + 1) % config.save_every_epochs == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': model_config,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = checkpoint_dir / "best_model.pt"
            torch.save({
                'epoch': epoch + 1,
                'step': step,
                'model_state_dict': model.state_dict(),
                'val_loss': avg_val_loss,
                'config': model_config,
            }, best_path)
            print(f"New best model! Val loss: {avg_val_loss:.4f}")

        print()

        if config.max_steps and step >= config.max_steps:
            break

    print("=" * 70)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation PPL: {math.exp(best_val_loss):.2f}")
    print("=" * 70)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train COSF Transformer")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (e.g., checkpoints/cosf_transformer/checkpoint_epoch_10.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: 1e-4, or 5e-5 if resuming)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)",
    )

    args = parser.parse_args()

    # Default configuration
    config = TrainConfig()
    config.num_epochs = args.epochs

    # Set learning rate
    if args.lr is not None:
        config.learning_rate = args.lr
    elif args.resume:
        # Use lower learning rate for fine-tuning when resuming
        config.learning_rate = 5e-5

    # Set device
    if args.device:
        config.device = args.device

    # Set resume path
    if args.resume:
        config.resume_from = args.resume

    print("\nConfiguration:")
    print(f"  Data: {config.data_path}")
    print(f"  Model: d_model={config.d_model}, layers={config.n_layers}")
    print(f"  Training: batch_size={config.batch_size}, seq_len={config.seq_len}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  LR: {config.learning_rate}, warmup={config.warmup_steps}")
    print(f"  Device: {config.device}")
    if config.resume_from:
        print(f"  Resuming from: {config.resume_from}")
    print()

    # Run training
    train(config)
