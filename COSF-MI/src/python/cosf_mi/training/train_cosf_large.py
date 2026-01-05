"""
Train Large COSF Transformer on Real Data

Train 50-200M parameter COSF transformers on TinyStories or other datasets
using BPE tokenization for efficient, useful language models.

Usage:
    # Train small model (78M params) on TinyStories
    python -m cosf_mi.training.train_cosf_large \
        --model-size small \
        --data-dir data/tinystories \
        --tokenizer models/tokenizer_8k.model \
        --epochs 20 \
        --device cuda

    # Resume training
    python -m cosf_mi.training.train_cosf_large \
        --model-size small \
        --resume checkpoints/cosf_small/checkpoint_epoch_10.pt \
        --epochs 10

Features:
- BPE tokenization (8K-32K vocab)
- Multiple model sizes (50M-340M params)
- Resume training from checkpoints
- Automatic batch size adjustment
- Text generation during training
"""

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from ..models.cosf_transformer import COSFTransformer
from ..models.model_configs import get_cosf_config, get_recommended_batch_size
from ..data.token_dataset_fast import create_datasets_fast
from ..data.tokenizer import Tokenizer, BPETokenizer


@dataclass
class TrainConfig:
    """Training configuration."""
    # Model
    model_size: str = "small"  # tiny, small, medium, large
    vocab_size: int = 8000
    max_seq_len: int = 512

    # Data
    data_dir: str = "data/tinystories"
    tokenizer_path: str = "models/tokenizer_8k.model"

    # Training
    batch_size: Optional[int] = None  # Auto-set based on model size
    gradient_accumulation_steps: int = 4  # Simulate larger batch size
    seq_len: int = 512
    num_epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 2000

    # Checkpointing
    checkpoint_dir: str = "checkpoints/cosf_large"
    save_every_epochs: int = 1

    # Monitoring
    log_every_steps: int = 100
    eval_every_steps: int = 500
    generate_every_steps: int = 1000
    generate_length: int = 200

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Resume
    resume_from: Optional[str] = None
    start_epoch: int = 1


def get_lr_schedule(optimizer, warmup_steps: int):
    """Get learning rate scheduler with linear warmup."""

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


def generate_text(
    model: COSFTransformer,
    tokenizer,
    prompt: str,
    max_length: int = 200,
    temperature: float = 0.8,
    device: str = "cuda",
) -> str:
    """Generate text from prompt.

    Args:
        model: Trained model
        tokenizer: Tokenizer instance
        prompt: Starting text
        max_length: Maximum tokens to generate
        temperature: Sampling temperature
        device: Device to run on

    Returns:
        Generated text
    """
    model.eval()
    device_obj = torch.device(device)

    # Encode prompt
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device_obj)

    generated_ids = input_ids[0].tolist()

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
            generated_ids.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

            # Truncate if too long
            if input_ids.size(1) > model.config.max_seq_len:
                input_ids = input_ids[:, -model.config.max_seq_len:]

    # Decode
    generated_text = tokenizer.decode(generated_ids)
    return generated_text


def evaluate(model, val_loader, device):
    """Evaluate model on validation set.

    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        device: Device to run on

    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, model.config.vocab_size),
                y.reshape(-1),
            )

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

    avg_loss = total_loss / total_samples
    return avg_loss


def train(config: TrainConfig):
    """Main training function.

    Args:
        config: Training configuration
    """
    print("=" * 70)
    print("COSF LARGE TRANSFORMER TRAINING")
    print("=" * 70)
    print()

    # Setup device
    device = torch.device(config.device)
    print(f"Device: {device}")

    # Create checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print(f"\nLoading tokenizer from: {config.tokenizer_path}")
    try:
        # Try loading as BPE model
        tokenizer = BPETokenizer(config.tokenizer_path)
        print(f"Loaded BPE tokenizer with vocab size: {tokenizer.vocab_size}")
    except:
        # Fallback to unified tokenizer
        tokenizer = Tokenizer.from_pretrained(config.tokenizer_path)
        print(f"Loaded tokenizer with vocab size: {tokenizer.vocab_size}")

    # Update vocab size from tokenizer
    config.vocab_size = tokenizer.vocab_size

    # Load data
    data_dir = Path(config.data_dir)
    train_path = data_dir / "train.txt"
    val_path = data_dir / "val.txt"

    print(f"\nLoading data from: {data_dir}")
    train_dataset, val_dataset = create_datasets_fast(
        train_path=str(train_path),
        val_path=str(val_path),
        tokenizer=tokenizer,
        seq_len=config.seq_len,
        cache_dir="cache/tokenized",
    )

    # Set batch size if not specified
    if config.batch_size is None:
        config.batch_size = get_recommended_batch_size(
            config.model_size,
            seq_len=config.seq_len,
        )
        print(f"Auto-set batch size: {config.batch_size}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Windows compatibility
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Create model
    print(f"\nCreating {config.model_size} model...")
    model_config = get_cosf_config(
        model_size=config.model_size,
        vocab_size=config.vocab_size,
        max_seq_len=config.max_seq_len,
    )

    model = COSFTransformer(model_config).to(device)

    param_counts = model.count_parameters()
    print(f"Model parameters: {param_counts['total']:,}")
    print(f"  Token embedding: {param_counts['token_embedding']:,}")
    print(f"  Transformer blocks: {param_counts['transformer_blocks']:,}")
    print(f"  Layer norm: {param_counts['ln_f']:,}")
    print(f"  Positional encoding: {param_counts['pos_encoding']:,}")

    # Optimizer and scheduler
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
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {checkpoint['epoch']}, will start at epoch {start_epoch}")

        if 'val_loss' in checkpoint:
            best_val_loss = checkpoint.get('best_val_loss', checkpoint['val_loss'])
            print(f"Previous best validation loss: {best_val_loss:.4f}")

        print(f"Resumed at step {step}")

    # Print configuration
    print("\n" + "=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    print(f"Model size: {config.model_size}")
    print(f"Parameters: {param_counts['total']:,}")
    print(f"Vocab size: {config.vocab_size}")
    print(f"Sequence length: {config.seq_len}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Warmup steps: {config.warmup_steps}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Device: {device}")
    if config.resume_from:
        print(f"Resuming from: {config.resume_from}")
    print()

    # Training loop
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(start_epoch, start_epoch + config.num_epochs):
        model.train()
        epoch_loss = 0
        epoch_start = time.time()

        # Progress bar for this epoch
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{start_epoch + config.num_epochs - 1}",
                    unit="batch", ncols=100)

        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)

            # Forward pass
            logits = model(x)

            # Compute loss (scale by accumulation steps)
            loss = F.cross_entropy(
                logits.reshape(-1, config.vocab_size),
                y.reshape(-1),
            ) / config.gradient_accumulation_steps

            # Backward pass (accumulate gradients)
            loss.backward()

            # Only update weights every N steps
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * config.gradient_accumulation_steps
            step += 1

            # Update progress bar with current loss and PPL
            ppl = math.exp(min(loss.item(), 10))
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'ppl': f'{ppl:.2f}'})

            # Logging
            if step % config.log_every_steps == 0:
                lr = optimizer.param_groups[0]['lr']
                tqdm.write(f"Step {step:6d} | Loss: {loss.item():.4f} | PPL: {ppl:7.2f} | LR: {lr:.6f}")

            # Evaluation
            if step % config.eval_every_steps == 0:
                val_loss = evaluate(model, val_loader, device)
                val_ppl = math.exp(min(val_loss, 10))
                print(f"  Validation: Loss {val_loss:.4f}, PPL {val_ppl:.2f}")
                model.train()

            # Generation
            if step % config.generate_every_steps == 0:
                print("\n" + "-" * 70)
                prompt = "Once upon a time, there was a"
                try:
                    generated = generate_text(
                        model,
                        tokenizer,
                        prompt,
                        max_length=config.generate_length,
                        temperature=0.8,
                        device=config.device,
                    )
                    print("Generated sample:")
                    try:
                        print(generated[:500])  # Limit output length
                    except UnicodeEncodeError:
                        print(generated.encode('ascii', errors='replace').decode('ascii')[:500])
                except Exception as e:
                    print(f"Generation failed: {e}")
                print("-" * 70 + "\n")
                model.train()

        # Close progress bar
        pbar.close()

        # Epoch end - evaluate and save
        epoch_time = time.time() - epoch_start
        avg_train_loss = epoch_loss / len(train_loader)

        print(f"\nEpoch {epoch}/{start_epoch + config.num_epochs - 1} ({epoch_time:.1f}s):")
        print(f"  Train Loss: {avg_train_loss:.4f}")

        # Validation
        val_loss = evaluate(model, val_loader, device)
        val_ppl = math.exp(min(val_loss, 10))
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val PPL:    {val_ppl:.2f}")

        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'config': model_config,
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = checkpoint_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'step': step,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'config': model_config,
            }, best_path)
            print(f"New best model! Val loss: {val_loss:.4f}")

        print()

    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation PPL: {math.exp(min(best_val_loss, 10)):.2f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Train large COSF Transformer")

    # Model
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=["tiny", "small", "medium", "large"],
        help="Model size (default: small)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)",
    )

    # Data
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/tinystories",
        help="Data directory with train.txt and val.txt (default: data/tinystories)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="models/tokenizer_8k.model",
        help="Path to tokenizer model (default: models/tokenizer_8k.model)",
    )

    # Training
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default: auto based on model size)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Training sequence length (default: 512)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of epochs (default: 20)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=2000,
        help="Warmup steps (default: 2000)",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/cosf_large",
        help="Checkpoint directory (default: checkpoints/cosf_large)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)",
    )

    args = parser.parse_args()

    # Create config
    config = TrainConfig(
        model_size=args.model_size,
        max_seq_len=args.max_seq_len,
        data_dir=args.data_dir,
        tokenizer_path=args.tokenizer,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume,
        device=args.device,
    )

    # Train
    train(config)


if __name__ == "__main__":
    main()
