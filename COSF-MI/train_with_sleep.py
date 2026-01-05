"""
Train COSF Oscillatory Model with Automatic Sleep Cycles

This script trains the oscillatory transformer with automatic homeostatic reset
("neural sleep") when the model becomes fatigued (high drift).

Features:
    - Automatic sleep when drift exceeds threshold
    - Memory consolidation tracking
    - Sleep cycle statistics
    - Configurable sleep parameters

Usage:
    python train_with_sleep.py --model-size tiny --epochs 3

    python train_with_sleep.py --model-size small --epochs 10 --drift-threshold 2.0
"""

import argparse
import sys
import torch
import math
from pathlib import Path
from tqdm import tqdm

# Add COSF-MI to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "python"))

from cosf_mi.models.cosf_oscillator import COSFOscillatorTransformer, COSFOscillatorConfig
from cosf_mi.data.tokenizer import BPETokenizer
from cosf_mi.data.token_dataset_fast import create_datasets_fast
from cosf_mi.training.sleep_cycle import SleepCycleHook


def train_with_sleep_cycles(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    num_epochs,
    device,
    sleep_hook,
    checkpoint_dir,
):
    """Train model with automatic sleep cycles."""

    model.train()
    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print()
        print("=" * 70)
        print(f"EPOCH {epoch + 1}/{num_epochs}")
        print("=" * 70)
        print()

        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100)

        for batch_idx, batch in enumerate(pbar):
            # Get batch
            if isinstance(batch, dict):
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
            else:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)

            # Forward
            logits, loss = model(inputs, targets)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # NOTE: Homeostatic decay is NOT applied during regular training
            # It only happens during sleep (homeostatic reset via sleep_hook)
            # Applying decay every step causes gradient version conflicts

            epoch_loss += loss.item()
            global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'ppl': f'{math.exp(loss.item()):.2f}',
            })

            # Check for sleep (every 10 steps to reduce overhead)
            if global_step % 10 == 0:
                sleep_hook.check_and_sleep(model, global_step, loss.item())

        # Epoch summary
        avg_loss = epoch_loss / len(train_loader)
        print()
        print(f"Epoch {epoch+1} complete:")
        print(f"  Avg train loss: {avg_loss:.4f}")
        print(f"  Avg train PPL: {math.exp(avg_loss):.2f}")

        # Validation
        val_loss = evaluate(model, val_loader, device)
        print(f"  Val loss: {val_loss:.4f}")
        print(f"  Val PPL: {math.exp(val_loss):.2f}")

        # Phase metrics
        metrics = model.get_phase_coherence_metrics()
        print()
        print("Phase metrics:")
        print(f"  Coherence: {metrics['mean_coherence']:.4f}")
        print(f"  Max velocity: {metrics['max_phase_velocity']:.2f}")
        print(f"  Homeostatic drift: {metrics['homeostatic_drift']:.4f}")
        print(f"  Refractory suppressions: {metrics['refractory_suppressions']}")

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                checkpoint_dir=checkpoint_dir,
                is_best=True,
            )
            print(f"  ✅ New best model saved (val_loss: {val_loss:.4f})")

        # Save periodic checkpoint
        if (epoch + 1) % 2 == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                checkpoint_dir=checkpoint_dir,
                is_best=False,
            )

    # Final sleep cycle summary
    sleep_hook.print_summary()


def evaluate(model, val_loader, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, dict):
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
            else:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)

            logits, loss = model(inputs, targets)
            total_loss += loss.item()

    model.train()
    return total_loss / len(val_loader)


def save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir, is_best):
    """Save model checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'config': model.config,
    }

    # Save epoch checkpoint
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
    torch.save(checkpoint, checkpoint_path)

    # Save best model
    if is_best:
        best_path = checkpoint_dir / "best_model.pt"
        torch.save(checkpoint, best_path)


def main():
    parser = argparse.ArgumentParser(description="Train COSF Oscillatory Model with Sleep Cycles")

    # Model args
    parser.add_argument('--model-size', type=str, default='tiny',
                        choices=['tiny', 'small', 'medium'],
                        help='Model size')

    # Training args
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')

    # Sleep cycle args
    parser.add_argument('--drift-threshold', type=float, default=2.0,
                        help='Drift threshold for sleep')
    parser.add_argument('--min-steps-awake', type=int, default=500,
                        help='Min steps between sleeps')
    parser.add_argument('--max-steps-awake', type=int, default=5000,
                        help='Force sleep after this many steps')

    # Data args
    parser.add_argument('--data-dir', type=str, default='data/tinystories',
                        help='Data directory')
    parser.add_argument('--tokenizer', type=str, default='models/tokenizer_8k.model',
                        help='Tokenizer path')

    # Other args
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/cosf_oscillatory',
                        help='Checkpoint directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')

    args = parser.parse_args()

    print()
    print("=" * 70)
    print("TRAINING COSF OSCILLATORY MODEL WITH SLEEP CYCLES")
    print("=" * 70)
    print()

    # Model configs
    model_configs = {
        'tiny': {'d_model': 256, 'n_layers': 4, 'n_heads': 4},
        'small': {'d_model': 768, 'n_layers': 12, 'n_heads': 12},
        'medium': {'d_model': 1024, 'n_layers': 16, 'n_heads': 16},
    }

    model_config = model_configs[args.model_size]

    print("Configuration:")
    print(f"  Model size: {args.model_size}")
    print(f"  d_model: {model_config['d_model']}")
    print(f"  n_layers: {model_config['n_layers']}")
    print(f"  n_heads: {model_config['n_heads']}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print()
    print("Sleep cycle settings:")
    print(f"  Drift threshold: {args.drift_threshold}")
    print(f"  Min steps awake: {args.min_steps_awake}")
    print(f"  Max steps awake: {args.max_steps_awake}")
    print()

    # Device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cpu':
        print("WARNING: Training on CPU will be very slow!")
    print()

    # Create model
    print("Creating model...")
    config = COSFOscillatorConfig(
        vocab_size=8000,
        d_model=model_config['d_model'],
        n_layers=model_config['n_layers'],
        n_heads=model_config['n_heads'],
        max_seq_len=256,  # Reduced from 512 for system safety
        homeostatic_decay_rate=0.01,
        enable_homeostasis=True,
        enable_refractory=False,  # Disabled to reduce computation load
    )

    model = COSFOscillatorTransformer(config).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print()

    # Load data
    print("Loading data...")
    tokenizer = BPETokenizer(args.tokenizer)

    train_dataset, val_dataset = create_datasets_fast(
        train_path=f"{args.data_dir}/train.txt",
        val_path=f"{args.data_dir}/val.txt",
        tokenizer=tokenizer,
        seq_len=256,  # Reduced from 512 for system safety
        cache_dir="cache/tokenized",
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print()

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

    # Create sleep cycle hook
    print("Initializing sleep cycle hook...")
    sleep_hook = SleepCycleHook(
        drift_threshold=args.drift_threshold,
        min_steps_awake=args.min_steps_awake,
        max_steps_awake=args.max_steps_awake,
        measure_consolidation=True,
        verbose=True,
    )
    print()

    # Train
    train_with_sleep_cycles(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        device=device,
        sleep_hook=sleep_hook,
        checkpoint_dir=args.checkpoint_dir,
    )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print()
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print()


if __name__ == "__main__":
    main()
