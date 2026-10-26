"""
Continue Training from Checkpoint

This script resumes training from an existing checkpoint without starting over.

Usage:
    python continue_training.py --resume checkpoints/styled_cosf/checkpoint_epoch_4.pt --epochs 5
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


def train_epoch(model, train_loader, optimizer, scheduler, epoch, num_epochs, device, sleep_hook, global_step):
    """Train for one epoch."""
    model.train()
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

        # Forward (disable coherence loss since this script doesn't track it)
        logits, loss = model(inputs, targets, return_coherence_loss=False)

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.apply_julia_correction()  # damping moved out of forward()
        scheduler.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()
        global_step += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ppl': f'{math.exp(loss.item()):.2f}',
        })

        # Check for sleep (every 10 steps)
        if global_step % 10 == 0:
            sleep_hook.check_and_sleep(model, global_step, loss.item())

    avg_loss = epoch_loss / len(train_loader)
    return avg_loss, global_step


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

            logits, loss = model(inputs, targets, return_coherence_loss=False)
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
    parser = argparse.ArgumentParser(description="Continue COSF Training from Checkpoint")

    # Resume args
    parser.add_argument('--resume', type=str, required=True,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of ADDITIONAL epochs to train')

    # Training args (will be reused from checkpoint, but can override)
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')

    # Gentle sleep args
    parser.add_argument('--drift-threshold', type=float, default=1.0,
                        help='Drift threshold for sleep')
    parser.add_argument('--min-steps-awake', type=int, default=2000,
                        help='Min steps between sleeps')
    parser.add_argument('--max-steps-awake', type=int, default=10000,
                        help='Force sleep after this many steps')

    # Data args
    parser.add_argument('--data-dir', type=str, default='data/tinystories',
                        help='Data directory')
    parser.add_argument('--tokenizer', type=str, default='models/tokenizer_8k.model',
                        help='Tokenizer path')

    # Other args
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/styled_cosf',
                        help='Checkpoint directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')

    args = parser.parse_args()

    print()
    print("=" * 70)
    print("CONTINUE TRAINING FROM CHECKPOINT")
    print("=" * 70)
    print()

    # Load checkpoint
    print(f"Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)

    start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
    best_val_loss = checkpoint['val_loss']

    print(f"  Previous epoch: {checkpoint['epoch'] + 1}")
    print(f"  Previous val loss: {checkpoint['val_loss']:.4f}")
    print(f"  Previous perplexity: {math.exp(checkpoint['val_loss']):.2f}")
    print(f"  Resuming from epoch: {start_epoch + 1}")
    print()

    # Create model from checkpoint config
    config = checkpoint['config']
    model = COSFOscillatorTransformer(config)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Model loaded:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  d_model: {config.d_model}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  Chirality: {'ENABLED' if config.enable_chirality else 'DISABLED'}")
    print()

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Device: {device}")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = BPETokenizer(args.tokenizer)
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print()

    # Load data (use styled data)
    print("Loading styled TinyStories dataset...")
    train_dataset, val_dataset = create_datasets_fast(
        train_path=f"{args.data_dir}/styled_train.txt",
        val_path=f"{args.data_dir}/styled_val.txt",
        tokenizer=tokenizer,
        seq_len=256,
        cache_dir="cache/tokenized_styled",
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print(f"  Train samples: {len(train_dataset):,}")
    print(f"  Val samples: {len(val_dataset):,}")
    print(f"  Batches per epoch: {len(train_loader):,}")
    print()

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1
    )

    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Optimizer state loaded from checkpoint")
    print()

    # Create scheduler (cosine with warmup)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = min(500, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"Scheduler created:")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Warmup steps: {warmup_steps:,}")
    print()

    # Create sleep hook
    sleep_hook = SleepCycleHook(
        drift_threshold=args.drift_threshold,
        min_steps_awake=args.min_steps_awake,
        max_steps_awake=args.max_steps_awake,
        measure_consolidation=True,
        verbose=True,
    )

    print("Gentle sleep configured:")
    print(f"  Min steps awake: {args.min_steps_awake}")
    print(f"  Max steps awake: {args.max_steps_awake}")
    print(f"  Drift threshold: {args.drift_threshold}")
    print()

    print("=" * 70)
    print(f"TRAINING FOR {args.epochs} MORE EPOCHS")
    print("=" * 70)
    print()

    # Training loop
    global_step = 0  # Reset global step counter

    for epoch in range(args.epochs):
        current_epoch = start_epoch + epoch

        print()
        print("=" * 70)
        print(f"EPOCH {current_epoch + 1} (continued)")
        print("=" * 70)
        print()

        # Train
        avg_loss, global_step = train_epoch(
            model, train_loader, optimizer, scheduler,
            current_epoch, start_epoch + args.epochs,
            device, sleep_hook, global_step
        )

        # Epoch summary
        print()
        print(f"Epoch {current_epoch + 1} complete:")
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

        # Chirality metrics (if enabled)
        if metrics.get('chirality_enabled', False):
            print()
            print("Chirality metrics:")
            print(f"  Right-handed oscillators: {metrics['total_right_handed']}")
            print(f"  Left-handed oscillators: {metrics['total_left_handed']}")
            print(f"  Balance (right ratio): {metrics['chirality_balance']:.3f}")
            print(f"  Total perspective flips: {metrics['total_chirality_flips']}")

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            print(f"  [OK] New best model saved (val_loss: {val_loss:.4f})")

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=current_epoch,
            val_loss=val_loss,
            checkpoint_dir=args.checkpoint_dir,
            is_best=is_best,
        )

    # Final summary
    print()
    print("=" * 70)
    print("CONTINUED TRAINING COMPLETE")
    print("=" * 70)
    print()
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best perplexity: {math.exp(best_val_loss):.2f}")
    print()
    sleep_hook.print_summary()
    print()


if __name__ == "__main__":
    main()
