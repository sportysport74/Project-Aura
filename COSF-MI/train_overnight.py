"""
Train COSF Oscillatory Model with AURA LEVEL 1: THE HEARTBEAT

This implements Kuramoto coherence loss to create global synchronization.

AURA LEVEL 1 FEATURES:
  - Kuramoto coherence loss (creates the "pulse")
  - Drift-gated coupling (sync when stable, explore when chaotic)
  - Multi-intent data with bridge words
  - Enhanced warmup (45k steps for heartbeat formation)
  - Coherence tracking and metrics

ALL COSF FEATURES PRESERVED:
  - Oscillators
  - Chirality
  - Homeostatic sleep (gentle)
  - Julia bounds
  - Phase coherence
  - Style token conditioning

Usage:
    python train_with_aura.py --model-size tiny --epochs 3 --batch-size 8
"""

import argparse
import os
import sys
import time
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


def train_with_gentle_sleep(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    num_epochs,
    device,
    sleep_hook,
    checkpoint_dir,
    decay_rate,
    start_epoch=0,
    best_val_loss=float('inf'),
    skip_batches=0,
    global_step=0,
    save_every=3000,
    max_steps=0,
    log_every=0,
    amp_dtype=None,
    eval_batches=0,
):
    """Train model with gentle sleep cycles.

    Every `save_every` steps the full state goes to latest.pt, so a crash
    (e.g. a Windows display reset killing the CUDA context) loses minutes, not
    hours. Each epoch's data order is fixed by its seed, so a resume can skip
    exactly the batches already done.
    """

    model.train()
    current_epoch = 0
    dataset, batch_size = train_loader.dataset, train_loader.batch_size
    window_start, window_tokens = time.time(), 0

    for epoch in range(start_epoch, num_epochs):
        current_epoch = epoch
        print()
        print("=" * 70)
        print(f"EPOCH {epoch + 1}/{num_epochs}")
        print("=" * 70)
        print()

        # Adjust warmup for heartbeat formation (AURA LEVEL 1)
        if epoch == 0:
            original_min_steps = sleep_hook.min_steps_awake
            original_max_steps = sleep_hook.max_steps_awake
            sleep_hook.min_steps_awake = 45000  # Extended warmup for heartbeat
            sleep_hook.max_steps_awake = 50000  # Prevent early sleep
            print(f"EPOCH 1 WARMUP (AURA): min_steps = 45000, max_steps = 50000")
            print(f"  (Extended warmup to allow heartbeat formation)")
            print()

        epoch_loss = 0
        order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(1000 + epoch))
        skip = skip_batches if epoch == start_epoch else 0
        epoch_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, sampler=order[skip * batch_size:].tolist(), num_workers=0)
        if skip:
            print(f"Resuming epoch {epoch + 1} at batch {skip} (global step {global_step})")
        pbar = tqdm(epoch_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100,
                    initial=skip, total=len(train_loader))

        for batch_idx, batch in enumerate(pbar, start=skip):
            # Get batch
            if isinstance(batch, dict):
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
            else:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)

            # Forward with Kuramoto coherence loss (AURA LEVEL 1).
            # Under bf16 the model keeps oscillators and losses in fp32 itself.
            with torch.autocast(inputs.device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                logits, task_loss, sync_loss, coherence = model(inputs, targets, return_coherence_loss=True)

            # Total loss = task loss + sync loss
            total_loss = task_loss + sync_loss

            # Backward
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # pre-clip norm, logged below
            optimizer.step()
            model.apply_julia_correction()  # damping moved out of forward()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += task_loss.item()
            global_step += 1

            # Plain log line (tqdm's \r output is hard to parse from a log file)
            if log_every:
                window_tokens += inputs.numel()
                if global_step % log_every == 0:
                    torch.cuda.synchronize() if inputs.is_cuda else None
                    dt = time.time() - window_start
                    print(f"step={global_step} loss={task_loss.item():.4f} sync={float(sync_loss):.5f} coh={coherence:.3f} "
                          f"lr={scheduler.get_last_lr()[0]:.2e} step_ms={dt / log_every * 1000:.1f} "
                          f"tok/s={window_tokens / dt:.0f} gnorm={float(grad_norm):.4f}", flush=True)
                    window_start, window_tokens = time.time(), 0

            if max_steps and global_step >= max_steps:
                peak = torch.cuda.max_memory_allocated() / 2**30 if inputs.is_cuda else 0.0
                print(f"Reached --max-steps {max_steps}; stopping early (peak VRAM {peak:.2f} GiB)", flush=True)
                if eval_batches:
                    val_loss = evaluate(model, val_loader, device, amp_dtype, max_batches=eval_batches)
                    print(f"val_loss={val_loss:.4f} (first {eval_batches} val batches)", flush=True)
                return

            # Update progress bar with coherence
            pbar.set_postfix({
                'loss': f'{task_loss.item():.4f}',
                'ppl': f'{math.exp(task_loss.item()):.2f}',
                'coh': f'{coherence:.3f}',
            })

            # Check for sleep (every 10 steps to reduce overhead)
            if global_step % 10 == 0:
                # Custom sleep trigger with gentle mode
                if sleep_hook.check_and_sleep(model, global_step, task_loss.item()):
                    # Sleep was triggered - apply gentle reset
                    # Note: sleep_hook already called model.apply_homeostatic_reset()
                    # We just need to configure the decay rate
                    pass

            if global_step % save_every == 0:
                save_latest(model, optimizer, scheduler, epoch, batch_idx + 1, global_step,
                            best_val_loss, checkpoint_dir)

        # Restore original sleep settings after first epoch
        if epoch == 0:
            sleep_hook.min_steps_awake = original_min_steps
            sleep_hook.max_steps_awake = original_max_steps
            print()
            print(f"EPOCH 1 COMPLETE: Restoring sleep settings (min={original_min_steps}, max={original_max_steps})")
            print()

        # Epoch summary
        avg_loss = epoch_loss / max(1, len(train_loader) - skip)
        print()
        print(f"Epoch {epoch+1} complete:")
        print(f"  Avg train loss: {avg_loss:.4f}")
        print(f"  Avg train PPL: {math.exp(avg_loss):.2f}")

        # Validation
        val_loss = evaluate(model, val_loader, device, amp_dtype)
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
            print(f"  [OK] New best model saved (val_loss: {val_loss:.4f})")

        # Epoch done: resume point is the start of the next epoch
        save_latest(model, optimizer, scheduler, epoch + 1, 0, global_step, best_val_loss, checkpoint_dir)

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
    print()
    sleep_hook.print_summary()


def save_latest(model, optimizer, scheduler, epoch, batch_in_epoch, global_step, best_val_loss, checkpoint_dir):
    """Full training state for crash recovery (written atomically)."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state = {
        'epoch': epoch,
        'batch_in_epoch': batch_in_epoch,
        'global_step': global_step,
        'best_val_loss': best_val_loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': model.config,
    }
    tmp = checkpoint_dir / "latest.pt.tmp"
    torch.save(state, tmp)
    os.replace(tmp, checkpoint_dir / "latest.pt")


def evaluate(model, val_loader, device, amp_dtype=None, max_batches=0):
    """Evaluate model on validation set (only the first max_batches if > 0)."""
    model.eval()
    total_loss = 0
    n = 0

    with torch.no_grad():
        for batch in val_loader:
            if max_batches and n >= max_batches:
                break
            n += 1
            if isinstance(batch, dict):
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
            else:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)

            # During validation, don't compute coherence loss (saves time)
            with torch.autocast(inputs.device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                logits, loss = model(inputs, targets, return_coherence_loss=False)
            total_loss += loss.item()

    model.train()
    return total_loss / max(n, 1)


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
    parser = argparse.ArgumentParser(description="Train COSF with Gentle Sleep")

    # Model args
    parser.add_argument('--model-size', type=str, default='tiny',
                        choices=['tiny', 'mid', 'big', 'small', 'medium'],
                        help='Model size')

    # Training args
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs (5 for overnight)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')

    # Gentle sleep args
    parser.add_argument('--drift-threshold', type=float, default=1.0,
                        help='Drift threshold for sleep (lowered from 2.0)')
    parser.add_argument('--min-steps-awake', type=int, default=2000,
                        help='Min steps between sleeps (increased for stability)')
    parser.add_argument('--max-steps-awake', type=int, default=10000,
                        help='Force sleep after this many steps')
    parser.add_argument('--decay-rate', type=float, default=0.5,
                        help='Gentle sleep decay rate (0.0-1.0)')

    # Data args
    parser.add_argument('--data-dir', type=str, default='data/tinystories',
                        help='Data directory')
    parser.add_argument('--tokenizer', type=str, default='models/tokenizer_8k.model',
                        help='Tokenizer path')

    # Other args
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/overnight_bridge_words',
                        help='Checkpoint directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--warmup', type=int, default=0,
                        help='Linear LR warmup steps (use ~200 with batch 32)')
    parser.add_argument('--heartbeat', type=str, default='live', choices=['live', 'static', 'live_control'],
                        help="Kuramoto heartbeat on live per-token phases, or the old static offsets")
    parser.add_argument('--heartbeat-strength', type=float, default=0.1,
                        help='Heartbeat loss weight')
    parser.add_argument('--heartbeat-target', type=float, default=0.3,
                        help='Only lift live R up to this value (A/B: free at 0.3)')
    parser.add_argument('--save-every', type=int, default=3000,
                        help='Steps between crash-recovery saves (latest.pt)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint to resume from (e.g. checkpoints/fixed_run_100ep/checkpoint_epoch_86.pt)')
    parser.add_argument('--config', type=str, default=None,
                        help='YAML file of argument defaults (keys like batch_size); command-line flags override it')
    parser.add_argument('--max-steps', type=int, default=0,
                        help='Stop after this many optimizer steps (0 = run all epochs; for smoke tests)')
    parser.add_argument('--log-every', type=int, default=0,
                        help='Print a plain "step=N loss=X tok/s=Y" line every N steps (0 = off)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed for model init and dropout (data order is already seeded per epoch)')
    parser.add_argument('--precision', type=str, default='bf16', choices=['fp32', 'bf16'],
                        help='bf16 autocast for attention and lm_head; oscillators and losses stay fp32')
    parser.add_argument('--eval-batches', type=int, default=0,
                        help='With --max-steps: report val loss on the first N val batches at the end')
    parser.add_argument('--matmul-precision', type=str, default='highest', choices=['highest', 'high'],
                        help="fp32 matmuls (the oscillators' Ω·x): 'highest' = true fp32, 'high' = TF32 tensor cores")
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, default=True,
                        help='torch.compile the model forward in place (checkpoint keys unchanged); --no-compile for eager')

    pre, _ = parser.parse_known_args()
    if pre.config:
        import yaml
        with open(pre.config) as f:
            defaults = {k.replace('-', '_'): v for k, v in (yaml.safe_load(f) or {}).items()}
        unknown = set(defaults) - {a.dest for a in parser._actions}
        if unknown:
            parser.error(f"unknown keys in {pre.config}: {', '.join(sorted(unknown))}")
        parser.set_defaults(**defaults)
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("OVERNIGHT TRAINING: AURA LEVEL 1 + BRIDGE WORDS")
    print("=" * 70)
    print()
    print("TRAINING CONFIGURATION:")
    print("  Dataset: Styled TinyStories ([STORY] tokens)")
    print("  Full dataset: 461k train, 24k val samples")
    print("  Epochs: 5 (extended overnight training)")
    print("  Warmup: 45k steps (heartbeat formation)")
    print()
    print("AURA LEVEL 1 features:")
    print("  [NEW] Kuramoto coherence loss (creates global pulse)")
    print("  [NEW] Drift-gated coupling (adaptive synchronization)")
    print("  [NEW] Coherence tracking in progress bar")
    print()
    print("EXPECTED RESULTS:")
    print("  - Coherence rises: 0.01 -> 0.20+")
    print("  - Chirality flips on bridge words")
    print("  - Semantic differentiation emerges!")
    print()
    print("=" * 70)
    print()

    # Model configs
    model_configs = {
        'tiny': {'d_model': 256, 'n_layers': 4, 'n_heads': 4, 'd_ff': 3072},
        # Oct 9 scale contest (experiments/run_scale.sh): best val loss per wall-clock
        'mid': {'d_model': 512, 'n_layers': 6, 'n_heads': 8, 'd_ff': 2048},
        'big': {'d_model': 512, 'n_layers': 8, 'n_heads': 8, 'd_ff': 2048},
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
    print("Gentle sleep settings:")
    print(f"  Mode: GENTLE (structure-preserving)")
    print(f"  Drift threshold: {args.drift_threshold} (lowered for earlier sleep)")
    print(f"  Min steps awake: {args.min_steps_awake} (epoch 1: 20000)")
    print(f"  Max steps awake: {args.max_steps_awake}")
    print(f"  Decay rate: {args.decay_rate} (0.5 = balanced)")
    print()
    print("Key improvements:")
    print("  1. Gentle sleep preserves phase relationships")
    print("  2. Gentle sleep preserves chirality orientation")
    print("  3. Lower drift threshold (earlier sleep)")
    print("  4. Warmup period (no sleep in epoch 1)")
    print()

    # Device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cpu':
        print("WARNING: Training on CPU will be very slow!")
    print()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        print(f"Seed: {args.seed}")
    torch.set_float32_matmul_precision(args.matmul_precision)
    print(f"Precision: {args.precision}, fp32 matmul precision: {args.matmul_precision}, compile: {args.compile}")

    # Create model
    print("Creating model with CHIRALITY enabled...")
    config = COSFOscillatorConfig(
        vocab_size=BPETokenizer(args.tokenizer).vocab_size,
        d_model=model_config['d_model'],
        n_layers=model_config['n_layers'],
        n_heads=model_config['n_heads'],
        d_ff=model_config.get('d_ff', 3072),
        max_seq_len=256,
        homeostatic_decay_rate=0.01,
        enable_homeostasis=True,
        enable_refractory=False,
        enable_chirality=True,
        chirality_init_prob=0.5,
        enable_terahertz_scaling=False,
        heartbeat_mode=args.heartbeat,
        heartbeat_strength=args.heartbeat_strength,
        heartbeat_target=args.heartbeat_target,
    )

    model = COSFOscillatorTransformer(config).to(device)
    if args.compile:
        model.compile()  # nn.Module.compile: state_dict keys stay free of the _orig_mod. prefix

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print()
    print("Chirality integration: ACTIVE")
    print("  All oscillators have learnable handedness")
    print("  Gentle sleep will preserve chirality orientation")
    print()

    # Load data
    print("Loading data...")
    tokenizer = BPETokenizer(args.tokenizer)

    # Use styled TinyStories data for overnight training
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
    # Phase offsets are angles: weight decay would drag them all toward 0
    # (fake synchrony, and a constant phase drift). Keep them out of it.
    phase_params = [p for n, p in model.named_parameters() if n.endswith(".phases")]
    other_params = [p for n, p in model.named_parameters() if not n.endswith(".phases")]
    optimizer = torch.optim.AdamW(
        [{"params": other_params}, {"params": phase_params, "weight_decay": 0.0}],
        lr=args.lr,
    )
    total_steps = len(train_loader) * args.epochs
    if args.warmup:
        # Linear warmup, then the same cosine decay
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (
            1 + math.cos(math.pi * min(1.0, s / total_steps))))
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    skip_batches = global_step = 0
    latest = Path(args.checkpoint_dir) / "latest.pt"
    if args.resume == 'auto' and not latest.exists():
        args.resume = None
    if args.resume == 'auto':
        print(f"Resuming from crash-recovery state: {latest}")
        state = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        start_epoch, skip_batches = state['epoch'], state['batch_in_epoch']
        global_step, best_val_loss = state['global_step'], state['best_val_loss']
        print(f"  epoch {start_epoch + 1}, batch {skip_batches}, step {global_step}, "
              f"lr={optimizer.param_groups[0]['lr']:.2e}, best val_loss={best_val_loss:.4f}")
        print()
    elif args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

        # Scheduler state isn't saved, so fast-forward it to where it was
        resume_steps = len(train_loader) * start_epoch
        scheduler.last_epoch = resume_steps
        for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
            group['lr'] = base_lr * (1 + math.cos(math.pi * resume_steps / total_steps)) / 2

        best_path = Path(args.resume).parent / "best_model.pt"
        if best_path.exists():
            best_val_loss = torch.load(best_path, map_location='cpu', weights_only=False)['val_loss']
        else:
            best_val_loss = checkpoint['val_loss']
        print(f"  Starting at epoch {start_epoch + 1}, lr={optimizer.param_groups[0]['lr']:.2e}, best val_loss={best_val_loss:.4f}")
        print()

    # Create sleep cycle hook with GENTLE sleep
    print("Initializing gentle sleep hook...")
    sleep_hook = SleepCycleHook(
        drift_threshold=args.drift_threshold,
        min_steps_awake=args.min_steps_awake,
        max_steps_awake=args.max_steps_awake,
        measure_consolidation=True,
        verbose=True,
    )
    print()

    # Train
    train_with_gentle_sleep(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        device=device,
        sleep_hook=sleep_hook,
        checkpoint_dir=args.checkpoint_dir,
        decay_rate=args.decay_rate,
        start_epoch=start_epoch,
        best_val_loss=best_val_loss,
        skip_batches=skip_batches,
        global_step=global_step,
        save_every=args.save_every,
        max_steps=args.max_steps,
        log_every=args.log_every,
        amp_dtype=torch.bfloat16 if args.precision == 'bf16' else None,
        eval_batches=args.eval_batches,
    )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print()
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print()
    print("Next steps:")
    print("  1. Review sleep cycle summary above")
    print("  2. Check consolidation success rate")
    print("  3. Analyze chirality evolution")
    print("  4. Test on ambiguous words")
    print()


if __name__ == "__main__":
    main()
