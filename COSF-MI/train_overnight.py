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
import json
import os
import random
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
from cosf_mi.data.memmap_dataset import load_corpus
from cosf_mi.training.sleep_cycle import SleepCycleHook

PAUSE_FLAG = Path(__file__).parent / "run_control" / "PAUSE"  # created by scripts/pause_train.sh
PAUSE_EXIT = 75  # exit code for "paused on request" (EX_TEMPFAIL); queues treat it as a clean stop


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
    save_final=False,
    sleep=True,
    save_every_minutes=0.0,
    save_at_steps=(),
):
    """Train model with gentle sleep cycles.

    Every `save_every` steps the full state goes to latest.pt, so a crash
    (e.g. a Windows display reset killing the CUDA context) loses minutes, not
    hours. Each epoch's data order is fixed by its seed, so a resume can skip
    exactly the batches already done. save_every_minutes adds a time-based save
    (whichever comes first), and save_at_steps keeps extra full-state snapshots
    (state_step<N>.pt) to branch from.
    """
    last_save = time.time()

    model.train()
    settle_on = getattr(model.config, 'settle_mode', 'none') != 'none'
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
            dataset, batch_size=batch_size, sampler=order[skip * batch_size:].tolist(), num_workers=0,
            pin_memory=(device == 'cuda'))  # pinned + non_blocking: the copy doesn't stall the CPU
        if skip:
            print(f"Resuming epoch {epoch + 1} at batch {skip} (global step {global_step})")
        pbar = tqdm(epoch_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100,
                    initial=skip, total=len(train_loader))

        for batch_idx, batch in enumerate(pbar, start=skip):
            # Get batch
            if isinstance(batch, dict):
                inputs = batch['input'].to(device, non_blocking=True)
                targets = batch['target'].to(device, non_blocking=True)
            else:
                inputs, targets = batch
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

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
            if settle_on:  # settling module: criticality controller + per-step diagnostics (outside the compiled forward)
                diag = model.settle_after_step(global_step + 1, bool(log_every) and (global_step + 1) % log_every == 0)
                if diag:
                    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
                    with open(Path(checkpoint_dir) / "settle_diag.jsonl", "a") as f:
                        f.write(json.dumps(diag) + "\n")
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += task_loss.detach().double()  # summed on the GPU (float64, same as the old Python float)
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
                val_loss = None
                if eval_batches:
                    val_loss = evaluate(model, val_loader, device, amp_dtype, max_batches=eval_batches)
                    print(f"val_loss={val_loss:.4f} (first {eval_batches} val batches)", flush=True)
                if save_final:
                    torch.save({"config": model.config, "model_state_dict": model.state_dict(), "step": global_step,
                                "val_loss": val_loss}, Path(checkpoint_dir) / "final_model.pt")
                # full state at the stop too, so a run stopped with --max-steps can be continued or branched exactly
                save_latest(model, optimizer, scheduler, epoch, batch_idx + 1, global_step, best_val_loss, checkpoint_dir)
                return

            # Progress bar and sleep check every 10 steps: reading the loss is a GPU sync
            if global_step % 10 == 0:
                loss_value = task_loss.item()
                pbar.set_postfix({
                    'loss': f'{loss_value:.4f}',
                    'ppl': f'{math.exp(loss_value):.2f}',
                    'coh': f'{coherence:.3f}',
                })

                # Custom sleep trigger with gentle mode
                if sleep and sleep_hook.check_and_sleep(model, global_step, loss_value):
                    # Sleep was triggered - apply gentle reset
                    # Note: sleep_hook already called model.apply_homeostatic_reset()
                    # We just need to configure the decay rate
                    pass

            if global_step % save_every == 0 or (save_every_minutes and time.time() - last_save >= 60 * save_every_minutes):
                t_save = time.time()
                save_latest(model, optimizer, scheduler, epoch, batch_idx + 1, global_step,
                            best_val_loss, checkpoint_dir)
                last_save = time.time()
                print(f"saved latest.pt at step {global_step} ({last_save - t_save:.1f} s)", flush=True)
            # Pause request (scripts/pause_train.sh): save the full state and exit with code 75; the queue stops cleanly
            if global_step % 20 == 0 and PAUSE_FLAG.exists():
                save_latest(model, optimizer, scheduler, epoch, batch_idx + 1, global_step, best_val_loss, checkpoint_dir)
                print(f"PAUSED at step {global_step}: latest.pt saved in {checkpoint_dir}; resume with scripts/resume_train.sh",
                      flush=True)
                sys.exit(PAUSE_EXIT)
            if global_step in save_at_steps:
                save_latest(model, optimizer, scheduler, epoch, batch_idx + 1, global_step,
                            best_val_loss, checkpoint_dir, name=f"state_step{global_step}.pt")
                print(f"saved state_step{global_step}.pt", flush=True)

        # Restore original sleep settings after first epoch
        if epoch == 0:
            sleep_hook.min_steps_awake = original_min_steps
            sleep_hook.max_steps_awake = original_max_steps
            print()
            print(f"EPOCH 1 COMPLETE: Restoring sleep settings (min={original_min_steps}, max={original_max_steps})")
            print()

        # Epoch summary
        avg_loss = float(epoch_loss) / max(1, len(train_loader) - skip)
        print()
        print(f"Epoch {epoch+1} complete:")
        print(f"  Avg train loss: {avg_loss:.4f}")
        print(f"  Avg train PPL: {math.exp(avg_loss):.2f}")

        # Validation
        val_loss = evaluate(model, val_loader, device, amp_dtype)
        print(f"  Val loss: {val_loss:.4f}")
        print(f"  Val PPL: {math.exp(val_loss):.2f}")

        # Phase / chirality metrics: oscillator models only, and never fatal (2026-11-04: an AttributeError here on a vanilla2 model
        # crashed the run after its full epoch, before the checkpoint save below)
        if getattr(model.config, 'projection', 'oscillator') == 'linear':
            print("Phase metrics: n/a (linear projections)")
        else:
            try:
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

            except Exception as e:  # diagnostics only
                print(f"WARNING: phase metrics failed ({type(e).__name__}: {e}); continuing to the checkpoint save")

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


def save_latest(model, optimizer, scheduler, epoch, batch_in_epoch, global_step, best_val_loss, checkpoint_dir,
                name="latest.pt"):
    """Full training state for crash recovery: weights, optimizer, scheduler, data position and RNG states.
    Written to a temp file and renamed (atomic), and the previous latest.pt is kept as latest.prev.pt, so a crash
    during a save, or a bad file, still leaves a state to resume from."""
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
        'rng': {'torch': torch.get_rng_state(), 'python': random.getstate(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
        'saved_at': time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    tmp = checkpoint_dir / f"{name}.tmp"
    torch.save(state, tmp)
    fd = os.open(tmp, os.O_RDONLY)
    os.fsync(fd)  # on disk before the rename, so a power loss can't leave a renamed but empty file
    os.close(fd)
    if name == "latest.pt" and (checkpoint_dir / name).exists():
        os.replace(checkpoint_dir / name, checkpoint_dir / "latest.prev.pt")
    os.replace(tmp, checkpoint_dir / name)


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
                        choices=['tiny', 'mid', 'big', 'small', 'medium', 's25', 'm50', 'l100'],
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
    parser.add_argument('--projection', type=str, default='oscillator', choices=['oscillator', 'linear'],
                        help="linear = vanilla baseline: every oscillator projection replaced by nn.Linear of the same shape")
    parser.add_argument('--emb-scale', type=str, default='none', choices=['none', 'sqrt'],
                        help='sqrt: token embeddings × sqrt(d_model) before the positional encoding')
    parser.add_argument('--ffn-activation', type=str, default='tanh', choices=['tanh', 'gelu'])
    parser.add_argument('--linear-init', type=str, default='default', choices=['default', 'gpt'],
                        help='gpt: GPT-2 init for plain nn.Linear projections (only with --projection linear)')
    parser.add_argument('--settle', type=str, default='none', choices=['none', 'settle', 'repeat'],
                        help='settling module after each block (models/settling.py); repeat = the R4 control')
    parser.add_argument('--settle-T', type=int, default=8, help='settling steps per block')
    parser.add_argument('--settle-K', type=float, default=1.0, help='coupling strength (fixed unless the controller is on)')
    parser.add_argument('--settle-clamp', type=str, default='none', choices=['none', 'phi', 'two'])
    parser.add_argument('--settle-controller', type=str, default='none', choices=['none', 'auto'])
    parser.add_argument('--settle-repeats', type=int, default=1, help='R4 control: plain-block repeats (set for compute matching)')
    parser.add_argument('--deterministic', action='store_true',
                        help='torch.use_deterministic_algorithms (slow; for bitwise tests such as scripts/test_resume.sh)')
    parser.add_argument('--save-every-minutes', type=float, default=0,
                        help='Also save latest.pt every N minutes of wall time (0 = off), whichever comes first')
    parser.add_argument('--save-at-steps', type=str, default='',
                        help='Comma-separated steps at which to keep a full-state snapshot state_step<N>.pt (to branch from)')
    parser.add_argument('--resume-state', type=str, default=None,
                        help='Resume the full training state (weights, optimizer, scheduler, data position, RNG) from this file '
                             '(a latest.pt or state_step<N>.pt); --resume auto does the same with <checkpoint-dir>/latest.pt')
    parser.add_argument('--init-from', type=str, default=None,
                        help='Initialise the weights from this checkpoint (weights only: fresh optimizer and schedule). '
                             'Ignored when a training state is resumed')
    parser.add_argument('--schedule', type=str, default='cosine', choices=['cosine', 'wsd'],
                        help='cosine (default) or wsd: linear warmup, constant, then linear decay to 0 at the end')
    parser.add_argument('--decay-frac', type=float, default=0.1, help='wsd: decay over the last fraction of the steps')
    parser.add_argument('--decay-start-step', type=int, default=0,
                        help='wsd: start the decay at this step instead (e.g. a branch off a stable-phase snapshot)')
    parser.add_argument('--val-corpus', type=str, default=None,
                        help='Take the val split from this corpus dir instead of --corpus (same tokenizer required)')
    parser.add_argument('--decay-steps', type=int, default=0,
                        help='Cosine LR decay length in steps (0 = all epochs). Set it to --max-steps so a short run '
                             'finishes its schedule (scaling sweep)')
    parser.add_argument('--save-final', action='store_true',
                        help='With --max-steps: save the weights to <checkpoint-dir>/final_model.pt at the stop')
    parser.add_argument('--seq-len', type=int, default=256, help='Context length (model positions and data windows)')
    parser.add_argument('--corpus', type=str, default=None,
                        help='Pre-tokenized corpus dir (train.bin/val.bin/meta.json, scripts/corpus/tokenize_corpus.py); '
                             'replaces --data-dir and --tokenizer')
    parser.add_argument('--matmul-precision', type=str, default='highest', choices=['highest', 'high'],
                        help="fp32 matmuls (the oscillators' Ω·x): 'highest' = true fp32, 'high' = TF32 tensor cores")
    parser.add_argument('--sleep', action=argparse.BooleanOptionalAction, default=True,
                        help='Gentle sleep cycles (homeostatic reset). Epoch 1 forces one by step 50k, so runs longer than '
                             'that differ from the sweep (12k steps, never slept) unless --no-sleep')
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
        # Scaling sweep (plan 2026-10-28), d_ff = 4·d_model, head dim 64. Unique params with a 24k vocab:
        's25': {'d_model': 384, 'n_layers': 8, 'n_heads': 6, 'd_ff': 1536},     # 23.5M
        'm50': {'d_model': 512, 'n_layers': 12, 'n_heads': 8, 'd_ff': 2048},    # 50.2M
        'l100': {'d_model': 768, 'n_layers': 12, 'n_heads': 12, 'd_ff': 3072},  # 103.7M
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

    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # needed by deterministic cuBLAS; before the first CUDA call
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        print("Deterministic algorithms ON")

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

    corpus_meta = None
    if args.corpus:  # a pre-tokenized corpus brings its own tokenizer
        corpus_meta = json.loads((Path(args.corpus) / "meta.json").read_text())
        args.tokenizer = corpus_meta["tokenizer"]
        print(f"Corpus: {args.corpus} ({corpus_meta['train_tokens']:,} train tokens), tokenizer {args.tokenizer}")
    print(f"Sequence length: {args.seq_len}")

    # Create model
    print("Creating model with CHIRALITY enabled...")
    config = COSFOscillatorConfig(
        vocab_size=BPETokenizer(args.tokenizer).vocab_size,
        d_model=model_config['d_model'],
        n_layers=model_config['n_layers'],
        n_heads=model_config['n_heads'],
        d_ff=model_config.get('d_ff', 3072),
        max_seq_len=args.seq_len,
        homeostatic_decay_rate=0.01,
        enable_homeostasis=True,
        enable_refractory=False,
        enable_chirality=True,
        chirality_init_prob=0.5,
        enable_terahertz_scaling=False,
        heartbeat_mode=args.heartbeat,
        projection=args.projection,
        emb_scale=args.emb_scale,
        ffn_activation=args.ffn_activation,
        linear_init=args.linear_init,
        settle_mode=args.settle,
        settle_T=args.settle_T,
        settle_K=args.settle_K,
        settle_clamp=args.settle_clamp,
        settle_controller=args.settle_controller,
        settle_repeats=args.settle_repeats,
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

    if args.corpus:
        # Pre-tokenized uint16 memmap corpus (scripts/corpus/tokenize_corpus.py): stays on disk, not in RAM
        train_dataset, val_dataset, _ = load_corpus(args.corpus, args.seq_len)
        if args.val_corpus:
            _, val_dataset, vmeta = load_corpus(args.val_corpus, args.seq_len)
            assert vmeta["tokenizer"] == corpus_meta["tokenizer"], "--val-corpus must use the same tokenizer"
            print(f"Val split from {args.val_corpus} ({vmeta['val_tokens']:,} tokens)")
    else:
        # Styled TinyStories data (the original pretraining mix)
        train_dataset, val_dataset = create_datasets_fast(
            train_path=f"{args.data_dir}/styled_train.txt",
            val_path=f"{args.data_dir}/styled_val.txt",
            tokenizer=tokenizer,
            seq_len=args.seq_len,
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
    total_steps = args.decay_steps or len(train_loader) * args.epochs
    if args.schedule == 'wsd':
        decay_start = args.decay_start_step or int(total_steps * (1 - args.decay_frac))

        def wsd(s):
            if s < decay_start:
                return min(1.0, (s + 1) / args.warmup) if args.warmup else 1.0
            return max(0.0, 1 - (s - decay_start) / max(1, total_steps - decay_start))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, wsd)
        print(f"Schedule: WSD, warmup {args.warmup}, constant to step {decay_start:,}, linear to 0 at step {total_steps:,}")
    elif args.warmup:
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
    if args.resume == 'auto' and not args.resume_state:
        args.resume_state = str(latest)
    if args.init_from and not args.resume_state:
        init = torch.load(args.init_from, map_location=device, weights_only=False)
        model.load_state_dict(init['model_state_dict'])
        print(f"Initialised weights from {args.init_from} (fresh optimizer and schedule)")
    if args.resume_state:
        print(f"Resuming full training state: {args.resume_state}")
        # Load on the CPU: optimizer.load_state_dict moves exp_avg/exp_avg_sq to the params' device but deliberately leaves
        # AdamW's 'step' where it was loaded. With map_location=cuda the step counters ended up on the GPU, and AdamW then
        # read them with .item() every step (one sync per param tensor): GPU util 98% -> 78%, ~15% slower after every resume.
        state = torch.load(args.resume_state, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        start_epoch, skip_batches = state['epoch'], state['batch_in_epoch']
        global_step, best_val_loss = state['global_step'], state['best_val_loss']
        rng = state.get('rng')
        if rng:
            # map_location moved the saved ByteTensors to the device; the setters need them on the CPU
            torch.set_rng_state(rng['torch'].cpu())
            random.setstate(rng['python'])
            if rng['cuda'] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([t.cpu() for t in rng['cuda']])
        print(f"  epoch {start_epoch + 1}, batch {skip_batches}, step {global_step}, "
              f"lr={optimizer.param_groups[0]['lr']:.2e}, best val_loss={best_val_loss:.4f}, "
              f"RNG {'restored' if rng else 'not in file (older state)'}, saved {state.get('saved_at', '?')}")
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
        save_final=args.save_final,
        sleep=args.sleep,
        save_every_minutes=args.save_every_minutes,
        save_at_steps={int(x) for x in args.save_at_steps.split(',') if x},
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
