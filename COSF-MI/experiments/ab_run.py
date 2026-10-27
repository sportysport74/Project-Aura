"""
Short, seeded A/B training runs for the heartbeat, sleep and chirality ideas.

Every arm trains the same tiny model from scratch on the same data order
(same seed), so the only difference between arms is the feature under test.
Results go to experiments/results/<arm>.json.

Arms:
  baseline        static heartbeat (Level 1 as trained), no sleep
  live_001        live heartbeat, strength 0.01, push R -> 1
  live_01         live heartbeat, strength 0.1,  push R -> 1
  live_01_t03     live heartbeat, strength 0.1,  lift R only up to 0.3
  sleep_gentle    current gentle sleep (resets fatigue counters only)
  sleep_consol    consolidation: pull frequencies 50% toward their running average
  sleep_decay0    shrink frequencies 10% toward 0
  sleep_phi       old January sleep: pull frequencies 50% toward phi
  chir_nowd       chirality excluded from weight decay
  no_julia        Julia-bound damping (frequencies x 1/phi) switched off
  phase_nowd      phase offsets excluded from weight decay, no sleep
  phase_wrap      phase offsets wrapped to [0, 2pi) after every step, no sleep
  recommended     live heartbeat (0.1, target 0.3) + no weight decay on phases
  context_chir    recommended + per-token handedness (4 feature groups per layer)
  context_chir0   same, but the gate starts at 0 instead of +0.76
  rec_sleep_prune       recommended + sleep zeroes the weakest 10% of frequencies
  rec_sleep_shy         recommended + sleep scales all frequencies by 0.98
  rec_sleep_replay      recommended + sleep replays the 200 hardest batches of
                        the last wake period (counted in the step budget, so it
                        sees fewer fresh batches than the other arms)
  rec_sleep_replay_rand same, but 200 random batches of the last wake period

Usage:
  python experiments/ab_run.py --arm baseline --steps 10000
  python experiments/ab_run.py --arm baseline --seed 1   (noise floor)
"""
import argparse
import contextlib
import io
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "python"))
from cosf_mi.models.cosf_oscillator import (  # noqa: E402
    COSFOscillatorConfig, COSFOscillatorTransformer, kuramoto_coherence_loss, PHI,
)
from cosf_mi.data.tokenizer import BPETokenizer  # noqa: E402
from cosf_mi.data.token_dataset_fast import create_datasets_fast  # noqa: E402

ARMS = {
    "baseline":     dict(heartbeat="static"),
    "live_001":     dict(heartbeat="live", strength=0.01),
    "live_01":      dict(heartbeat="live", strength=0.1),
    "live_01_t03":  dict(heartbeat="live", strength=0.1, target=0.3),
    "sleep_gentle": dict(heartbeat="static", sleep="gentle"),
    "sleep_consol": dict(heartbeat="static", sleep="consol"),
    "sleep_decay0": dict(heartbeat="static", sleep="decay0"),
    "sleep_phi":    dict(heartbeat="static", sleep="phi"),
    "chir_nowd":    dict(heartbeat="static", chir_no_wd=True),
    "no_julia":     dict(heartbeat="static", no_julia=True),
    "phase_nowd":   dict(heartbeat="static", no_wd=(".phases",)),
    "phase_wrap":   dict(heartbeat="static", wrap_every_step=True),
    "recommended":  dict(heartbeat="live", strength=0.1, target=0.3, no_wd=(".phases",)),
    "context_chir": dict(heartbeat="live", strength=0.1, target=0.3, no_wd=(".phases",),
                         context_groups=4),
}
ARMS["context_chir0"] = dict(ARMS["context_chir"], context_bias=0.0)
# Round 4: sleeps that do work on their own, on top of 'recommended'
for _mode in ("prune", "shy", "replay", "replay_rand"):
    ARMS["rec_sleep_" + _mode] = dict(ARMS["recommended"], sleep=_mode)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, val_inputs, val_targets, batch_size=32):
    model.eval()
    total, n = 0.0, 0
    live_r = []
    neg, cnt = 0, 0
    for i in range(0, len(val_inputs), batch_size):
        x, y = val_inputs[i:i + batch_size], val_targets[i:i + batch_size]
        _, loss = model(x, y, return_coherence_loss=False)
        total += loss.item() * len(x)
        n += len(x)
        live_r.append(model.live_coherence().item())
        for o in model.live_oscillators():
            g = getattr(o, "last_context_chirality", None)
            if g is not None:
                g = g[..., ::o.out_features // o.context_groups]  # one value per group
                neg += (g < 0).sum().item()
                cnt += g.numel()
    model.train()
    evaluate.neg_fraction = neg / cnt if cnt else 0.0
    return total / n, sum(live_r) / len(live_r)


@torch.no_grad()
def sleep(model, mode, freq_ema):
    """Apply one sleep of the given kind to every oscillator's frequencies."""
    with contextlib.redirect_stdout(io.StringIO()):
        model.apply_homeostatic_reset(gentle=True)  # all arms keep the fatigue reset
    for i, osc in enumerate(model.live_oscillators()):
        w = osc.frequencies
        if mode == "consol":
            w.mul_(0.5).add_(freq_ema[i], alpha=0.5)
        elif mode == "decay0":
            w.mul_(0.9)
        elif mode == "phi":
            w.mul_(0.5).add_(PHI * 0.5)
        elif mode == "prune":
            k = max(1, int(0.1 * w.numel()))
            thresh = w.abs().flatten().kthvalue(k).values
            w[w.abs() <= thresh] = 0.0
        elif mode == "shy":
            w.mul_(0.98)
        # "replay" / "replay_rand" change no weights here; see the training loop


def chirality_vector(model):
    return torch.cat([o.chirality.detach().flatten().clone() for o in model.live_oscillators()])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=list(ARMS))
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--sleep-every", type=int, default=2000)
    p.add_argument("--replay", type=int, default=200, help="batches replayed per sleep")
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--val-samples", type=int, default=512)
    p.add_argument("--tag", default="")
    p.add_argument("--out", default="results", help="results subfolder of experiments/")
    p.add_argument("--save", action="store_true", help="save the final model next to the results")
    # Model size (defaults = the 'tiny' model train_overnight.py uses)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=3072)
    p.add_argument("--warmup", type=int, default=0, help="linear LR warmup steps")
    args = p.parse_args()
    arm = ARMS[args.arm]
    name = args.arm + (f"_seed{args.seed}" if args.seed else "") + args.tag
    device = "cuda"

    # Same model as train_overnight.py ('tiny')
    set_seed(args.seed)
    config = COSFOscillatorConfig(
        vocab_size=8000, d_model=args.d_model, n_layers=args.layers, n_heads=args.heads,
        d_ff=args.d_ff, max_seq_len=256,
        homeostatic_decay_rate=0.01, enable_homeostasis=True, enable_refractory=False,
        enable_chirality=True, chirality_init_prob=0.5, enable_terahertz_scaling=False,
        heartbeat_mode=arm["heartbeat"],
        heartbeat_strength=arm.get("strength", 0.01),
        heartbeat_target=arm.get("target"),
        apply_phase_correction=not arm.get("no_julia", False),
        chirality_context_groups=arm.get("context_groups", 0),
        chirality_context_bias=arm.get("context_bias", 1.0),
    )
    model = COSFOscillatorTransformer(config).to(device)

    with contextlib.redirect_stdout(io.StringIO()):
        tok = BPETokenizer(str(ROOT / "models/tokenizer_8k.model"))
        train_ds, val_ds = create_datasets_fast(
            train_path=str(ROOT / "data/tinystories/styled_train.txt"),
            val_path=str(ROOT / "data/tinystories/styled_val.txt"),
            tokenizer=tok, seq_len=256, cache_dir=str(ROOT / "cache/tokenized_styled"),
        )

    # Fixed validation subset, identical for every arm
    g = torch.Generator().manual_seed(1234)
    vidx = torch.randperm(len(val_ds), generator=g)[:args.val_samples]
    vb = [val_ds[int(i)] for i in vidx]
    val_inputs = torch.stack([b[0] for b in vb]).to(device)
    val_targets = torch.stack([b[1] for b in vb]).to(device)

    # Same data order for every arm with the same seed
    g = torch.Generator().manual_seed(args.seed)
    # (a fresh shuffle per pass if the run needs more than one pass over the data)
    need = args.steps * args.batch_size
    passes = [torch.randperm(len(train_ds), generator=g) for _ in range(-(-need // len(train_ds)))]
    order = torch.cat(passes)[:need]

    no_wd = arm.get("no_wd", (".chirality",) if arm.get("chir_no_wd") else ())
    if no_wd:
        skip = [p for n, p in model.named_parameters() if n.endswith(no_wd)]
        rest = [p for n, p in model.named_parameters() if not n.endswith(no_wd)]
        groups = [{"params": rest}, {"params": skip, "weight_decay": 0.0}]
    else:
        groups = model.parameters()
    optimizer = torch.optim.AdamW(groups, lr=args.lr)  # default weight_decay=0.01, as in training
    if args.warmup:
        warm = max(1, args.warmup)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (
            1 + math.cos(math.pi * min(1.0, s / args.steps))))
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)

    freq_ema = [o.frequencies.detach().clone() for o in model.live_oscillators()]
    chir0 = chirality_vector(model)
    history, sleeps = [], []
    julia_total = 0
    sleep_mode = arm.get("sleep")
    print(f"[{name}] start: steps {args.steps}", flush=True)
    t0 = time.time()
    model.train()
    fresh = 0            # next unseen batch in `order`
    replay_queue = []    # batch numbers to replay before taking fresh data again
    wake_losses = []     # (loss, batch number) for fresh batches since the last sleep

    for step in range(1, args.steps + 1):
        replaying = bool(replay_queue)
        if replaying:
            bi = replay_queue.pop()
        else:
            bi, fresh = fresh, fresh + 1
        idx = order[bi * args.batch_size:(bi + 1) * args.batch_size]
        b = [train_ds[int(i)] for i in idx]
        x = torch.stack([s[0] for s in b]).to(device)
        y = torch.stack([s[1] for s in b]).to(device)

        _, task_loss, sync_loss, coh = model(x, y, return_coherence_loss=True)
        (task_loss + sync_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        before_jc = model.phase_corrections.item()
        model.apply_julia_correction()
        julia_total += model.phase_corrections.item() - before_jc
        if arm.get("wrap_every_step"):
            with torch.no_grad():
                for o in model.live_oscillators():
                    o.phases.remainder_(2 * math.pi)
        scheduler.step()
        optimizer.zero_grad()
        if not replaying:
            wake_losses.append((task_loss.item(), bi))

        with torch.no_grad():
            for e, o in zip(freq_ema, model.live_oscillators()):
                e.mul_(0.999).add_(o.frequencies, alpha=0.001)

        if sleep_mode and step % args.sleep_every == 0 and step < args.steps:
            before, _ = evaluate(model, val_inputs, val_targets)
            sleep(model, sleep_mode, freq_ema)
            if sleep_mode == "replay":
                replay_queue = [b for _, b in sorted(wake_losses, reverse=True)[:args.replay]]
            elif sleep_mode == "replay_rand":
                pick = torch.randperm(len(wake_losses), generator=torch.Generator().manual_seed(step))
                replay_queue = [wake_losses[int(i)][1] for i in pick[:args.replay]]
            wake_losses = []
            after, _ = evaluate(model, val_inputs, val_targets)
            sleeps.append({"step": step, "val_before": before, "val_after": after})
            print(f"[{name}] sleep @ {step}: val {before:.4f} -> {after:.4f}", flush=True)

        if step % args.eval_every == 0 or step == args.steps:
            val_loss, live_r = evaluate(model, val_inputs, val_targets)
            static_r = kuramoto_coherence_loss(
                torch.cat([o.phases for o in model.live_oscillators()]), 0.0)[1]
            c = chirality_vector(model)
            row = {
                "step": step, "train_loss": task_loss.item(), "val_loss": val_loss,
                "live_R": live_r, "static_R": static_r, "heartbeat_coh": float(coh),
                "chir_abs_mean": c.abs().mean().item(),
                "chir_flipped": (torch.sign(c) != torch.sign(chir0)).sum().item(),
                "julia_corrections": julia_total,
                "context_flipped_frac": evaluate.neg_fraction,
                "elapsed_min": (time.time() - t0) / 60,
            }
            history.append(row)
            print(f"[{name}] step {step}: val {val_loss:.4f} liveR {live_r:.3f} "
                  f"staticR {static_r:.3f} |c| {row['chir_abs_mean']:.3f} "
                  f"flipped {row['chir_flipped']} ctx-flip {evaluate.neg_fraction:.3f} julia {julia_total} ({row['elapsed_min']:.1f} min)", flush=True)

    out = ROOT / "experiments" / args.out
    out.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] fresh batches seen: {fresh} of {args.steps} steps")
    result = {"arm": args.arm, "name": name, "settings": arm, "args": vars(args), "fresh_batches": fresh,
              "final": history[-1], "history": history, "sleeps": sleeps}
    (out / f"{name}.json").write_text(json.dumps(result, indent=2))
    if args.save:
        torch.save({"config": config, "model_state_dict": model.state_dict(),
                    "epoch": 0, "val_loss": history[-1]["val_loss"]}, out / f"{name}.pt")
    print(f"[{name}] done: final val {history[-1]['val_loss']:.4f}")


if __name__ == "__main__":
    main()
