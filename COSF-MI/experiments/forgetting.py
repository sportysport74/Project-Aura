"""
Forgetting test: does replay sleep protect old knowledge while learning something new?

Phase A: train on TinyStories.  Phase B: switch to Shakespeare.
During phase B we track both val losses: story (what is being forgotten) and
Shakespeare (what is being learned). Every arm takes the same total number of
steps; replayed batches count toward it.

Arms:
  nosleep         phase B is all fresh Shakespeare
  replay_old      every --sleep-every B steps, sleep replays --replay random
                  STORY batches from phase A (old memories)
  replay_recent   same replay budget, but on random recent Shakespeare batches
                  (control: is it the old memories that matter, or just replay?)

Model: the 'recommended' setup (fixed phases + live heartbeat, target 0.3).

Usage: python experiments/forgetting.py --arm replay_old --seed 0
"""
import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "python"))
sys.path.insert(0, str(ROOT / "experiments"))
from ab_run import set_seed  # noqa: E402
from cosf_mi.models.cosf_oscillator import COSFOscillatorConfig, COSFOscillatorTransformer  # noqa: E402
from cosf_mi.data.tokenizer import BPETokenizer  # noqa: E402
from cosf_mi.data.token_dataset_fast import create_datasets_fast  # noqa: E402

SEQ = 256


@torch.no_grad()
def val_loss(model, inputs, targets, bs=32):
    model.eval()
    total = 0.0
    for i in range(0, len(inputs), bs):
        _, loss = model(inputs[i:i + bs], targets[i:i + bs], return_coherence_loss=False)
        total += loss.item() * len(inputs[i:i + bs])
    model.train()
    return total / len(inputs)


def shakespeare_chunks(tok, device):
    """Tokenize Shakespeare once (cached) and cut it into (input, target) windows."""
    cache = ROOT / "cache" / "shakespeare_tokens.pt"
    if cache.exists():
        ids = torch.load(cache)
    else:
        text = (ROOT / "data" / "shakespeare.txt").read_text(encoding="utf-8")
        ids = torch.tensor(tok.encode(text), dtype=torch.long)
        cache.parent.mkdir(exist_ok=True)
        torch.save(ids, cache)
    n = (len(ids) - 1) // SEQ
    x = ids[: n * SEQ].view(n, SEQ)
    y = ids[1: n * SEQ + 1].view(n, SEQ)
    split = int(n * 0.95)
    return (x[:split], y[:split]), (x[split:].to(device), y[split:].to(device))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=["nosleep", "replay_old", "replay_recent"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps-a", type=int, default=6000)
    p.add_argument("--steps-b", type=int, default=4000)
    p.add_argument("--sleep-every", type=int, default=500)
    p.add_argument("--replay", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--out", default="forgetting")
    args = p.parse_args()
    name = args.arm + (f"_seed{args.seed}" if args.seed else "")
    device = "cuda"
    bs = args.batch_size

    set_seed(args.seed)
    config = COSFOscillatorConfig(
        vocab_size=8000, d_model=256, n_layers=4, n_heads=4, max_seq_len=SEQ,
        homeostatic_decay_rate=0.01, enable_homeostasis=True, enable_refractory=False,
        enable_chirality=True, chirality_init_prob=0.5, enable_terahertz_scaling=False,
        heartbeat_mode="live", heartbeat_strength=0.1, heartbeat_target=0.3,
    )
    model = COSFOscillatorTransformer(config).to(device)

    with contextlib.redirect_stdout(io.StringIO()):
        tok = BPETokenizer(str(ROOT / "models/tokenizer_8k.model"))
        story_train, story_val_ds = create_datasets_fast(
            train_path=str(ROOT / "data/tinystories/styled_train.txt"),
            val_path=str(ROOT / "data/tinystories/styled_val.txt"),
            tokenizer=tok, seq_len=SEQ, cache_dir=str(ROOT / "cache/tokenized_styled"),
        )
    (sh_x, sh_y), (sh_val_x, sh_val_y) = shakespeare_chunks(tok, device)

    g = torch.Generator().manual_seed(1234)
    vidx = torch.randperm(len(story_val_ds), generator=g)[:512]
    vb = [story_val_ds[int(i)] for i in vidx]
    st_val_x = torch.stack([b[0] for b in vb]).to(device)
    st_val_y = torch.stack([b[1] for b in vb]).to(device)

    g = torch.Generator().manual_seed(args.seed)
    story_order = torch.randperm(len(story_train), generator=g)[:args.steps_a * bs]
    sh_order = torch.randint(0, len(sh_x), (args.steps_b * bs,), generator=g)
    replay_gen = torch.Generator().manual_seed(args.seed + 99)

    def story_batch(bi):
        b = [story_train[int(i)] for i in story_order[bi * bs:(bi + 1) * bs]]
        return torch.stack([s[0] for s in b]), torch.stack([s[1] for s in b])

    def sh_batch(bi):
        idx = sh_order[bi * bs:(bi + 1) * bs]
        return sh_x[idx], sh_y[idx]

    phase_params = [p for n, p in model.named_parameters() if n.endswith(".phases")]
    other = [p for n, p in model.named_parameters() if not n.endswith(".phases")]
    optimizer = torch.optim.AdamW([{"params": other}, {"params": phase_params, "weight_decay": 0.0}], lr=args.lr)
    total_steps = args.steps_a + args.steps_b
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

    def train_step(x, y):
        _, task, sync, _ = model(x.to(device), y.to(device), return_coherence_loss=True)
        (task + sync).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.apply_julia_correction()
        scheduler.step()
        optimizer.zero_grad()

    print(f"[{name}] start: steps {total_steps}", flush=True)
    t0 = time.time()
    history = []

    def log(step, phase):
        st, sh = val_loss(model, st_val_x, st_val_y), val_loss(model, sh_val_x, sh_val_y)
        history.append({"step": step, "phase": phase, "story_val": st, "shakespeare_val": sh})
        print(f"[{name}] step {step}: val {st:.4f} liveR 0.000 story {st:.4f} "
              f"shakespeare {sh:.4f} ({(time.time() - t0) / 60:.1f} min)", flush=True)

    # Phase A: stories
    model.train()
    for step in range(1, args.steps_a + 1):
        train_step(*story_batch(step - 1))
        if step % args.eval_every == 0:
            log(step, "A")

    # Phase B: Shakespeare, with sleeps every --sleep-every steps
    step, fresh, replayed = args.steps_a, 0, 0
    queue, recent = [], []
    while step < total_steps:
        if queue:
            kind, bi = queue.pop()
            x, y = story_batch(bi) if kind == "story" else sh_batch(bi)
            replayed += 1
        else:
            x, y = sh_batch(fresh)
            recent.append(fresh)
            fresh += 1
        train_step(x, y)
        step += 1
        if args.arm != "nosleep" and fresh and fresh % args.sleep_every == 0 and not queue and recent:
            pick = torch.randint(0, args.steps_a if args.arm == "replay_old" else len(recent),
                                 (args.replay,), generator=replay_gen)
            queue = ([("story", int(i)) for i in pick] if args.arm == "replay_old"
                     else [("sh", recent[int(i)]) for i in pick])
            recent = []
            print(f"[{name}] sleep @ {step}: replaying {args.replay} "
                  f"{'story' if args.arm == 'replay_old' else 'recent Shakespeare'} batches", flush=True)
        if (step - args.steps_a) % args.eval_every == 0 or step == total_steps:
            log(step, "B")

    after_a = next(h for h in history if h["step"] == args.steps_a)
    final = history[-1]
    result = {"arm": args.arm, "name": name, "args": vars(args), "history": history,
              "fresh_shakespeare_batches": fresh, "replayed_batches": replayed,
              "story_after_A": after_a["story_val"], "story_final": final["story_val"],
              "forgetting": final["story_val"] - after_a["story_val"],
              "shakespeare_final": final["shakespeare_val"]}
    out = ROOT / "experiments" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(result, indent=2))
    print(f"[{name}] done: final val {final['story_val']:.4f} | forgetting "
          f"{result['forgetting']:+.4f}, shakespeare {final['shakespeare_val']:.4f}, "
          f"replayed {replayed}, fresh {fresh}", flush=True)


if __name__ == "__main__":
    main()
