"""
Chat fine-tune: teach a story-trained AURA model to hold a conversation.

Data: TinyChat (data/tinychat/tinychat_80mb.txt, one conversation per line,
"[INST] user [/INST] reply ..."), reformatted as

    You: <user turn> Aura: <reply> You: <user turn> Aura: <reply> ... You:

Loss is only counted on Aura's replies plus the "You:" that follows each
reply (so the model learns when its turn is over). The user's words are
context only.

Replay sleep protects storytelling: a memory of story batches is replayed
every --sleep-every steps (see cosf_mi/training/replay.py and the Oct 9
forgetting test, where replay cut forgetting by ~76%).

Usage:
  python finetune_chat.py --checkpoint checkpoints/heartbeat_fixed_10ep/best_model.pt
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src" / "python"))
from cosf_mi.models.cosf_oscillator import COSFOscillatorTransformer  # noqa: E402
from cosf_mi.data.tokenizer import BPETokenizer  # noqa: E402
from cosf_mi.data.token_dataset_fast import create_datasets_fast  # noqa: E402
from cosf_mi.training.replay import ReplayMemory, replay_sleep  # noqa: E402

SEQ = 256
USER, AURA = "You:", "Aura:"
IGNORE = -100


def parse_conversation(line):
    """'[INST] a [/INST] b [INST] c [/INST] d' -> [('a', 'b'), ('c', 'd')]"""
    turns = []
    for chunk in line.split("[INST]")[1:]:
        if "[/INST]" not in chunk:
            continue
        user, reply = chunk.split("[/INST]", 1)
        user, reply = user.strip(), reply.strip()
        if "INST]" in reply or "INST]" in user:  # ~0.3% of TinyChat lost their [INST] tags
            break                                 # keep only the clean turns before it
        if user and reply:
            turns.append((user, reply))
    return turns


def encode_chat(tok, turns, train_on_replies=True):
    """Token ids and per-token loss mask for a conversation (pieces encoded separately,
    exactly as chat_with_aura.py builds the prompt at chat time)."""
    user_ids, aura_ids = tok.encode(USER), tok.encode(AURA)
    ids, mask = [], []
    for i, (user, reply) in enumerate(turns):
        marker_mask = 1 if (i > 0 and train_on_replies) else 0  # "You:" after a reply ends Aura's turn
        ids += user_ids
        mask += [marker_mask] * len(user_ids)
        u = tok.encode(user)
        ids += u + aura_ids
        mask += [0] * (len(u) + len(aura_ids))
        r = tok.encode(reply)
        ids += r
        mask += [1 if train_on_replies else 0] * len(r)
    ids += user_ids
    mask += [1] * len(user_ids)
    return ids, mask


def build_chat_tensors(tok, lines):
    """Inputs [N, SEQ] and targets [N, SEQ] with non-reply positions set to IGNORE."""
    xs, ys = [], []
    for line in lines:
        turns = parse_conversation(line)
        if not turns:
            continue
        ids, mask = encode_chat(tok, turns)
        ids, mask = ids[:SEQ + 1], mask[:SEQ + 1]
        if len(ids) < 8:
            continue
        x = ids[:-1]
        y = [t if m else IGNORE for t, m in zip(ids[1:], mask[1:])]
        pad = SEQ - len(x)
        xs.append(x + [0] * pad)
        ys.append(y + [IGNORE] * pad)
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def count_truncated(tok, lines):
    """Conversations longer than the SEQ window (their tail gets cut off in training)."""
    return sum(len(encode_chat(tok, t)[0]) > SEQ + 1 for t in map(parse_conversation, lines) if t)


def load_distill(path, share, n_chat_train, max_repeats, seed):
    """Verified teacher data (scripts/generate_cosf_distill.py) -> (train lines, val lines, chat keep).
    Distill lines are repeated up to max_repeats times to reach `share` of the training mix;
    if that's still short, TinyChat is cut to `chat keep` lines so the share holds."""
    lines = []
    for l in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            lines.append(json.loads(l)["line"])
        except (json.JSONDecodeError, KeyError):
            pass  # a line cut off by a shutdown mid-write
    random.Random(seed).shuffle(lines)
    n_val = max(50, len(lines) // 20)
    val, train = lines[:n_val], lines[n_val:]
    want = share / (1 - share) * n_chat_train
    repeats = max(1, min(max_repeats, math.ceil(want / max(1, len(train)))))
    train = train * repeats
    chat_keep = min(n_chat_train, int(len(train) * (1 - share) / share))
    return train, val, chat_keep, repeats


@torch.no_grad()
def val_loss(model, x, y, device, bs=32):
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(x), bs):
        xb, yb = x[i:i + bs].to(device), y[i:i + bs].to(device)
        _, loss = model(xb, yb, return_coherence_loss=False)
        k = (yb != IGNORE).sum().item()
        total += loss.item() * k
        n += k
    model.train()
    return total / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", default="data/tinychat/tinychat_80mb.txt")
    p.add_argument("--distill", default=None,
                   help="verified teacher data (.jsonl) to blend in, e.g. data/cosf_distill_verified.jsonl")
    p.add_argument("--distill-share", type=float, default=0.3, help="fraction of training conversations from --distill")
    p.add_argument("--distill-max-repeats", type=int, default=4, help="max times each distill sample is reused")
    p.add_argument("--conversations", type=int, default=100000)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--sleep-every", type=int, default=300, help="0 = no replay sleep")
    p.add_argument("--replay", type=int, default=30, help="story batches replayed per sleep")
    p.add_argument("--memory", type=int, default=2000, help="story batches kept in replay memory")
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--out", default="checkpoints/aura_chat")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = COSFOscillatorTransformer(ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Base model: {args.checkpoint} (story val loss {ckpt['val_loss']:.4f})")

    # the base model's vocab says which tokenizer/pretraining data it came from
    mix = ckpt["config"].vocab_size == 16000
    tok = BPETokenizer(str(ROOT / ("models/tokenizer_mix16k.model" if mix else "models/tokenizer_8k.model")))
    base_data = ROOT / ("data/mix" if mix else "data/tinystories")
    lines = Path(args.data).read_text(encoding="utf-8").splitlines()[:-1]  # last line may be cut off
    random.Random(1234).shuffle(lines)
    lines = lines[:args.conversations]
    n_val = max(500, len(lines) // 50)
    print(f"Tokenizing {len(lines):,} conversations...")
    chat_train = lines[n_val:]
    dval_x = None
    if args.distill:
        d_train, d_val, keep, repeats = load_distill(args.distill, args.distill_share, len(chat_train),
                                                     args.distill_max_repeats, args.seed)
        chat_train = chat_train[:keep]
        share = len(d_train) / (len(d_train) + len(chat_train))
        print(f"  distill: {len(d_train) // repeats:,} samples x{repeats} + {len(chat_train):,} TinyChat "
              f"= {100 * share:.0f}% distill; {len(d_val)} held out; "
              f"{count_truncated(tok, d_train[:len(d_train) // repeats])} longer than {SEQ} tokens")
        dval_x, dval_y = build_chat_tensors(tok, d_val)
        chat_train = chat_train + d_train  # shuffled by the training order below
    val_x, val_y = build_chat_tensors(tok, lines[:n_val])
    train_x, train_y = build_chat_tensors(tok, chat_train)
    print(f"  train {len(train_x):,}  val {len(val_x):,}  "
          f"(reply tokens per conversation: {(train_y != IGNORE).sum().item() / len(train_x):.0f})")

    # Story data: replay memory and a story val set to measure forgetting
    with contextlib.redirect_stdout(io.StringIO()):
        story_train, story_val = create_datasets_fast(
            train_path=str(base_data / "styled_train.txt"),
            val_path=str(base_data / "styled_val.txt"),
            tokenizer=tok, seq_len=SEQ, cache_dir=str(ROOT / "cache/tokenized_styled"),
        )
    g = torch.Generator().manual_seed(1234)
    vidx = torch.randperm(len(story_val), generator=g)[:512]
    sv = [story_val[int(i)] for i in vidx]
    story_vx, story_vy = torch.stack([s[0] for s in sv]), torch.stack([s[1] for s in sv])
    memory = ReplayMemory(capacity=args.memory, seed=args.seed)
    if args.sleep_every:
        pick = torch.randperm(len(story_train), generator=g)[:args.memory * args.batch_size]
        for i in range(args.memory):
            b = [story_train[int(j)] for j in pick[i * args.batch_size:(i + 1) * args.batch_size]]
            memory.add(torch.stack([s[0] for s in b]), torch.stack([s[1] for s in b]))
        print(f"  replay memory: {len(memory)} story batches")

    phase_params = [p for n, p in model.named_parameters() if n.endswith(".phases")]
    other = [p for n, p in model.named_parameters() if not n.endswith(".phases")]
    optimizer = torch.optim.AdamW([{"params": other}, {"params": phase_params, "weight_decay": 0.0}], lr=args.lr)
    steps = int(len(train_x) / args.batch_size * args.epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))

    story_before = val_loss(model, story_vx, story_vy, device)
    chat_before = val_loss(model, val_x, val_y, device)
    distill_before = val_loss(model, dval_x, dval_y, device) if dval_x is not None else None
    print(f"Before: chat val {chat_before:.4f}  story val {story_before:.4f}"
          + (f"  distill val {distill_before:.4f}" if dval_x is not None else ""))
    print(f"Training {steps:,} steps (batch {args.batch_size}, lr {args.lr}, "
          f"replay {'every ' + str(args.sleep_every) if args.sleep_every else 'off'})")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    t0 = time.time()
    order = torch.randperm(len(train_x))
    model.train()
    for step in range(1, steps + 1):
        i = ((step - 1) * args.batch_size) % len(train_x)
        idx = order[i:i + args.batch_size]
        x, y = train_x[idx].to(device), train_y[idx].to(device)
        _, loss, sync, coh = model(x, y, return_coherence_loss=True)
        (loss + sync).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.apply_julia_correction()
        scheduler.step()
        optimizer.zero_grad()

        if args.sleep_every and step % args.sleep_every == 0 and step < steps:
            dream = replay_sleep(model, optimizer, memory, args.replay, device)
            print(f"  sleep @ {step}: replayed {args.replay} story batches (dream loss {dream:.3f})", flush=True)

        if step % args.eval_every == 0 or step == steps:
            chat = val_loss(model, val_x, val_y, device)
            story = val_loss(model, story_vx, story_vy, device)
            distill = val_loss(model, dval_x, dval_y, device) if dval_x is not None else None
            mins = (time.time() - t0) / 60
            rate = step / max(1.0, time.time() - t0)
            left = (steps - step) / rate / 60
            # tqdm-style line so fish.bat / tank.bat can follow the fine-tune
            print(f"Epoch 1/1: {100 * step // steps}%|| {step}/{steps} [{mins:.0f}m<{left:.0f}m, "
                  f"{rate:.2f}it/s, loss={loss.item():.4f}]", flush=True)
            print(f"step {step}: chat val {chat:.4f}  story val {story:.4f} "
                  f"(story change {story - story_before:+.4f})  liveR {coh:.3f}  {mins:.1f} min"
                  + (f"  distill val {distill:.4f}" if distill is not None else ""), flush=True)
            print(f"  Val loss: {chat:.4f}", flush=True)
            # with distill data, keep the model that's best on both kinds of chat
            score = chat if distill is None else (chat + distill) / 2
            if score < best:
                best = score
                torch.save({"config": ckpt["config"], "model_state_dict": model.state_dict(),
                            "epoch": 0, "val_loss": chat, "story_val_loss": story, "distill_val_loss": distill,
                            "chat_format": {"user": USER, "aura": AURA},
                            "base_checkpoint": args.checkpoint}, out / "best_model.pt")

    print(f"Done. Best {'(chat+distill)/2' if dval_x is not None else 'chat'} val {best:.4f} "
          f"(chat was {chat_before:.4f}"
          + (f", distill was {distill_before:.4f}" if dval_x is not None else "") + "); "
          f"story val {story:.4f} (was {story_before:.4f}). Saved to {out / 'best_model.pt'}")


if __name__ == "__main__":
    main()
