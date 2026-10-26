"""
Probe a trained checkpoint: live heartbeat R, static R, and chirality values.

Usage: python experiments/probe_checkpoint.py checkpoints/fixed_run_100ep/best_model.pt
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "python"))
from cosf_mi.models.cosf_oscillator import COSFOscillatorTransformer, kuramoto_coherence_loss
from cosf_mi.data.token_dataset_fast import create_datasets_fast
from cosf_mi.data.tokenizer import BPETokenizer

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/fixed_run_100ep/best_model.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model = COSFOscillatorTransformer(ckpt["config"]).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"checkpoint: {ckpt_path} (epoch {ckpt['epoch'] + 1}, val loss {ckpt['val_loss']:.4f})")

tok = BPETokenizer("models/tokenizer_8k.model")
_, val = create_datasets_fast(
    train_path="data/tinystories/styled_train.txt",
    val_path="data/tinystories/styled_val.txt",
    tokenizer=tok, seq_len=256, cache_dir="cache/tokenized_styled",
)
torch.manual_seed(0)
idx = torch.randperm(len(val))[:64]
batch = [val[int(i)] for i in idx]
inputs = torch.stack([b["input"] if isinstance(b, dict) else b[0] for b in batch]).to(device)

with torch.no_grad():
    model(inputs)
    live_r = model.live_coherence().item()
    # per-oscillator live R, and the spread of R across tokens
    per_layer = []
    for osc in model.live_oscillators():
        p = osc.last_total_phase
        r = torch.sqrt(torch.cos(p).mean(-1) ** 2 + torch.sin(p).mean(-1) ** 2)
        per_layer.append((r.mean().item(), r.std().item()))
    static = torch.cat([o.phases for o in model.live_oscillators()])
    _, static_r = kuramoto_coherence_loss(static, drift=0.0)

print(f"static R (phase-offset params, Level 1 'coh'): {static_r:.4f}")
print(f"live R (per-token, all oscillators):           {live_r:.4f}")
names = ["q", "k", "v", "out", "ff1", "ff2"]
for i, (m, s) in enumerate(per_layer):
    print(f"  block {i // 6} {names[i % 6]:>3}: live R {m:.4f} (token std {s:.4f})")

# A random-phase baseline: R for N uniform phases is about 1/sqrt(N)
n = sum(o.last_total_phase.shape[-1] for o in model.live_oscillators())
print(f"random-phase baseline for {n} features: ~{(1 / n) ** 0.5:.4f}")

chir = torch.cat([o.chirality.flatten() for o in model.live_oscillators()])
print(f"chirality: n={chir.numel()} requires_grad={next(model.live_oscillators()).chirality.requires_grad}")
print(f"  min {chir.min():.4f} max {chir.max():.4f} |c| mean {chir.abs().mean():.4f}")
print(f"  right {(chir > 0).sum().item()} left {(chir < 0).sum().item()}  |c|<0.5: {(chir.abs() < 0.5).sum().item()}")
