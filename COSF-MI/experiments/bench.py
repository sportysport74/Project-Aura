"""
Throughput benchmark: tokens/second and memory for model sizes, batch sizes
and math precision, on random tokens (training step: forward + backward + AdamW).

Usage: python experiments/bench.py
"""
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "python"))
from cosf_mi.models.cosf_oscillator import COSFOscillatorConfig, COSFOscillatorTransformer  # noqa: E402

SIZES = {
    "tiny   256x4": dict(d_model=256, n_layers=4, n_heads=4),
    "mid    512x6": dict(d_model=512, n_layers=6, n_heads=8),
    "big    512x8": dict(d_model=512, n_layers=8, n_heads=8),
    "large  768x8": dict(d_model=768, n_layers=8, n_heads=12),
}


def bench(size, batch, precision, steps=12):
    torch.backends.cuda.matmul.allow_tf32 = precision != "fp32"
    torch.backends.cudnn.allow_tf32 = precision != "fp32"
    cfg = COSFOscillatorConfig(vocab_size=8000, max_seq_len=256, enable_refractory=False,
                               enable_chirality=True, heartbeat_mode="live",
                               heartbeat_strength=0.1, heartbeat_target=0.3,
                               d_ff=4 * SIZES[size]["d_model"], **SIZES[size])
    model = COSFOscillatorTransformer(cfg).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randint(0, 8000, (batch, 256), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    for i in range(steps + 3):
        if i == 3:
            torch.cuda.synchronize()
            t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
            _, loss, sync, _ = model(x, x, return_coherence_loss=True)
        (loss + sync).backward()
        opt.step()
        opt.zero_grad()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / steps
    params = sum(p.numel() for p in model.parameters()) / 1e6
    mem = torch.cuda.max_memory_allocated() / 1e9
    del model, opt
    torch.cuda.empty_cache()
    return params, batch * 256 / dt, 1 / dt, mem


print(f"{'size':<14} {'params':>7} {'batch':>5} {'prec':>5} {'tok/s':>9} {'step/s':>7} {'mem GB':>7}")
for size in SIZES:
    for batch in (8, 32, 64):
        for prec in ("fp32", "tf32", "bf16"):
            if size != "tiny   256x4" and prec == "fp32":
                continue
            try:
                p, tps, sps, mem = bench(size, batch, prec)
                print(f"{size:<14} {p:>6.1f}M {batch:>5} {prec:>5} {tps:>9.0f} {sps:>7.1f} {mem:>7.1f}", flush=True)
            except torch.cuda.OutOfMemoryError:
                print(f"{size:<14} {'':>7} {batch:>5} {prec:>5}  out of memory", flush=True)
                torch.cuda.empty_cache()
