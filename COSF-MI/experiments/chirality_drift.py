"""
Where did the learnable chirality χ go during training? Compares each oscillator's trained χ with its ±1 init.

The init is rebuilt exactly: train_overnight.py calls torch.manual_seed(seed) right before building the model on the CPU, so
rebuilding COSFOscillatorTransformer(config) after the same seed reproduces every initial χ. Check: the fraction of χ whose trained
sign matches the rebuilt init. A wrong rebuild would give ~50% (random signs); --check CKPT compares against a second run with the
same seed (its init is the same).

Per layer and oscillator type (q/k/v/out/ff1/ff2): sign kept / flipped, and |χ| bands: < 0.5 (shrank toward 0), 0.5-0.9 (shrank),
0.9-1.1 (about the same), > 1.1 (grew). Also the per-token gate (context_gate) if the config has it on. CPU only.

Usage (from COSF-MI/): CUDA_VISIBLE_DEVICES= .venv/bin/python experiments/chirality_drift.py CKPT --seed 1 --out experiments/results/chirality/NAME.md
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "python"))
from cosf_mi.models.cosf_oscillator import COSFOscillatorTransformer  # noqa: E402

TYPES = {"attn.q_oscillator": "q", "attn.k_oscillator": "k", "attn.v_oscillator": "v", "attn.out_oscillator": "out",
         "ff1": "ff1", "ff2": "ff2"}
BANDS = [("<0.5", 0, 0.5), ("0.5-0.9", 0.5, 0.9), ("0.9-1.1", 0.9, 1.1), (">1.1", 1.1, float("inf"))]


def chis(state):
    """{(layer, type): tensor} for every oscillator's chirality."""
    out = {}
    for k, v in state.items():
        if k.endswith(".chirality") and k.startswith("blocks."):
            layer = int(k.split(".")[1])
            osc = ".".join(k.split(".")[2:-1])
            out[(layer, TYPES[osc])] = v.float()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt")
    p.add_argument("--seed", type=int, required=True, help="the run's --seed (rebuilds the init)")
    p.add_argument("--check", default=None, help="another checkpoint trained from the same seed, for the sign-agreement check")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    ck = torch.load(ROOT / args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    torch.manual_seed(args.seed)
    init = chis(COSFOscillatorTransformer(cfg).state_dict())
    trained = chis(ck["model_state_dict"])
    assert init.keys() == trained.keys() and init, "no chirality found"
    assert all(set(v.unique().tolist()) <= {-1.0, 1.0} for v in init.values()), "rebuilt init isn't ±1"

    allx = torch.cat([trained[k] for k in sorted(trained)])
    alli = torch.cat([init[k] for k in sorted(init)])
    kept_all = (torch.sign(allx) == alli).float().mean().item()
    L = [f"# Chirality drift: `{args.ckpt}`\n",
         f"Init rebuilt from seed {args.seed} (exact ±1). {allx.numel():,} oscillators in {len(trained)} layers × types. "
         f"Context gate: {'ON, groups ' + str(cfg.chirality_context_groups) if getattr(cfg, 'chirality_context_groups', 0) else 'off (chirality_context_groups = 0)'}.\n",
         f"**Sign kept: {kept_all:.2%}** (a wrong init rebuild would give ~50%).\n"]
    if args.check:
        other = chis(torch.load(ROOT / args.check, map_location="cpu", weights_only=False)["model_state_dict"])
        ox = torch.cat([other[k] for k in sorted(other)])
        L.append(f"Check run `{args.check}` (same seed): sign kept vs this rebuilt init {(torch.sign(ox) == alli).float().mean():.2%}, "
                 f"sign agreement with this run {(torch.sign(ox) == torch.sign(allx)).float().mean():.2%}.\n")
    a = allx.abs()
    L.append(f"Overall |χ|: mean {a.mean():.3f}, median {a.median():.3f}, min {a.min():.3f}, max {a.max():.3f}, "
             f"p5 {a.quantile(0.05):.3f}, p95 {a.quantile(0.95):.3f}; signed mean {allx.mean():+.4f} "
             f"(init {alli.mean():+.4f}).\n")
    agg = defaultdict(list)
    for (layer, t), x in trained.items():
        agg[t].append((x, init[(layer, t)]))

    def row(name, xs, inits):
        x, i = torch.cat(xs), torch.cat(inits)
        kept = (torch.sign(x) == i)
        ax = x.abs()
        bands = " | ".join(f"{((ax >= lo) & (ax < hi)).float().mean():.1%}" for _, lo, hi in BANDS)
        return (f"| {name} | {x.numel():,} | {kept.float().mean():.1%} | {(~kept).sum().item():,} | {ax.mean():.3f} | "
                f"{ax.min():.3f} / {ax.max():.3f} | {bands} |")
    head = ("| {} | n | sign kept | flipped | mean |χ| | min / max |χ| | " + " | ".join(f"|χ| {b}" for b, _, _ in BANDS)
            + " |\n|---|---|---|---|---|---|" + "---|" * len(BANDS))
    L.append("\n## By oscillator type (all layers)\n\n" + head.format("type"))
    for t in TYPES.values():
        L.append(row(t, [x for x, _ in agg[t]], [i for _, i in agg[t]]))
    L.append("\n## By layer (all types)\n\n" + head.format("layer"))
    for layer in sorted({k[0] for k in trained}):
        ks = [k for k in trained if k[0] == layer]
        L.append(row(str(layer), [trained[k] for k in ks], [init[k] for k in ks]))
    L.append("\n## Every layer × type: mean |χ| (sign kept %)\n\n| layer | " + " | ".join(TYPES.values()) + " |\n|---|"
             + "---|" * len(TYPES))
    for layer in sorted({k[0] for k in trained}):
        cells = []
        for t in TYPES.values():
            x, i = trained[(layer, t)], init[(layer, t)]
            cells.append(f"{x.abs().mean():.2f} ({(torch.sign(x) == i).float().mean():.0%})")
        L.append(f"| {layer} | " + " | ".join(cells) + " |")
    gate = {k: v for k, v in ck["model_state_dict"].items() if "context_gate" in k}
    if gate:
        L.append("\n## Per-token gate (context_gate)\n")
        for k, v in sorted(gate.items()):
            L.append(f"- `{k}`: shape {tuple(v.shape)}, mean {v.float().mean():+.4f}, |max| {v.float().abs().max():.4f}")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
