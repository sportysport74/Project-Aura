"""
Heartbeat probe for sweep checkpoints: true R, R_ψ (the 'live_control' target) and the drift gate on val batches.

The training log only shows the sync loss, which is 0 both when R_ψ reaches the 0.3 target and when the drift gate is
closed, so this measures R_ψ itself. R_ψ uses the same fixed offsets as the model (seed 271828), so it can be measured
on 'on' checkpoints too. Writes experiments/results/sweep/hb_probe.tsv.

Usage (from COSF-MI/): .venv/bin/python experiments/hb_probe.py l100_control_s1 l100_on_s1 ... [--batches 20]
"""
import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "python"))
from cosf_mi.data.memmap_dataset import load_corpus  # noqa: E402
from cosf_mi.models.cosf_oscillator import COSFOscillatorTransformer  # noqa: E402

OUT = ROOT / "experiments/results/sweep/hb_probe.tsv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run names under checkpoints/sweep/")
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--corpus", default="data/corpus_v1/tok24k_v2")
    args = ap.parse_args()
    _, val, _ = load_corpus(ROOT / args.corpus, 512)
    loader = torch.utils.data.DataLoader(val, batch_size=32, shuffle=False)
    rows = ["run\tval_loss\tR_true\tR_psi\tgate\tdrift"]
    for run in args.runs:
        ckpt = torch.load(ROOT / "checkpoints/sweep" / run / "final_model.pt", map_location="cpu", weights_only=False)
        model = COSFOscillatorTransformer(ckpt["config"]).cuda()
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        c = model.config
        n_features = c.n_layers * (5 * c.d_model + c.d_ff)
        psi = (torch.rand(n_features, generator=torch.Generator().manual_seed(271828)) * (2 * math.pi)).cuda()
        first = model.blocks[0].attn.q_oscillator
        drift = float((first.frequencies - first.rest_frequency).abs().mean().detach())
        loss = r_true = r_psi = 0.0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i, batch in enumerate(loader):
                if i >= args.batches:
                    break
                inputs, targets = (t.cuda() for t in (batch if not isinstance(batch, dict) else
                                                      (batch["input"], batch["target"])))
                _, task_loss = model(inputs, targets, return_coherence_loss=False)
                loss += float(task_loss)
                r_true += float(model.live_coherence())
                r_psi += float(model.live_coherence(psi))
        n = min(args.batches, len(loader))
        rows.append(f"{run}\t{loss / n:.4f}\t{r_true / n:.4f}\t{r_psi / n:.4f}\t{max(0.0, min(1.0, 1 - drift)):.4f}\t{drift:.4f}")
        print(rows[-1], flush=True)
        del model
        torch.cuda.empty_cache()
    OUT.write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
