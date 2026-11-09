"""Settling diagnostics for the settling-retest B runs (plans/settling_retest.md, "Also reported").

For each checkpoint, runs the whole arithmetic val split (data/arith_settle/tok24k/val.bin, 512-token sequences) and captures each
settling module's input. Per layer and settling step t = 0..T (t = 0 is φ⁰, before any step), over every val token:
  R_t      Kuramoto order |mean_i exp(iφ_i)| per token, averaged over tokens
  var(R)_t variance of the per-token R across tokens
  λ_t      finite-difference Lyapunov estimate ln(‖δ_t‖/‖δ_0‖)/t from a 1e-4 perturbation of φ⁰ (same as SettlingModule.diagnose)
  |Δφ|_t   mean |φ_t − φ⁰| (how far the phases actually moved)
Usage: settle_diag.py OUT.md NAME=path/to/final_model.pt ...
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "python"))
from cosf_mi.models.cosf_oscillator import COSFOscillatorTransformer  # noqa: E402

SEQ, BS, EPS = 512, 16, 1e-4


@torch.no_grad()
def diag(model, val):
    mods = [b.settle for b in model.blocks]
    T = mods[0].T
    acc = [dict(R=np.zeros(T + 1), R2=np.zeros(T + 1), lam=np.zeros(T + 1), dphi=np.zeros(T + 1), n=0) for _ in mods]
    caught = {}
    hooks = [m.register_forward_pre_hook(lambda mod, inp, i=i: caught.__setitem__(i, inp[0].detach())) for i, m in enumerate(mods)]
    g = torch.Generator(device="cuda").manual_seed(0)
    n_seq = len(val) // SEQ
    for s in range(0, n_seq, BS):
        x = torch.from_numpy(val[s * SEQ:min(s + BS, n_seq) * SEQ].astype(np.int64)).view(-1, SEQ).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(x)
        for i, m in enumerate(mods):
            phi0 = m.w_in(m.ln(caught[i].float())).reshape(-1, m.n)
            d0 = torch.randn(phi0.shape, device="cuda", generator=g)
            d0 = EPS * d0 / d0.norm(dim=-1, keepdim=True)
            a, b = phi0, phi0 + d0
            a_ = acc[i]
            a_["n"] += phi0.shape[0]
            for t in range(T + 1):
                if t:
                    a, _ = m.step(a, phi0, m.K)
                    b, _ = m.step(b, phi0 + d0, m.K)
                    a_["lam"][t] += (torch.log(((b - a).norm(dim=-1) / EPS).clamp_min(1e-12)) / t).sum().item()
                r = torch.complex(torch.cos(a), torch.sin(a)).mean(-1).abs()
                a_["R"][t] += r.sum().item()
                a_["R2"][t] += (r * r).sum().item()
                a_["dphi"][t] += (a - phi0).abs().mean(-1).sum().item()
    for h in hooks:
        h.remove()
    out = []
    for a_ in acc:
        n = a_["n"]
        R = a_["R"] / n
        out.append(dict(R=R, varR=a_["R2"] / n - R * R, lam=a_["lam"] / n, dphi=a_["dphi"] / n, n=n))
    return out, float(mods[0].K)


def main():
    out_md = ROOT / sys.argv[1]
    meta = json.load(open(ROOT / "data/arith_settle/tok24k/meta.json"))
    val = np.fromfile(ROOT / "data/arith_settle/tok24k/val.bin", dtype=meta["dtype"])
    L = ["# Settling retest: settling diagnostics (B runs)\n",
         f"Arithmetic val split ({len(val) // SEQ} × {SEQ} tokens), every token. t = 0 is φ⁰ (before settling); λ_t is undefined at t = 0.\n"]
    summary = {}
    for spec in sys.argv[2:]:
        name, path = spec.split("=", 1)
        ck = torch.load(ROOT / path, map_location="cpu", weights_only=False)
        model = COSFOscillatorTransformer(ck["config"]).cuda().eval()
        model.load_state_dict(ck["model_state_dict"])
        res, K = diag(model, val)
        T = len(res[0]["R"]) - 1
        L += [f"## {name} (K = {K:g}, T = {T}, {res[0]['n']:,} tokens)\n",
              "| layer | R_0 | R_T | var(R)_0 | var(R)_T | λ_1 | λ_T | |Δφ|_T | λ_t, t = 1..T |", "|---|---|---|---|---|---|---|---|---|"]
        for i, r in enumerate(res):
            L.append(f"| {i} | {r['R'][0]:.3f} | {r['R'][T]:.3f} | {r['varR'][0]:.4f} | {r['varR'][T]:.4f} | {r['lam'][1]:+.3f} | "
                     f"{r['lam'][T]:+.3f} | {r['dphi'][T]:.3f} | " + " ".join(f"{v:+.2f}" for v in r["lam"][1:]) + " |")
        L.append("")
        summary[name] = [{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in r.items()} for r in res]
        del model
        torch.cuda.empty_cache()
    out_md.write_text("\n".join(L) + "\n")
    out_md.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
