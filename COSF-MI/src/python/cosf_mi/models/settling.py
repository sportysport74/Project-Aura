"""
Criticality-controlled settling module (2026-11-03; experiments/plans/settling_vanilla2.md).

A block added to the residual stream after each transformer block's FFN. Per token (never across tokens, so it's causal by
construction), it maps the normalized hidden state to N oscillator phases and lets them settle for T weight-tied Kuramoto steps:

    φ⁰ = W_in · LN(x)
    φ ← φ + v,   v = γ · [ K · Σ_j J_ij sin(φ_j − φ_i) + β · sin(φ⁰ − φ) ]        (T steps)
    out = W_out · [cos φ ; sin φ]                                                 (added to the residual stream)

- J is symmetric by construction: J = U diag(s) Uᵀ (rank r), so Σ_j J_ij sin(φ_j − φ_i) = cos φ_i (J sin φ)_i − sin φ_i (J cos φ)_i
  costs four rank-r matmuls per step, not N².
- K is a fixed constant (a buffer, not learned). The criticality controller (`controller='auto'`) adjusts it between training steps.
- Velocity clamp (`clamp='phi'|'two'`): any |v_i| > v_max is divided by φ (golden ratio) or by 2 (the question: is φ special?).
- K = 0 ⇒ the phases never move (the anchor term is 0 at φ = φ⁰), so the module reduces to a plain single-pass block
  W_out·[cos, sin](W_in·LN(x)).
- mode 'repeat' (the R4 control): no dynamics. The same plain block is applied `repeats` times, weight-tied, on the residual
  (z ← z + B(z)), and the module returns z − x. With repeats = 1 it equals settling with K = 0.

Phases are fp32 (autocast off), as in the oscillator layers. Diagnostics and the controller run outside the compiled forward, on a detached
4-sequence sub-batch of φ⁰ kept from the last forward: per settling step, the Kuramoto order R (mean over tokens), var(R) across the batch's
tokens, and a finite-difference Lyapunov estimate λ_t = ln(‖δ_t‖ / ‖δ_0‖) / t from a 1e-4 perturbation of φ⁰.
"""
import math

import torch
import torch.nn as nn

PHI = (1 + 5 ** 0.5) / 2


class SettlingModule(nn.Module):
    def __init__(self, d_model, n_layers, n_osc=None, rank=64, T=8, K=1.0, gamma=0.25, beta=0.5, clamp="none", v_max=0.25,
                 mode="settle", repeats=1):
        super().__init__()
        n = n_osc or d_model
        self.n, self.T, self.gamma, self.beta, self.mode, self.repeats = n, T, gamma, beta, mode, repeats
        self.clamp_div = {"none": None, "phi": PHI, "two": 2.0}[clamp]
        self.v_max = v_max
        self.ln = nn.LayerNorm(d_model)
        self.w_in = nn.Linear(d_model, n)
        nn.init.normal_(self.w_in.weight, 0.0, 1.0 / math.sqrt(d_model))  # φ⁰ ≈ unit-scale phases
        nn.init.zeros_(self.w_in.bias)
        self.U = nn.Parameter(torch.randn(n, rank) / math.sqrt(n))
        self.s = nn.Parameter(torch.ones(rank))
        self.w_out = nn.Linear(2 * n, d_model)
        nn.init.normal_(self.w_out.weight, 0.0, 0.02 / math.sqrt(2 * n_layers))  # GPT residual rule: starts near a no-op
        nn.init.zeros_(self.w_out.bias)
        self.register_buffer("K", torch.tensor(float(K)))
        self.last_phi0 = None
        self.last_clamp_frac = None

    # ------------------------------------------------------------------ dynamics
    def coupling(self, phi):
        s_, c_ = torch.sin(phi), torch.cos(phi)
        js = ((s_ @ self.U) * self.s) @ self.U.t()
        jc = ((c_ @ self.U) * self.s) @ self.U.t()
        return c_ * js - s_ * jc  # Σ_j J_ij sin(φ_j − φ_i)

    def step(self, phi, phi0, K):
        v = self.gamma * (K * self.coupling(phi) + self.beta * torch.sin(phi0 - phi))
        if self.clamp_div is not None:
            big = v.abs() > self.v_max
            v = torch.where(big, v / self.clamp_div, v)
            return phi + v, big
        return phi + v, None

    def readout(self, phi):
        return self.w_out(torch.cat([torch.cos(phi), torch.sin(phi)], dim=-1))

    def plain(self, x):
        return self.readout(self.w_in(self.ln(x)))

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            x = x.float()
            if self.mode == "repeat":
                z = x
                for _ in range(self.repeats):
                    z = z + self.plain(z)
                return z - x
            phi0 = self.w_in(self.ln(x))
            self.last_phi0 = phi0[:4].detach()
            phi, frac = phi0, []
            for _ in range(self.T):
                phi, big = self.step(phi, phi0, self.K)
                if big is not None:
                    frac.append(big.float().mean())
            if frac:
                self.last_clamp_frac = torch.stack(frac).mean().detach()
            return self.readout(phi)

    # ------------------------------------------------------------------ diagnostics (outside the compiled forward)
    @torch.no_grad()
    def diagnose(self, eps=1e-4):
        """Per settling step t = 1..T on the last sub-batch: R_t, var(R)_t across tokens, λ_t. Returns GPU tensors [T] (no sync)."""
        if self.mode == "repeat" or self.last_phi0 is None:
            return None
        phi0 = self.last_phi0.float()
        d0 = torch.randn_like(phi0)
        d0 = eps * d0 / d0.norm(dim=-1, keepdim=True)
        a, b = phi0, phi0 + d0
        R, varR, lyap = [], [], []
        for t in range(1, self.T + 1):
            a, _ = self.step(a, phi0, self.K)
            b, _ = self.step(b, phi0 + d0, self.K)
            r = torch.complex(torch.cos(a), torch.sin(a)).mean(-1).abs()  # per token
            R.append(r.mean())
            varR.append(r.var())
            ratio = (b - a).norm(dim=-1) / eps
            lyap.append(torch.log(ratio.clamp_min(1e-12)).mean() / t)
        return torch.stack(R), torch.stack(varR), torch.stack(lyap)


class CriticalityController:
    """Keeps each module near the edge of chaos by adjusting its K between training steps (all on the GPU; one sync at the decision).

    Signal: the Lyapunov estimate at the last settling step, smoothed (EMA 0.9): K ← K · exp(−η · λ̄), so λ̄ > 0 (diverging) lowers K and
    λ̄ < 0 (contracting) raises it. At step `decide_at`, the noise of the raw estimate over steps [decide_at/2, decide_at) is checked: if
    its sd exceeds 2 × |mean| and |mean| < sd, the Lyapunov signal counts as too noisy and the controller switches to keeping R in a band
    [r_lo, r_hi]: K ← K · exp(η · (r_lo − R̄)) below the band, K · exp(−η · (R̄ − r_hi)) above it. `signal` records which one is in use.
    """

    def __init__(self, modules, eta=0.05, k_min=0.0, k_max=20.0, decide_at=500, r_lo=0.3, r_hi=0.6):
        self.modules, self.eta, self.k_min, self.k_max = modules, eta, k_min, k_max
        self.decide_at, self.r_lo, self.r_hi = decide_at, r_lo, r_hi
        self.signal = "lyapunov"
        self.ema_l = [None] * len(modules)
        self.ema_r = [None] * len(modules)
        self.hist = []  # raw λ (last step) of module 0, steps [decide_at/2, decide_at)
        self.decision = None

    def after_step(self, step):
        for i, m in enumerate(self.modules):
            d = m.diagnose()
            if d is None:
                continue
            R, _, lyap = d
            lam, r = lyap[-1], R[-1]
            self.ema_l[i] = lam if self.ema_l[i] is None else 0.9 * self.ema_l[i] + 0.1 * lam
            self.ema_r[i] = r if self.ema_r[i] is None else 0.9 * self.ema_r[i] + 0.1 * r
            if i == 0 and self.decide_at // 2 <= step < self.decide_at:
                self.hist.append(lam)
            if self.signal == "lyapunov":
                factor = torch.exp(-self.eta * self.ema_l[i])
            else:
                rb = self.ema_r[i]
                factor = torch.where(rb < self.r_lo, torch.exp(self.eta * (self.r_lo - rb)),
                                     torch.where(rb > self.r_hi, torch.exp(-self.eta * (rb - self.r_hi)), torch.ones_like(rb)))
            m.K.copy_((m.K * factor).clamp(self.k_min, self.k_max))
        if step == self.decide_at and self.hist and self.decision is None:
            h = torch.stack(self.hist).float()
            mean, sd = h.mean().item(), h.std().item()
            noisy = sd > 2 * abs(mean) and abs(mean) < sd
            self.signal = "r_band" if noisy else "lyapunov"
            self.decision = {"step": step, "lyap_mean": mean, "lyap_sd": sd, "noisy": noisy, "signal": self.signal}
        return self.decision
