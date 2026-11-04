"""Unit tests for the settling module (models/settling.py), 2026-11-03."""
import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "python"))
from cosf_mi.models.cosf_oscillator import COSFOscillatorConfig, COSFOscillatorTransformer  # noqa: E402
from cosf_mi.models.settling import SettlingModule  # noqa: E402


def cfg(**kw):
    c = COSFOscillatorConfig(vocab_size=500, d_model=32, n_layers=2, n_heads=4, d_ff=64, max_seq_len=64, projection="linear",
                             emb_scale="sqrt", ffn_activation="gelu", linear_init="gpt", heartbeat_mode="live", dropout=0.0)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_k0_reduces_to_plain_block():
    torch.manual_seed(0)
    m = SettlingModule(32, 2, T=8, K=0.0).eval()
    x = torch.randn(2, 10, 32)
    assert torch.allclose(m(x), m.plain(x), atol=1e-6)


def test_repeat_once_equals_k0_settle():
    torch.manual_seed(0)
    a = SettlingModule(32, 2, T=8, K=0.0)
    b = copy.deepcopy(a)
    b.mode, b.repeats = "repeat", 1
    x = torch.randn(2, 10, 32)
    assert torch.allclose(a(x), b(x), atol=1e-6)


def test_coupling_is_symmetric_and_matches_definition():
    torch.manual_seed(0)
    m = SettlingModule(16, 2, n_osc=16, rank=4)
    J = m.U @ torch.diag(m.s) @ m.U.t()
    assert torch.allclose(J, J.t(), atol=1e-6)
    phi = torch.randn(3, 16)
    ref = (J.unsqueeze(0) * torch.sin(phi.unsqueeze(1) - phi.unsqueeze(2))).sum(-1)  # Σ_j J_ij sin(φ_j − φ_i)
    assert torch.allclose(m.coupling(phi), ref, atol=1e-5)


def test_settling_moves_phases_when_coupled():
    torch.manual_seed(0)
    m = SettlingModule(32, 2, T=8, K=2.0).eval()
    x = torch.randn(2, 10, 32)
    assert not torch.allclose(m(x), m.plain(x), atol=1e-4)


def test_clamp_divisors():
    torch.manual_seed(0)
    for clamp, div in (("phi", (1 + 5 ** 0.5) / 2), ("two", 2.0)):
        m = SettlingModule(16, 2, T=1, K=50.0, clamp=clamp, v_max=0.01)
        phi0 = torch.randn(1, 4, 16)
        free = SettlingModule(16, 2, T=1, K=50.0)
        free.load_state_dict(m.state_dict())
        v_free = free.step(phi0, phi0 + 0.5, m.K)[0] - phi0
        v = m.step(phi0, phi0 + 0.5, m.K)[0] - phi0
        big = v_free.abs() > 0.01
        assert big.any()
        assert torch.allclose(v[big], v_free[big] / div, atol=1e-6)


def test_no_future_token_leakage():
    for mode in ("settle", "repeat"):
        torch.manual_seed(0)
        model = COSFOscillatorTransformer(cfg(settle_mode=mode, settle_repeats=3)).eval()
        x = torch.randint(0, 500, (1, 20))
        y = x.clone()
        y[0, 12:] = (y[0, 12:] + 7) % 500
        with torch.no_grad():
            a, b = model(x), model(y)
        assert torch.allclose(a[0, :12], b[0, :12], atol=1e-5), mode
        assert not torch.allclose(a[0, 12:], b[0, 12:], atol=1e-5)


def test_diagnostics_and_controller():
    torch.manual_seed(0)
    model = COSFOscillatorTransformer(cfg(settle_mode="settle", settle_controller="auto")).train()
    x = torch.randint(0, 500, (4, 20))
    _, loss, _, _ = model(x, x)
    loss.backward()
    k0 = float(model.blocks[0].settle.K)
    out = model.settle_after_step(1, log=True)
    assert len(out["modules"]) == 2 and len(out["modules"][0]["R"]) == 8 and len(out["modules"][0]["lyap"]) == 8
    assert out["controller_signal"] == "lyapunov"
    assert float(model.blocks[0].settle.K) != k0  # the controller moved K
