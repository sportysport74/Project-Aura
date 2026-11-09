"""Unit tests for spike_guard.SpikeGuard (CPU, no training): the five behaviours the v3 plan requires, on synthetic loss streams."""
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spike_guard import SpikeGuard  # noqa: E402

KW = dict(margin=0.10, window=100, baseline=2000, arm_step=4000, recover_steps=500, notify=False)


def ck(d, name, step):
    torch.save({"global_step": step}, d / name)


def run(g, start, stop, f):
    """Feed 10-step means f(step) for step = start, start+10, ... < stop; return the first non-None action."""
    for s in range(start, stop, 10):
        a = g.update(s, f(s), (0, s))
        if a:
            return s, a
    return None, None


def flat(s):
    return 3.0 + 0.003 * math.sin(s / 37.0)  # a little deterministic noise, sd ~0.002


def test_quiet_never_triggers(tmp_path):
    g = SpikeGuard(tmp_path, **KW)
    s, a = run(g, 10, 40000, flat)
    assert a is None and g.save_ok() and not (tmp_path / "CRASHES.md").exists()


def test_spike_that_recovers_no_rollback(tmp_path):
    g = SpikeGuard(tmp_path, **KW)
    run(g, 10, 5000, flat)
    s, a = run(g, 5000, 8000, lambda s: flat(s) + (0.5 if 5000 <= s < 5200 else 0.0))
    assert a is None
    ev = json.loads((tmp_path / "guard_state.json").read_text())
    assert ev["rollbacks"] == 0 and [e["kind"] for e in ev["events"]] == ["recovered"]
    log = (tmp_path / "CRASHES.md").read_text()
    assert "TRIGGER" in log and "RECOVERED" in log and "ROLLBACK" not in log
    assert g.save_ok()


def test_spike_without_recovery_rolls_back_once_with_skip(tmp_path):
    ck(tmp_path, "latest.prev.pt", 4800)
    ck(tmp_path, "latest.pt", 5050)  # saved after the spike began (before onset was seen): must NOT be chosen
    g = SpikeGuard(tmp_path, **KW)
    run(g, 10, 5000, flat)
    s, a = run(g, 5000, 9000, lambda s: flat(s) + (0.5 if s >= 5000 else 0.0))
    kind, info = a
    assert kind == "rollback"
    assert info["checkpoint_step"] == 4800 and info["checkpoint_step"] <= info["onset"] - 100
    assert info["target_step"] == info["trigger"] and s - info["trigger"] >= 500
    assert info["skipped_steps"] == [4800, info["trigger"]]
    assert (tmp_path / "guard_pending.json").exists()
    assert "ROLLBACK" in (tmp_path / "CRASHES.md").read_text()


def test_saves_held_during_episode(tmp_path):
    g = SpikeGuard(tmp_path, **KW)
    run(g, 10, 5000, flat)
    g.update(5000, flat(5000) + 2.0, (0, 5000))  # one reading of +2.0 -> 100-step signal +0.2 > m: onset and trigger
    assert not g.save_ok()


def test_second_rollback_stops(tmp_path):
    ck(tmp_path, "latest.prev.pt", 4800)
    g = SpikeGuard(tmp_path, **KW)
    run(g, 10, 5000, flat)
    _, a = run(g, 5000, 9000, lambda s: flat(s) + (0.5 if s >= 5000 else 0.0))
    assert a[0] == "rollback"
    g2 = SpikeGuard(tmp_path, **KW)  # the relaunched process: rollbacks = 1 persists in guard_state.json
    run(g2, 10, 5000, flat)
    _, a2 = run(g2, 5000, 9000, lambda s: flat(s) + (0.5 if s >= 5000 else 0.0))
    assert a2[0] == "stop" and "second spike" in a2[1]["why"]


def test_nonfinite_immediate_even_before_arming(tmp_path):
    ck(tmp_path, "latest.pt", 100)
    for bad in (float("nan"), float("inf")):
        g = SpikeGuard(tmp_path, **KW)
        g.state["rollbacks"] = 0
        a = g.update(250, bad, (0, 250))
        assert a[0] == "rollback" and a[1]["target_step"] == 250 and a[1]["checkpoint_step"] == 100


def test_no_checkpoint_before_onset_stops(tmp_path):
    ck(tmp_path, "latest.pt", 5050)
    g = SpikeGuard(tmp_path, **KW)
    run(g, 10, 5000, flat)
    _, a = run(g, 5000, 9000, lambda s: flat(s) + (0.5 if s >= 5000 else 0.0))
    assert a[0] == "stop"


def test_history_survives_checkpoint(tmp_path):
    g = SpikeGuard(tmp_path, **KW)
    run(g, 10, 5000, flat)
    g2 = SpikeGuard(tmp_path, **KW)
    g2.load_state_dict(g.state_dict())
    assert list(g2.recent) == list(g.recent) and list(g2.base) == list(g.base)
    g2.update(5000, flat(5000) + 2.0, (0, 5000))
    assert not g2.save_ok()  # armed straight away after a resume
