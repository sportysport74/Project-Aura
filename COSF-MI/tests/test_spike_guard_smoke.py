"""Checks on the outputs of scripts/smoke_spike_guard.sh (run it first; it calls this file at the end). Written before the smoke run."""
import glob
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent / "checkpoints" / "smoke" / "guard"
pytestmark = pytest.mark.skipif(not ROOT.exists(), reason="run scripts/smoke_spike_guard.sh first")


def art(name):
    d = ROOT / name
    log = (d / "train.log").read_text(errors="replace").replace("\r", "\n")
    crashes = (d / "CRASHES.md").read_text() if (d / "CRASHES.md").exists() else ""
    state = json.loads((d / "guard_state.json").read_text()) if (d / "guard_state.json").exists() else {"rollbacks": 0, "events": []}
    rc = (d / "rc").read_text().strip()
    done = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(d / "guard_pending.done.*.json")))]
    return d, log, crashes, state, rc, done


def steps(log):
    return [(int(s), float(x)) for s, x in re.findall(r"step=(\d+) loss=([0-9.naif]+)", log)]


def test_t1_recovers_without_rollback():
    d, log, crashes, state, rc, done = art("T1_recover")
    assert rc == "rc=0" and (d / "final_model.pt").exists()
    assert "TRIGGER" in crashes and "RECOVERED" in crashes and "ROLLBACK" not in crashes
    assert state["rollbacks"] == 0 and not done
    assert "ALERT" in log


def test_t2_one_rollback_with_data_skip():
    d, log, crashes, state, rc, done = art("T2_rollback")
    assert rc == "rc=0" and (d / "final_model.pt").exists()
    assert state["rollbacks"] == 1 and len(done) == 1
    p = done[0]
    assert p["checkpoint_step"] <= p["onset"] - 50 and p["target_step"] == p["trigger"]
    a, b = p["skipped_steps"]
    assert f"data of steps {a}-{b} skipped" in crashes
    after = log.split("rollback applied", 1)[1]
    s_after = [s for s, _ in steps(after)]
    assert s_after and s_after[0] == b + 10, "training resumes right after the trigger step"
    assert not [s for s in s_after if a < s <= b], "no step of the skipped window is trained again"
    assert "HELD" in log, "the scheduled save inside the episode was held"
    assert log.count("ALERT") >= 2  # trigger + rollback


def test_t3_second_spike_stops():
    d, log, crashes, state, rc, done = art("T3_second")
    assert rc != "rc=0"
    assert "spike guard STOP" in (d / "FAILED_STUB").read_text()
    assert "STOP:" in crashes and "second spike" in crashes
    assert state["rollbacks"] == 1 and log.count("rollback applied") == 1
    assert not (d / "final_model.pt").exists()


def test_t4_nan_immediate():
    d, log, crashes, state, rc, done = art("T4_nan")
    assert rc == "rc=0" and (d / "final_model.pt").exists()
    assert "non-finite" in crashes and state["rollbacks"] == 1
    p = done[0]
    assert p["target_step"] == 250 and p["at"] == 250, "triggered at the first check after the NaN step (not after N)"
    assert p["checkpoint_step"] <= 250 - 10 - 50
    after = log.split("rollback applied", 1)[1]
    assert all(x == x and x < 20 for _, x in steps(after)), "finite losses after the rollback"


def test_t5_rollback_loads_pre_spike_weights():
    d, log, crashes, state, rc, done = art("T5_savewindow")
    assert rc == "rc=0" and state["rollbacks"] == 1
    first_life = log.split("rollback applied", 1)[0]  # saves after the rollback belong to the relaunched run (its own steps 500+)
    saves = dict((int(s), fp) for s, fp in re.findall(r"saved latest\.pt at step (\d+) \(scheduled\); weights (\S+)", first_life))
    assert 495 in saves and 450 in saves, "the 495 save (spiked weights, between two checks) happened"
    p = done[0]
    assert p["checkpoint_step"] == 450, f"rolled back to {p['checkpoint_step']}"
    held = [int(s) for s in re.findall(r"scheduled save at step (\d+) HELD", first_life)]
    assert held and all(p["onset"] <= s < p["at"] for s in held)
    assert not [s for s in saves if p["onset"] <= s <= p["at"]], "no save inside the episode"
    loaded = re.findall(r"loaded state at step (\d+); weights (\S+)", log)
    assert loaded and int(loaded[-1][0]) == 450
    assert loaded[-1][1] == saves[450] and loaded[-1][1] != saves[495], "the loaded weights are the pre-spike save, not the 495 one"
    pre = [x for s, x in steps(log.split("rollback applied")[0]) if 400 <= s <= 490]
    spiked = [x for s, x in steps(log.split("rollback applied")[0]) if 500 <= s <= 540]
    post = [x for s, x in steps(log.split("rollback applied", 1)[1])][:5]
    assert max(spiked) > max(pre) + 0.5, "the injected noise really spiked the loss"
    assert sum(post) / len(post) < sum(pre) / len(pre) + 0.3, "loss after the rollback is back at the pre-spike level"
