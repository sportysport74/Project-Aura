"""Loss-spike guard for long pretrains (experiments/plans/aura_v3_base.md, "Step 3 changes", 2026-11-09).

The trainer adds every step's task loss to a GPU accumulator and hands the guard the 10-step mean at its existing sync point (every 10 steps),
so the guard adds no GPU syncs except right before a checkpoint save.

  signal   = mean loss over the last `window` steps
  baseline = mean over the `baseline` steps before those (frozen while an episode is open, so a spike never raises its own baseline)
  excess   = signal - baseline
  armed from step `arm_step`; a non-finite loss counts at any step.

Episode: onset = first check with excess > m/2; trigger = excess > m. After a trigger, excess <= m/2 within `recover_steps` -> RECOVERED (logged,
alert). Otherwise -> ROLLBACK: the newest full-state checkpoint with step <= onset - window is chosen, the trainer writes guard_pending.json
and exits with ROLLBACK_EXIT; the relaunch loads that checkpoint and moves the data position and step counter to the trigger step (that
window of data is skipped). A non-finite loss -> ROLLBACK at once. A second rollback, or no checkpoint before onset -> STOP (STOP_EXIT).
No checkpoint is saved while an episode is open (from onset until recovery), so a save can't capture spiked weights.
Every trigger, recovery, rollback (with the skipped step range) and stop is appended to <checkpoint_dir>/CRASHES.md and sent by ntfy.
"""
import json
import math
import subprocess
import time
from collections import deque
from pathlib import Path

ROLLBACK_EXIT = 76
STOP_EXIT = 77
CHECK_EVERY = 10


class SpikeGuard:
    def __init__(self, checkpoint_dir, margin=0.10, window=100, baseline=2000, arm_step=4000, recover_steps=500, max_rollbacks=1,
                 crash_log=None, notify=True):
        assert window % CHECK_EVERY == 0 and baseline % CHECK_EVERY == 0
        self.dir = Path(checkpoint_dir)
        self.m, self.W, self.B = margin, window // CHECK_EVERY, baseline // CHECK_EVERY
        self.window, self.arm_step, self.recover_steps, self.max_rollbacks = window, arm_step, recover_steps, max_rollbacks
        self.crash_log = Path(crash_log) if crash_log else self.dir / "CRASHES.md"
        self.notify_on = notify
        self.log_only = False  # soak calibration: log what would happen, never roll back (a non-finite loss still stops)
        self.recent = deque(maxlen=self.W)   # last W 10-step means (the signal)
        self.base = deque(maxlen=self.B)     # the B before those; readings that leave `recent` during an episode are dropped
        self.onset = None          # (step, position) of the first check above m/2 in this episode
        self.trigger = None        # (step, position, excess) once above m
        self.frozen_base = None
        self.state_file = self.dir / "guard_state.json"
        self.state = json.loads(self.state_file.read_text()) if self.state_file.exists() else {"rollbacks": 0, "events": []}

    # ------------------------------------------------------------------ persistence (history goes into each full-state checkpoint)
    def state_dict(self):
        return {"recent": list(self.recent), "base": list(self.base)}

    def load_state_dict(self, d):
        if d:
            self.recent.extend(d.get("recent", []))
            self.base.extend(d.get("base", []))

    @property
    def episode_open(self):
        return self.onset is not None

    def save_ok(self):
        return not self.episode_open

    # ------------------------------------------------------------------ main entry, every 10 steps
    def update(self, step, mean10, position):
        """mean10: mean task loss of the last 10 steps (float). position: (epoch, batch_in_epoch) after this step.
        Returns None, or ('rollback', info) / ('stop', info) for the trainer to act on."""
        if not math.isfinite(mean10):
            onset = self.onset or (step - CHECK_EVERY, position)
            return self._rollback(step, position, onset, f"non-finite loss ({mean10}) at step {step}")
        if len(self.recent) == self.W:
            y = self.recent[0]
            if self.onset is None:
                self.base.append(y)
        self.recent.append(mean10)
        if len(self.base) < self.B or step < self.arm_step:
            return None
        signal = sum(self.recent) / self.W
        base = self.frozen_base if self.frozen_base is not None else sum(self.base) / self.B
        excess = signal - base
        if self.trigger is None:
            if excess > self.m / 2:
                if self.onset is None:
                    self.onset, self.frozen_base = (step, position), base
                    self._log(f"onset: excess {excess:+.3f} > m/2 at step {step} (baseline {base:.4f} frozen; saves held)", alert=False)
                if excess > self.m:
                    self.trigger = (step, position, excess)
                    self._log(f"TRIGGER: excess {excess:+.3f} > m = {self.m} at step {step} (onset step {self.onset[0]})", alert=True)
            elif self.onset is not None:  # back under m/2 before a trigger: not an episode
                self._log(f"onset cleared at step {step} without a trigger (excess {excess:+.3f}); saves resume", alert=False)
                self.onset = self.frozen_base = None
        else:
            if excess <= self.m / 2:
                t = self.trigger[0]
                self._log(f"RECOVERED at step {step}, {step - t} steps after the trigger at {t} (excess {excess:+.3f}); no rollback",
                          alert=True, event={"kind": "recovered", "trigger": t, "onset": self.onset[0], "at": step})
                self.onset = self.trigger = self.frozen_base = None
            elif step - self.trigger[0] >= self.recover_steps:
                return self._rollback(step, position, self.onset,
                                      f"no recovery within {self.recover_steps} steps of the trigger at {self.trigger[0]} "
                                      f"(excess {excess:+.3f} at step {step})")
        return None

    # ------------------------------------------------------------------ rollback / stop
    def _rollback(self, step, position, onset, why):
        onset_step = onset[0]
        if self.log_only and "non-finite" not in why:
            self._log(f"(log-only) WOULD ROLL BACK: {why}; episode closed, training continues", alert=True,
                      event={"kind": "would_rollback", "onset": onset_step, "at": step})
            self.onset = self.trigger = self.frozen_base = None
            return None
        if self.log_only:
            return self._stop(f"{why} (log-only run)", step)
        trig_step, trig_pos = (self.trigger[0], self.trigger[1]) if self.trigger else (step, position)
        if self.state["rollbacks"] >= self.max_rollbacks:
            return self._stop(f"second spike needing a rollback ({why}); rollbacks so far: {self.state['rollbacks']}", step)
        ck = self._checkpoint_before(onset_step - self.window)
        if ck is None:
            return self._stop(f"{why}; no full-state checkpoint at or before step {onset_step - self.window}", step)
        path, ck_step = ck
        info = {"kind": "rollback", "why": why, "onset": onset_step, "trigger": trig_step, "at": step, "checkpoint": str(path),
                "checkpoint_step": ck_step, "target_step": trig_step, "target_epoch": trig_pos[0], "target_batch": trig_pos[1],
                "skipped_steps": [ck_step, trig_step], "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        self.state["rollbacks"] += 1
        self._log(f"ROLLBACK: {why}. Loading {path.name} (step {ck_step}); data of steps {ck_step}-{trig_step} skipped "
                  f"(never trained on again), training continues at step {trig_step}", alert=True, event=info)
        (self.dir / "guard_pending.json").write_text(json.dumps(info, indent=1))
        return "rollback", info

    def _stop(self, why, step):
        info = {"kind": "stop", "why": why, "at": step, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        self._log(f"STOP: {why}. Training stopped; waiting for Jon (no further rollbacks)", alert=True, event=info)
        return "stop", info

    def _checkpoint_before(self, max_step):
        import torch
        best = None
        for name in ("latest.pt", "latest.prev.pt"):
            p = self.dir / name
            if not p.exists():
                continue
            try:
                s = torch.load(p, map_location="cpu", weights_only=False, mmap=True)["global_step"]
            except Exception:
                continue
            if s <= max_step and (best is None or s > best[1]):
                best = (p, s)
        return best

    # ------------------------------------------------------------------ logging
    def _log(self, msg, alert=False, event=None):
        line = f"- {time.strftime('%Y-%m-%d %H:%M:%S')} **spike guard** ({self.dir}): {msg}"
        print(f"[spike guard] {msg}", flush=True)
        new = not self.crash_log.exists()
        with open(self.crash_log, "a") as f:
            if new:
                f.write(f"# Faults and guard events: {self.dir}\n\n")
            f.write(line + "\n")
        if event:
            self.state["events"].append(event)
        self.state_file.write_text(json.dumps(self.state, indent=1))
        if alert:
            print(f"[spike guard] ALERT{'' if self.notify_on else ' (ntfy off)'}: {msg[:200]}", flush=True)
        if alert and self.notify_on:
            try:
                subprocess.Popen(["bash", "scripts/notify.sh", f"spike guard: {msg[:200]}", "4"],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
