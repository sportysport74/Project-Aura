"""
Experiment tank: one fish per running job, swimming in the terminal.

Watches the main training log (like fish.py) plus every A/B arm log in
experiments/*/logs/ that changed in the last few hours. Each job gets its own
fish and colour:
  - swims faster when its job runs faster, blows bubbles
  - naps (zzz) right after a sleep cycle
  - rests on the sand with a check mark when its job is done
  - floats belly-up if its log goes stale (job stopped?)
Below the tank: progress bar, step, val loss, live R and ETA for every job.

Usage:  python tank.py [training_log]      Ctrl+C to quit
"""

import glob
import math
import os
import random
import re
import shutil
import sys
import time

from fish import LogWatcher, bar

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_LOG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "logs", "heartbeat_fixed_10ep.log")
ARM_GLOB = os.path.join(HERE, "experiments", "*", "logs", "*.log")
RECENT_HOURS = 6
STALE_SECONDS = 240

STEP_RE = re.compile(r"\[([^\]]+)\] step (\d+): val ([\d.]+) liveR ([\d.]+).*\(([\d.]+) min\)")
START_RE = re.compile(r"\[([^\]]+)\] start: steps (\d+)")
DONE_RE = re.compile(r"\[([^\]]+)\] done: final val ([\d.]+)")
SLEEP_RE = re.compile(r"\[([^\]]+)\] sleep @ (\d+)")
ERR_RE = re.compile(r"Traceback|Error")

RESET, BLUE, CYAN, WHITE, DIM, RED, GREEN = (
    "\x1b[0m", "\x1b[34m", "\x1b[36m", "\x1b[97m", "\x1b[2m", "\x1b[91m", "\x1b[32m")
FISH_COLORS = ["\x1b[93m", "\x1b[95m", "\x1b[96m", "\x1b[92m", "\x1b[33m", "\x1b[35m", "\x1b[94m", "\x1b[37m"]
SHAPES = [("><(((o>", "<o)))><"), ("><>", "<><"), ("><((o>", "<o))><"), (">=)o>", "<o(=<")]


class Job:
    """One thing being waited on: the training run or an A/B arm."""

    def __init__(self, key, name, color, shape, row):
        self.key, self.name, self.color = key, name, color
        self.right, self.left = shape
        self.row = row  # home depth, 0..1
        self.x = random.uniform(2, 30)
        self.dir = random.choice((-1, 1))
        self.phase = random.uniform(0, 6.28)
        self.frac, self.rate, self.eta = 0.0, 0.0, "--"
        self.info, self.done, self.stale, self.error = "", False, False, False
        self.nap_until = 0.0


def read_arm(path, job, now):
    try:
        st = os.stat(path)
        text = open(path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return
    total = 10000
    for _, n in START_RE.findall(text):
        total = int(n)
    steps = STEP_RE.findall(text)
    done = DONE_RE.findall(text)
    job.error = bool(ERR_RE.search(text)) and not done
    job.done = bool(done)
    job.stale = not job.done and now - st.st_mtime > STALE_SECONDS
    sleeps = SLEEP_RE.findall(text)
    if sleeps and now - st.st_mtime < 6 and not job.done:
        job.nap_until = now + 6
    if steps:
        _, step, val, live_r, minutes = steps[-1]
        step, minutes = int(step), float(minutes)
        job.frac = 1.0 if job.done else min(1.0, step / total)
        job.rate = step / max(minutes * 60, 1)
        left = (total - step) / job.rate if job.rate and not job.done else 0
        job.eta = "done" if job.done else f"{int(left // 60)}m"
        shown = done[-1][1] if done else val
        job.info = f"val {float(shown):.4f}  R {float(live_r):.2f}  {step}/{total}"
    elif not job.done:
        job.info = "loading data..."


def read_train(watch, job):
    watch.poll()
    s = watch.status
    job.stale = bool(watch.stale)
    if time.time() < watch.sleeping_until:
        job.nap_until = watch.sleeping_until
    if s:
        ep_frac = s["step"] / max(1, s["steps"])
        job.frac = ((s["ep"] - 1) + ep_frac) / max(1, s["eps"])
        job.rate = s["its"]
        job.eta = f"ep {s['eta']}"
        best = f"{min(watch.vals):.4f}" if watch.vals else "--"
        job.info = f"loss {s['loss']:.4f}  best val {best}  epoch {s['ep']}/{s['eps']}"
        job.done = job.frac >= 0.999 and job.stale
    else:
        job.info = "waiting for log..."


def main():
    os.system("")  # enable ANSI escapes on Windows consoles
    watch = LogWatcher(TRAIN_LOG)
    jobs = {}
    bubbles = []
    weeds = None
    t0 = time.time()
    last_poll = 0.0
    sys.stdout.write("\x1b[?25l\x1b[2J")

    def add(key, name):
        i = len(jobs)
        jobs[key] = Job(key, name, FISH_COLORS[i % len(FISH_COLORS)],
                        SHAPES[0] if key == "train" else SHAPES[1 + i % 3], random.uniform(0.15, 0.7))

    try:
        while True:
            now = time.time()
            if now - last_poll > 1.5:
                last_poll = now
                if os.path.exists(TRAIN_LOG) and "train" not in jobs:
                    add("train", "TRAINING " + os.path.basename(TRAIN_LOG)[:-4])
                if "train" in jobs:
                    read_train(watch, jobs["train"])
                for path in sorted(glob.glob(ARM_GLOB), key=os.path.getmtime):
                    if now - os.path.getmtime(path) > RECENT_HOURS * 3600:
                        continue
                    if path not in jobs:
                        rnd = os.path.basename(os.path.dirname(os.path.dirname(path)))
                        add(path, f"{rnd}/{os.path.basename(path)[:-4].lstrip('-')}")
                    read_arm(path, jobs[path], now)
                # keep the 8 most relevant: training, running, then newest finished
                if len(jobs) > 8:
                    keep = sorted(jobs.values(), key=lambda j: (j.key != "train", j.done))[:8]
                    jobs = {j.key: j for j in keep}

            cols, rows = shutil.get_terminal_size((100, 30))
            w = max(40, cols - 1)
            n = len(jobs)
            h = max(8, min(rows - n - 5, 18))
            if weeds is None or len(weeds) != w:
                weeds = [random.random() < 0.1 for _ in range(w)]

            grid = [[" "] * w for _ in range(h)]
            colors = [[BLUE] * w for _ in range(h)]
            sway = int(now * 2) % 2
            for i, has in enumerate(weeds):
                if has:
                    for k in range(1, 4):
                        grid[h - k][i] = "(" if (k + sway) % 2 else ")"
                        colors[h - k][i] = GREEN
            for i in range(w):  # sand
                if grid[h - 1][i] == " ":
                    grid[h - 1][i] = "."
                    colors[h - 1][i] = "\x1b[33m"

            for j in jobs.values():
                napping = now < j.nap_until
                fish = j.right if j.dir > 0 else j.left
                if j.done:
                    fy, speed = h - 2, 0.03
                elif j.stale or j.error:
                    fy, speed = 0, 0.0
                    fish = fish.replace("o", "x")
                else:
                    fy = int(1 + j.row * (h - 4) + math.sin((now - t0) * 1.2 + j.phase) * 1.5)
                    speed = 0.0 if napping else 0.12 + min(j.rate, 20) / 20 * 0.6
                fy = max(0, min(h - 2, fy))
                j.x += j.dir * speed
                if j.x >= w - len(fish) - 1:
                    j.x, j.dir = w - len(fish) - 1, -1
                elif j.x <= 1:
                    j.x, j.dir = 1, 1
                elif random.random() < 0.004:
                    j.dir *= -1
                fx = int(j.x)
                if speed > 0.05 and random.random() < 0.08:
                    mouth = fx + len(fish) if j.dir > 0 else fx - 1
                    bubbles.append([mouth, fy, random.choice(".oO")])
                color = RED if (j.stale or j.error) else j.color
                for i, ch in enumerate(fish):
                    if 0 <= fx + i < w:
                        grid[fy][fx + i] = ch
                        colors[fy][fx + i] = color
                tag = "zzz"[: int(now * 2) % 4] if napping else ("ok" if j.done else "")
                for i, ch in enumerate(tag):
                    tx = fx + len(fish) + i
                    if fy > 0 and tx < w:
                        grid[fy - 1][tx] = ch
                        colors[fy - 1][tx] = CYAN if napping else GREEN

            for b in bubbles:
                b[1] -= 0.3
                b[0] += random.choice((-1, 0, 0, 1)) * 0.3
            bubbles = [b for b in bubbles if b[1] > 0 and 0 <= b[0] < w][-60:]
            for b in bubbles:
                if grid[int(b[1])][int(b[0])] == " ":
                    grid[int(b[1])][int(b[0])] = b[2]
                    colors[int(b[1])][int(b[0])] = WHITE

            out = ["\x1b[H", CYAN + "~" * w + RESET + "\n"]
            for r in range(h):
                line, cur = [], None
                for c in range(w):
                    if colors[r][c] != cur:
                        cur = colors[r][c]
                        line.append(cur)
                    line.append(grid[r][c])
                out.append("".join(line) + RESET + "\x1b[K\n")
            out.append(DIM + "=" * w + RESET + "\x1b[K\n")

            name_w = min(30, max((len(j.name) for j in jobs.values()), default=10))
            bw = max(8, w - name_w - 58)
            for j in jobs.values():
                state = (RED + "STALE" if j.stale else RED + "ERROR" if j.error
                         else GREEN + "done " if j.done else CYAN + "nap  " if now < j.nap_until
                         else j.color + "swim ")
                out.append(f" {j.color}{'><>':<4}{RESET}"
                           f"{j.name[:name_w]:<{name_w}} [{bar(j.frac, bw)}] {j.frac * 100:5.1f}% "
                           f"{state}{RESET} {j.eta:>8}  {DIM}{j.info}{RESET}\x1b[K\n")
            if not jobs:
                out.append(" No jobs yet - waiting for logs...\x1b[K\n")
            out.append(DIM + " Ctrl+C to quit" + RESET + "\x1b[K\x1b[J")
            sys.stdout.write("".join(out))
            sys.stdout.flush()
            time.sleep(0.08)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write(RESET + "\x1b[?25h\n")


if __name__ == "__main__":
    main()
