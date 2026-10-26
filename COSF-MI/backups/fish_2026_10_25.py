"""
Training aquarium: a fish swims in the terminal while COSF trains.

The fish swims faster when training runs faster, blows bubbles, and the
status bar shows epoch, progress, loss and best val loss from the log.
It naps (zzz) during sleep cycles and floats belly-up if the log goes stale.
A school of smaller fish swims along, and a fisherman drifts by in his boat;
he gets excited (!) whenever a fish swims past his hook.

Usage:  python fish.py [path/to/log]      Ctrl+C to quit
"""

import math
import os
import random
import re
import shutil
import sys
import time

def newest_log():
    """The most recently written training log in logs/."""
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    logs = [os.path.join(logs_dir, f) for f in os.listdir(logs_dir) if f.endswith(".log")]
    return max(logs, key=os.path.getmtime) if logs else os.path.join(logs_dir, "training.log")


LOG = sys.argv[1] if len(sys.argv) > 1 else newest_log()

TQDM_RE = re.compile(
    r"Epoch (\d+)/(\d+):\s+(\d+)%\|[^|]*\|\s*(\d+)/(\d+) \[[^<]*<([^,]*),\s*([\d.]+)it/s, loss=([\d.]+)")
VAL_RE = re.compile(r"Val loss: ([\d.]+)")

FISH_R = "><(((o>"
FISH_L = "<o)))><"
FISH_DEAD_R = "><(((x>"
FISH_DEAD_L = "<x)))><"

# ANSI colours
RESET, BLUE, CYAN, YELLOW, GREEN, WHITE, DIM, RED = (
    "\x1b[0m", "\x1b[34m", "\x1b[36m", "\x1b[93m", "\x1b[32m", "\x1b[97m", "\x1b[2m", "\x1b[91m")


class LogWatcher:
    """Reads only the tail of the (large, tqdm-heavy) log on each poll."""

    def __init__(self, path):
        self.path = path
        self.vals = []
        self.pos = 0
        self.status = None
        self.sleeping_until = 0.0
        self.mtime = 0.0
        self.finished = False  # "TRAINING COMPLETE" (train_overnight) or "Done." (finetune_chat)
        self.is_chat = False   # log of a chat fine-tune (finetune_chat.py)

    def poll(self):
        try:
            st = os.stat(self.path)
        except OSError:
            return
        self.mtime = st.st_mtime
        if st.st_size < self.pos:  # log was replaced
            self.pos, self.vals = 0, []
        first_read = self.pos == 0
        with open(self.path, "rb") as f:
            # First poll: scan whole file for val losses; later: only new bytes
            f.seek(self.pos)
            new = f.read().decode("utf-8", "ignore")
            self.pos = f.tell()
        self.vals += [float(v) for v in VAL_RE.findall(new)]
        if "Base model:" in new:
            self.is_chat = True
        if "TRAINING COMPLETE" in new or re.search(r"^Done\.", new, re.M):
            self.finished = True
        if "homeostatic reset" in new and not first_read:  # only nap for new sleeps
            self.sleeping_until = time.time() + 8
        m = TQDM_RE.findall(new[-3000:])
        if m:
            ep, eps, pct, step, steps, eta, its, loss = m[-1]
            self.status = dict(ep=int(ep), eps=int(eps), step=int(step), steps=int(steps),
                               eta=eta.strip(), its=float(its), loss=float(loss))

    @property
    def stale(self):
        return not self.finished and self.mtime and time.time() - self.mtime > 120


# Fisherman's boat, drawn above the water line (the hull sits on the water)
BOAT = [
    r"    o     /",
    r"   /|\___/ ",
    r"   / \     ",
    r"\_________/",
]
LINE_COL = 11  # fishing line hangs from the pole tip, just right of the hull
SKY = len(BOAT) - 1  # rows above the water line
SCHOOL_SHAPES = [("><>", "<><"), ("><>", "<><"), ("><(((*>", "<*)))><"), ("><((o>", "<o))><"), (">=>", "<=<")]
SCHOOL_COLORS = ["\x1b[95m", "\x1b[96m", "\x1b[92m", "\x1b[33m", "\x1b[94m", "\x1b[91m", "\x1b[97m"]


def make_school(n=7):
    """Small background fish: [x, depth 0..1, direction, speed, (right, left), colour, wobble phase]."""
    return [[random.uniform(2, 60), random.uniform(0.1, 0.85), random.choice((-1, 1)),
             random.uniform(0.08, 0.35), random.choice(SCHOOL_SHAPES),
             random.choice(SCHOOL_COLORS), random.uniform(0, 6.28)] for _ in range(n)]


def bar(frac, width):
    filled = int(frac * width)
    return "#" * filled + "-" * (width - filled)


def main():
    os.system("")  # enable ANSI escapes on Windows consoles
    watch = LogWatcher(LOG)
    x, direction = 5.0, 1
    bubbles = []  # [x, y, char]
    weeds = None
    school = make_school()
    bite_until = 0.0
    t0 = time.time()
    last_poll = 0.0
    sys.stdout.write("\x1b[?25l\x1b[2J")
    try:
        while True:
            now = time.time()
            if now - last_poll > 1.0:
                # Without a log argument, follow whichever training log is newest,
                # e.g. hop from the finished training run to the chat fine-tune
                if len(sys.argv) < 2:
                    newest = newest_log()
                    if newest != watch.path and os.path.getmtime(newest) > watch.mtime + 1:
                        watch = LogWatcher(newest)
                watch.poll()
                last_poll = now

            cols, rows = shutil.get_terminal_size((80, 24))
            w = max(40, cols - 1)
            h = max(8, min(rows - 6 - SKY, 16))
            if weeds is None or len(weeds) != w:
                weeds = [random.random() < 0.12 for _ in range(w)]

            s = watch.status
            dead = watch.stale
            napping = now < watch.sleeping_until
            speed = 0.0 if (dead or napping) else (0.15 + (s["its"] / 15 if s else 0.2) * 0.6)

            # Swim
            x += direction * speed
            fish = FISH_R if direction > 0 else FISH_L
            if x >= w - len(fish) - 1:
                x, direction = w - len(fish) - 1, -1
            elif x <= 1:
                x, direction = 1, 1
            if not dead and random.random() < 0.005:
                direction *= -1
            fy = 1 if dead else int(h / 2 + math.sin((now - t0) * 1.3) * (h / 4))
            fx = int(x)

            # The school swims on its own, at its own depths and speeds
            for f in school:
                f[0] += f[2] * f[3]
                if f[0] >= w - len(f[4][0]) - 1:
                    f[0], f[2] = w - len(f[4][0]) - 1, -1
                elif f[0] <= 1:
                    f[0], f[2] = 1, 1
                elif random.random() < 0.003:
                    f[2] *= -1

            # Bubbles from the mouth
            if not dead and random.random() < 0.18:
                mouth = fx + len(fish) if direction > 0 else fx - 1
                bubbles.append([mouth, fy, random.choice(".oO")])
            for b in bubbles:
                b[1] -= 0.35
                b[0] += random.choice((-1, 0, 0, 1)) * 0.3
            bubbles = [b for b in bubbles if b[1] > 0 and 0 <= b[0] < w]

            # The boat drifts slowly back and forth across the surface
            span = w - len(BOAT[-1]) - 3
            bx = int(span * (0.5 + 0.45 * math.sin((now - t0) * 0.04)))
            line_x = bx + LINE_COL
            hook_y = int(h * 0.45 + math.sin((now - t0) * 0.7) * 1.2)

            # Scene: SKY rows above the surface, the surface row, then h rows of water
            H = SKY + 1 + h
            grid = [[" "] * w for _ in range(H)]
            colors = [[BLUE] * w for _ in range(H)]

            def put(r, c, ch, color):
                if 0 <= r < H and 0 <= c < w:
                    grid[r][c] = ch
                    colors[r][c] = color

            wave = int(now * 3)
            for c in range(w):
                put(SKY, c, "~" if (c + wave) % 6 else "-", CYAN)
            for b in bubbles:
                put(SKY + 1 + int(b[1]), int(b[0]), b[2], WHITE)
            sway = int(now * 2) % 2
            for i, has in enumerate(weeds):
                if has:
                    for k in range(1, 4):
                        put(H - k, i, "(" if (k + sway) % 2 else ")", GREEN)

            # Fishing line from the pole tip down to the hook, with a bobber
            for r in range(0, SKY + 1 + hook_y):
                put(r, line_x, "|", DIM)
            put(SKY, line_x, "o", RED)
            put(SKY + 1 + hook_y, line_x, "J", WHITE)

            # Boat and fisherman
            for r, row in enumerate(BOAT):
                for c, ch in enumerate(row):
                    if ch != " ":
                        put(r, bx + c, ch, "\x1b[33m" if r == len(BOAT) - 1 else WHITE)

            # Fish: the school first, then the training fish on top
            bite = False
            for f in school:
                shape = f[4][0] if f[2] > 0 else f[4][1]
                sy = max(0, min(h - 2, int(1 + f[1] * (h - 3) + math.sin((now - t0) * 0.9 + f[6]))))
                for i, ch in enumerate(shape):
                    put(SKY + 1 + sy, int(f[0]) + i, ch, f[5])
                if sy == hook_y and int(f[0]) - 1 <= line_x <= int(f[0]) + len(shape):
                    bite = True
            if dead:
                fish = FISH_DEAD_R if direction > 0 else FISH_DEAD_L
            for i, ch in enumerate(fish):
                put(SKY + 1 + fy, fx + i, ch, RED if dead else YELLOW)
            if fy == hook_y and fx - 1 <= line_x <= fx + len(fish):
                bite = True
            if napping and fy > 0:
                zx = fx + len(fish)
                for i, ch in enumerate("zzz"[: int(now * 2) % 4]):
                    put(SKY + fy, zx + i, ch, CYAN)

            # The fisherman gets excited when a fish swims past his hook
            if bite:
                bite_until = now + 1.5
            if now < bite_until:
                put(0, bx + 6, "!", YELLOW)
                put(SKY + 1 + hook_y, line_x, "?" if int(now * 6) % 2 else "J", YELLOW)

            out = ["\x1b[H"]
            for r in range(H):
                line, cur = [], None
                for c in range(w):
                    if colors[r][c] != cur:
                        cur = colors[r][c]
                        line.append(cur)
                    line.append(grid[r][c])
                out.append("".join(line) + RESET + "\x1b[K\n")
            out.append(DIM + "=" * w + RESET + "\n")

            # Status
            if s:
                ep_frac = s["step"] / max(1, s["steps"])
                all_frac = ((s["ep"] - 1) + ep_frac) / max(1, s["eps"])
                bw = max(10, w - 42)
                best = f"{min(watch.vals):.4f}" if watch.vals else "--"
                mood = (GREEN + "\x1b[1mAURA CHAT READY! Run chat.bat" if watch.finished and watch.is_chat
                        else GREEN + "training complete - chat fine-tune starts next" if watch.finished
                        else CYAN + "chat fine-tune" if watch.is_chat and not dead and not napping
                        else RED + "LOG STALE - training stopped?" if dead
                        else CYAN + "sleep cycle - napping" if napping
                        else GREEN + "swimming")
                out.append(f" Epoch {s['ep']:>3}/{s['eps']:<3} [{bar(ep_frac, bw)}] {ep_frac*100:5.1f}%  ETA {s['eta']}\x1b[K\n")
                out.append(f" Overall     [{bar(all_frac, bw)}] {all_frac*100:5.1f}%\x1b[K\n")
                out.append(f" loss {YELLOW}{s['loss']:.4f}{RESET}  ppl {math.exp(s['loss']):8.1f}  "
                           f"best val {YELLOW}{best}{RESET}  {s['its']:.1f} it/s  {mood}{RESET}\x1b[K\n")
            else:
                out.append(f" Waiting for training log: {LOG}\x1b[K\n\x1b[K\n\x1b[K\n")
            out.append(DIM + " Ctrl+C to quit" + RESET + "\x1b[K")
            sys.stdout.write("".join(out))
            sys.stdout.flush()
            time.sleep(0.08)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write(RESET + "\x1b[?25h\n")


if __name__ == "__main__":
    main()
