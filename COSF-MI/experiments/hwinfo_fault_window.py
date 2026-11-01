"""
GPU readings from an HWiNFO CSV log in the window before a fault, against the run's normal readings (2026-11-01).

Window: the --minutes (default 10) before --fault-time. "Normal": all rows of the log under training load (GPU power ≥ --load-watts,
default 400 W) earlier than one hour before the fault. Per sensor: min / median / max in the window vs median / p5-p95 of normal, the
last 6 rows before the fault, and any PCIe error counter that is non-zero anywhere. Output is Markdown (append it to CRASHES.md).

HWiNFO quirks handled: dates as D.M.YYYY (one-digit day), a repeated header and a "System: …" row at the end, latin-1 encoding, values
"Yes"/"No" for limiters, and the file still being written.

Usage: python experiments/hwinfo_fault_window.py LOG.csv --fault-time "2026-11-01 14:23:05" [--minutes 10]
"""
import argparse
import csv
import datetime as dt
import statistics as st

SENSORS = [  # (column prefix, short name)
    ("GPU Temperature", "GPU temp °C"), ("GPU Hot Spot", "hotspot °C"), ("GPU Memory Junction", "mem junction °C"),
    ("GPU Power [W]", "GPU power W"), ("GPU 12VHPWR Power", "12VHPWR W"), ("GPU 12VHPWR Voltage", "12VHPWR V"),
    ("GPU PCIe +12V Input Voltage", "slot +12V V"), ("+12V [V]", "board +12V V"), ("GPU Core Voltage", "core V"),
    ("GPU Clock [MHz]", "core MHz"), ("GPU Effective Clock", "effective MHz"), ("GPU Memory Clock [MHz]", "mem MHz"),
    ("GPU Core Load", "core load %"), ("GPU Bus Load", "bus load %"),
]
LIMITERS = ["Performance Limit - Power", "Performance Limit - Thermal", "Performance Limit - Reliability Voltage",
            "Performance Limit - Max Operating Voltage"]
ERRORS = ["Receiver Errors", "LCRC Error Count", "Correctable Error Count", "Non-Fatal Error Count", "Fatal Error Count", "PCIe Lane"]


def num(x):
    try:
        return float(x)
    except ValueError:
        return None


def parse_time(d, t):
    day, mon, year = (int(x) for x in d.split("."))
    h, m, s = t.split(":")
    return dt.datetime(year, mon, day, int(h), int(m), int(float(s)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log")
    p.add_argument("--fault-time", required=True, help='local time, "YYYY-MM-DD HH:MM:SS"')
    p.add_argument("--minutes", type=float, default=10)
    p.add_argument("--load-watts", type=float, default=400)
    args = p.parse_args()
    fault = dt.datetime.strptime(args.fault_time, "%Y-%m-%d %H:%M:%S")
    rows = list(csv.reader(open(args.log, encoding="latin-1")))
    head = [h.strip('"') for h in rows[0]]
    data = []
    for r in rows[1:]:
        if len(r) < len(head) - 2 or not r[0][:1].isdigit():
            continue
        try:
            data.append((parse_time(r[0], r[1]), r))
        except ValueError:
            continue

    def col(prefix):
        return next((i for i, h in enumerate(head) if h.startswith(prefix)), None)
    pw = col("GPU Power [W]")
    win = [(t, r) for t, r in data if fault - dt.timedelta(minutes=args.minutes) <= t <= fault]
    normal = [(t, r) for t, r in data if t < fault - dt.timedelta(hours=1) and (num(r[pw]) or 0) >= args.load_watts]
    print(f"### HWiNFO readings before the fault at {fault:%Y-%m-%d %H:%M:%S} (`{args.log.split('/')[-1]}`)\n")
    print(f"Log covers {data[0][0]:%m-%d %H:%M:%S} .. {data[-1][0]:%m-%d %H:%M:%S} ({len(data)} rows). Window: last {args.minutes:g} min "
          f"({len(win)} rows). Normal: {len(normal)} rows at ≥ {args.load_watts:g} W, earlier than 1 h before the fault.\n")
    if not win:
        print("**No HWiNFO rows in the window** (logging stopped, or the clock/file doesn't cover the fault).\n")
        return
    print("| sensor | window min | window median | window max | normal median | normal p5-p95 |\n|---|---|---|---|---|---|")
    for prefix, name in SENSORS:
        i = col(prefix)
        if i is None:
            continue
        w = sorted(x for x in (num(r[i]) for _, r in win) if x is not None)
        n = sorted(x for x in (num(r[i]) for _, r in normal) if x is not None)
        if not w:
            continue
        ns = f"{st.median(n):.2f} | {n[len(n) // 20]:.2f}-{n[19 * len(n) // 20]:.2f}" if n else "- | -"
        print(f"| {name} | {w[0]:.2f} | {st.median(w):.2f} | {w[-1]:.2f} | {ns} |")
    print("\n**Limiters active in the window (share of rows):** " + ", ".join(
        f"{lim.split(' - ')[1]} {sum(r[col(lim)].strip().lower() in ('yes', '1') for _, r in win) / len(win):.0%}"
        for lim in LIMITERS if col(lim) is not None))
    errs = [(h, i) for i, h in enumerate(head) if any(h.startswith(e) for e in ERRORS)]
    nz = {h: max(num(r[i]) or 0 for _, r in data) for h, i in errs}
    nz = {h: v for h, v in nz.items() if v}
    print(f"\n**PCIe error counters ({len(errs)}), max over the whole log:** " + (", ".join(f"{h} {v:g}" for h, v in nz.items()) if nz else "all 0"))
    last = win[-6:]
    keys = [(p, n) for p, n in SENSORS if n in ("hotspot °C", "mem junction °C", "GPU power W", "12VHPWR V", "core MHz", "core load %")]
    print("\n**Last rows before the fault:**\n\n| time | " + " | ".join(n for _, n in keys) + " |\n|---|" + "---|" * len(keys))
    for t, r in last:
        print(f"| {t:%H:%M:%S} | " + " | ".join(str(r[col(p)]) if col(p) is not None else "-" for p, _ in keys) + " |")
    print()


if __name__ == "__main__":
    main()
