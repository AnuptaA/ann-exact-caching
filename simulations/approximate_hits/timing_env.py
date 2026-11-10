#!/usr/bin/env python3
"""
Quiet-machine guard for timing experiments (context doc 9d).

A timing result is only trustworthy if nothing else was competing for the CPU,
memory bandwidth or the disk while it ran. This module makes that checkable
instead of assumed:
  1. before a timing run, measure for a few seconds how much CPU and disk
     traffic other processes cause (everything except the timing process
     itself) and refuse to start, or wait, unless the machine is quiet;
  2. during the run, sample the same quantities every few seconds in a
     background thread (a /proc read; negligible cost);
  3. store the summary (and who was busy at the start) in the result JSON under
     "environment", with a verdict "quiet" or "contended".

"Other" CPU = busy time of all CPUs (/proc/stat) minus the timing process's own
(/proc/self/stat, all its threads). "Other" disk traffic = bytes moved on the
block devices (/proc/diskstats) minus the process's own (/proc/self/io).
Stdlib only (also runs in .venv-ann).

Use in a script:
    guard = QuietGuard(require=True, wait_min=0)
    guard.start()          # raises SystemExit if the machine is busy
    ... timed work ...
    guard.stop()
    result["environment"] = guard.report()

Command line (stand-alone check, exit status 1 if busy):
    python simulations/approximate_hits/timing_env.py check [--wait-min 60]
The sampler runs as a separate process (`timing_env.py monitor`): a thread inside the
timing process is starved while diskannpy holds the GIL (e.g. loading an index).
"""

import os
import re
import sys
import json
import time
import socket
import argparse
import platform
import subprocess

CLK = os.sysconf("SC_CLK_TCK")
DISK = re.compile(r"(nvme\d+n\d+|sd[a-z]+|vd[a-z]+|xvd[a-z]+)$")


def _busy_jiffies():
    """Busy CPU time over all CPUs (user + nice + system + irq + softirq + steal)."""
    with open("/proc/stat") as f:
        v = [int(x) for x in f.readline().split()[1:]]
    return v[0] + v[1] + v[2] + v[5] + v[6] + (v[7] if len(v) > 7 else 0)


def _proc_jiffies(pid="self"):
    """utime + stime of a process (all its threads), or None if it is gone."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            s = f.read()
        rest = s[s.rindex(")") + 2:].split()
        return int(rest[11]) + int(rest[12])
    except (OSError, ValueError, IndexError):
        return None


def _disk_bytes():
    r = w = 0
    with open("/proc/diskstats") as f:
        for line in f:
            p = line.split()
            if len(p) > 9 and DISK.match(p[2]):
                r += int(p[5]) * 512
                w += int(p[9]) * 512
    return r, w


def _io_of(pid):
    d = {}
    try:
        with open(f"/proc/{pid}/io") as f:
            for line in f:
                k, v = line.split(": ")
                d[k] = int(v)
    except OSError:
        pass
    return d.get("read_bytes", 0) + d.get("write_bytes", 0)


def snapshot(pid="self", also=None):
    """Machine-wide busy CPU and disk bytes, and the part caused by `pid` (the timing
    process; all its threads) plus, if given, `also` (the monitor process itself)."""
    r, w = _disk_bytes()
    own = _proc_jiffies(pid) or 0
    own_io = _io_of(pid)
    if also is not None:
        own += _proc_jiffies(also) or 0
        own_io += _io_of(also)
    return {"t": time.time(), "busy": _busy_jiffies(), "own": own, "disk": r + w, "own_io": own_io}


def other_load(a, b):
    """(cores of CPU used by other processes, MB/s of disk traffic caused by others)
    between two snapshots."""
    dt = max(b["t"] - a["t"], 1e-6)
    cores = max(((b["busy"] - a["busy"]) - (b["own"] - a["own"])) / CLK / dt, 0.0)
    mbps = max(((b["disk"] - a["disk"]) - (b["own_io"] - a["own_io"])) / dt / 1e6, 0.0)
    return cores, mbps


def top_processes(seconds, n=6):
    """The processes using the most CPU over a window (excluding this one):
    list of (pid, cores, command)."""
    me = os.getpid()
    pids = [p for p in os.listdir("/proc") if p.isdigit() and int(p) != me]
    a = {p: _proc_jiffies(p) for p in pids}
    t0 = time.time()
    time.sleep(seconds)
    dt = time.time() - t0
    use = []
    for p, j0 in a.items():
        j1 = _proc_jiffies(p)
        if j0 is not None and j1 is not None and j1 > j0:
            use.append((p, (j1 - j0) / CLK / dt))
    use.sort(key=lambda x: -x[1])
    out = []
    for p, c in use[:n]:
        try:
            with open(f"/proc/{p}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode(errors="replace").strip()[:110]
        except OSError:
            cmd = "?"
        out.append({"pid": int(p), "cores": round(c, 2), "cmd": cmd})
    return out


def check_quiet(seconds=10, max_other_cores=1.0, max_other_io_mbps=50.0):
    """Measure other processes' load for `seconds`. ok = both within the limits."""
    a = snapshot()
    top = top_processes(seconds)
    b = snapshot()
    cores, mbps = other_load(a, b)
    return {"ok": bool(cores <= max_other_cores and mbps <= max_other_io_mbps),
            "other_cores": round(cores, 2), "other_io_mbps": round(mbps, 1),
            "limits": {"other_cores": max_other_cores, "other_io_mbps": max_other_io_mbps},
            "window_s": seconds, "busiest": top}


def _cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def _pct(xs, q):
    xs = sorted(xs)
    return xs[min(int(q * len(xs)), len(xs) - 1)] if xs else 0.0


class QuietGuard:
    """Refuse (or wait) at start unless the machine is quiet; sample the load of
    other processes during the run; summarize in report()."""

    def __init__(self, require=True, wait_min=0, max_other_cores=1.0, max_other_io_mbps=50.0,
                 check_s=10, sample_s=2.0):
        self.require, self.wait_min = require, wait_min
        self.max_cores, self.max_io = max_other_cores, max_other_io_mbps
        self.check_s, self.sample_s = check_s, sample_s
        self.samples, self.start_check = [], None
        self._proc, self._t0, self._path = None, None, None

    def start(self):
        deadline = time.time() + self.wait_min * 60
        while True:
            chk = check_quiet(self.check_s, self.max_cores, self.max_io)
            if chk["ok"] or not self.require:
                break
            if time.time() >= deadline:
                busy = "; ".join(f"{p['cores']} cores: {p['cmd'][:60]}" for p in chk["busiest"][:4])
                raise SystemExit(f"machine not quiet: other processes use {chk['other_cores']} cores and "
                                 f"{chk['other_io_mbps']} MB/s of disk (limits {self.max_cores} / "
                                 f"{self.max_io}); busiest: {busy}. Wait for the queues to drain, use "
                                 f"--wait-quiet MIN, or --allow-busy to record the contention instead.")
            time.sleep(60)
        self.start_check = chk
        self.load_start = os.getloadavg()
        self._t0 = time.time()
        # the sampler is a separate process, so it keeps sampling while this process holds the GIL
        # (diskannpy does while loading an index) and does not disturb the timed threads
        self._path = f"/tmp/timing_env_{os.getpid()}.samples"
        self._proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "monitor",
                                       "--pid", str(os.getpid()), "--every", str(self.sample_s),
                                       "--out", self._path], stdin=subprocess.DEVNULL)
        return chk

    def stop(self):
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
        self.samples = []
        try:
            with open(self._path) as f:
                for line in f:
                    t, c, m = line.split()
                    self.samples.append((float(t), float(c), float(m)))
            os.remove(self._path)
        except (OSError, ValueError):
            pass
        self.load_end = os.getloadavg()
        self.duration_s = time.time() - (self._t0 or time.time())

    def report(self):
        cores = [s[1] for s in self.samples]
        io = [s[2] for s in self.samples]
        n = max(len(cores), 1)
        above = sum(1 for c, m in zip(cores, io) if c > self.max_cores or m > self.max_io)
        quiet = bool(self.start_check and self.start_check["ok"] and _pct(cores, 0.95) <= self.max_cores
                     and max(io, default=0.0) <= self.max_io)
        return {"verdict": "quiet" if quiet else "contended",
                "limits": {"other_cores": self.max_cores, "other_io_mbps": self.max_io},
                "start_check": self.start_check,
                "during": {"samples": len(cores), "every_s": self.sample_s,
                           "other_cores_mean": round(sum(cores) / n, 2),
                           "other_cores_p95": round(_pct(cores, 0.95), 2),
                           "other_cores_max": round(max(cores, default=0.0), 2),
                           "other_io_mbps_mean": round(sum(io) / n, 1),
                           "other_io_mbps_max": round(max(io, default=0.0), 1),
                           "share_samples_over_limit": round(above / n, 3)},
                "loadavg_start": [round(x, 2) for x in getattr(self, "load_start", (0, 0, 0))],
                "loadavg_end": [round(x, 2) for x in getattr(self, "load_end", (0, 0, 0))],
                "duration_s": round(getattr(self, "duration_s", 0.0), 1),
                "host": {"name": socket.gethostname(), "cpu": _cpu_model(), "cores": os.cpu_count(),
                         "kernel": platform.release()},
                "finished": time.strftime("%Y-%m-%d %H:%M:%S")}


def add_args(ap):
    """The two flags every timing script takes."""
    ap.add_argument("--allow-busy", action="store_true",
                    help="run even if other processes are busy; the contention is recorded "
                         "(environment.verdict = contended)")
    ap.add_argument("--wait-quiet", type=float, default=0.0,
                    help="minutes to wait for the machine to become quiet before refusing")
    ap.add_argument("--max-other-cores", type=float, default=1.0,
                    help="quiet = other processes use at most this many CPU cores")


def make_guard(args):
    return QuietGuard(require=not args.allow_busy, wait_min=args.wait_quiet,
                      max_other_cores=args.max_other_cores)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("cmd", choices=["check", "monitor", "audit"])
    ap.add_argument("--pid", type=int, default=None, help="monitor: the timing process")
    ap.add_argument("--every", type=float, default=2.0, help="monitor: seconds between samples")
    ap.add_argument("--out", default=None, help="monitor: file the samples are appended to")
    ap.add_argument("--seconds", type=int, default=10)
    ap.add_argument("--wait-min", type=float, default=0.0)
    ap.add_argument("--max-other-cores", type=float, default=1.0)
    ap.add_argument("--max-other-io-mbps", type=float, default=50.0)
    args = ap.parse_args()
    if args.cmd == "monitor":
        # one line per sample: seconds since start, cores used by others, MB/s of disk by others;
        # ends when the timing process is gone or on SIGTERM
        me, t0 = os.getpid(), time.time()
        prev = snapshot(args.pid, also=me)
        with open(args.out, "w", buffering=1) as f:
            while os.path.exists(f"/proc/{args.pid}"):
                time.sleep(args.every)
                cur = snapshot(args.pid, also=me)
                cores, mbps = other_load(prev, cur)
                f.write(f"{cur['t'] - t0:.1f} {cores:.2f} {mbps:.1f}\n")
                prev = cur
        return
    if args.cmd == "audit":
        # verdict of every timing result under results/ (python timing_env.py audit)
        import glob
        files = sorted(glob.glob("simulations/approximate_hits/results/**/e2e_*.json", recursive=True)
                       + glob.glob("simulations/approximate_hits/results/**/variants_*.json", recursive=True))
        bad = 0
        for fn in files:
            e = json.load(open(fn)).get("environment")
            if e is None:
                v = "no record (ran before the guard)"
            else:
                v = f"{e['verdict']} (others: mean {e['during']['other_cores_mean']}, p95 {e['during']['other_cores_p95']} cores)"
            bad += e is None or e["verdict"] != "quiet"
            print(f"{v:70s} {fn.replace('simulations/approximate_hits/results/', '')}")
        print(f"{len(files) - bad} of {len(files)} timing results have a 'quiet' record")
        return
    deadline = time.time() + args.wait_min * 60
    while True:
        chk = check_quiet(args.seconds, args.max_other_cores, args.max_other_io_mbps)
        if chk["ok"] or time.time() >= deadline:
            break
        time.sleep(60)
    print(json.dumps(chk, indent=1))
    sys.exit(0 if chk["ok"] else 1)


if __name__ == "__main__":
    main()
