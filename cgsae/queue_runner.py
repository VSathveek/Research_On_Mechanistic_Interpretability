"""
Push queued Kaggle kernels as GPU slots free up, and exit once everything is done.

Kaggle allows 2 concurrent batch GPU sessions. This walks a queue of already-built kernel
directories, pushing the next one whenever a slot opens, and returns only when every kernel
in `watch` and `queue` has reached a terminal state.

Usage:
    python queue_runner.py --watch <id> <id> --queue <dir> <dir> ...
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

USER = os.environ.get("KAGGLE_USER", "varanasilvsssathveek")
TERMINAL = ("COMPLETE", "ERROR", "CANCEL")
MAX_SLOTS = 2


def status(kid: str) -> str:
    """Parse only the authoritative status line.

    Matching bare words against the whole output is wrong: a transient network failure
    prints "...Error(...)" and was being read as a kernel ERROR, which falsely retired two
    still-healthy runs. Anchor on the KernelWorkerStatus token instead, and treat anything
    unparseable as UNKNOWN so the caller keeps waiting rather than concluding.
    """
    r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "status", f"{USER}/{kid}"],
                       capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    m = re.search(r"KernelWorkerStatus\.([A-Z_]+)", out)
    return m.group(1) if m else "UNKNOWN"


def push(kdir: str) -> bool:
    r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", kdir],
                       capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    ok = "successfully pushed" in out.lower()
    if not ok:
        print(f"    push failed: {out.strip()[:200]}", flush=True)
    return ok


def kid_of(kdir: str) -> str:
    meta = json.load(open(os.path.join(kdir, "kernel-metadata.json"), encoding="utf-8"))
    return meta["id"].split("/")[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", nargs="*", default=[], help="kernel slugs already running")
    ap.add_argument("--queue", nargs="*", default=[], help="kernel dirs to push in order")
    ap.add_argument("--poll", type=int, default=90)
    a = ap.parse_args()

    pending = list(a.queue)
    active = list(a.watch)
    done: dict[str, str] = {}

    print(f"watching {len(active)}, queued {len(pending)}", flush=True)
    while pending or active:
        for kid in list(active):
            st = status(kid)
            if st in TERMINAL:
                done[kid] = st
                active.remove(kid)
                print(f"[{time.strftime('%H:%M')}] {kid} -> {st}", flush=True)

        while pending and len(active) < MAX_SLOTS:
            kdir = pending.pop(0)
            kid = kid_of(kdir)
            print(f"[{time.strftime('%H:%M')}] pushing {kid}", flush=True)
            if push(kdir):
                active.append(kid)
            else:
                pending.insert(0, kdir)   # slot probably still busy; retry next cycle
                break

        if pending or active:
            time.sleep(a.poll)

    print("\n=== ALL RUNS FINISHED ===", flush=True)
    for kid, st in done.items():
        print(f"  {st:9s} {kid}", flush=True)
    bad = [k for k, v in done.items() if v != "COMPLETE"]
    print(f"\n{len(done) - len(bad)} complete, {len(bad)} failed", flush=True)
    if bad:
        print("failed: " + ", ".join(bad), flush=True)


if __name__ == "__main__":
    main()
