"""
Runs self-play data generation (engine --datagen) in parallel processes.

Each process writes to its own part directory under data/selfplay/<gen>/, with its own
random seed, and appends, so a stopped run can simply be started again. Progress (total
positions) is printed every minute.

    python scripts/datagen.py --gen gen1 --procs 12 --positions 30000000
    python training/train_nnue.py --data data/selfplay/gen1/part* ...
"""

import argparse
import glob
import os
import random
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def count(gen_dir):
    return sum(os.path.getsize(p) for p in glob.glob(os.path.join(gen_dir, "part*", "stm.u8")))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", required=True, help="output name, data/selfplay/<gen>")
    ap.add_argument("--engine", default=os.path.join(ROOT, "cmake-build-release", "engine.exe"))
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--nodes", type=int, default=5000)
    ap.add_argument("--positions", type=int, default=30_000_000, help="stop once this many are on disk")
    args = ap.parse_args()

    args.engine = os.path.abspath(args.engine)
    gen_dir = os.path.join(ROOT, "data", "selfplay", args.gen)
    procs = []
    for i in range(args.procs):
        part = os.path.join(gen_dir, f"part{i:02d}")
        os.makedirs(part, exist_ok=True)
        seed = random.randrange(1, 2**31)
        procs.append(subprocess.Popen([args.engine, "--datagen", part, "100000000", str(args.nodes), str(seed)],
                                      cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    print(f"{args.procs} processes, {args.nodes} nodes/move -> {gen_dir}", flush=True)
    start, n0 = time.time(), count(gen_dir)
    try:
        while True:
            time.sleep(60)
            n = count(gen_dir)
            rate = (n - n0) / (time.time() - start)
            print(f"{n:,} positions  {rate:,.0f}/s  eta {(args.positions - n) / max(rate, 1) / 3600:.1f} h", flush=True)
            if n >= args.positions or all(p.poll() is not None for p in procs):
                break
    finally:
        for p in procs:
            p.terminate()
    print(f"done: {count(gen_dir):,} positions")


if __name__ == "__main__":
    sys.exit(main())
