"""
Calibrates the engine's Elo setting against Stockfish's Elo limiter: for each point,
our engine at UCI_Elo=X plays Stockfish at UCI_Elo=X (or --sf-elo), both at a fixed
time per move, with fastchess over tests/openings.epd (each opening with both colours).
Prints the measured strength, setting + Elo difference, per point.

    python scripts/calibrate_elo.py --points 1400 1800 2200 2600 --games 150
    python scripts/calibrate_elo.py --points 2900 --sf-elo 3000      # full strength

Stockfish can't be limited below 1320, so lower settings are extrapolated.
"""

import argparse
import datetime
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_change import FASTCHESS, ROOT, resolve  # noqa: E402

sys.path.insert(0, ROOT)
from python.semantic_analysis import STOCKFISH_PATH  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default="HEAD", help="engine .exe or git ref")
    ap.add_argument("--points", type=int, nargs="+", default=[1400, 1800, 2200, 2600])
    ap.add_argument("--sf-elo", type=int, default=0, help="Stockfish UCI_Elo (default: same as the point)")
    ap.add_argument("--games", type=int, default=150)
    ap.add_argument("--st", type=float, default=0.3, help="seconds per move, both engines")
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()

    exe = resolve(args.engine, "calib")
    out_dir = os.path.join(ROOT, "data", "calibration")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results = []
    for elo in args.points:
        sf_elo = args.sf_elo or max(1320, min(3190, elo))
        ours = ["option.UCI_LimitStrength=false"] if elo >= 2900 else \
            ["option.UCI_LimitStrength=true", f"option.UCI_Elo={elo}"]
        cmd = [FASTCHESS[0],
               "-engine", f"cmd={exe}", "name=ours", f"dir={ROOT}", *ours,
               "-engine", f"cmd={STOCKFISH_PATH}", "name=sf", "option.UCI_LimitStrength=true",
               f"option.UCI_Elo={sf_elo}", "option.Threads=1",
               "-each", "proto=uci", f"st={args.st}", "timemargin=200",
               "-openings", f"file={os.path.join(ROOT, 'tests', 'openings.epd')}", "format=epd", "order=random",
               "-repeat", "-games", "2", "-rounds", str(args.games // 2), "-concurrency", str(args.concurrency),
               "-recover", "-pgnout", f"file={os.path.join(out_dir, f'{stamp}-{elo}.pgn')}"]
        text = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).stdout
        with open(os.path.join(out_dir, f"{stamp}-{elo}.log"), "w", encoding="utf-8") as f:
            f.write(text)
        m = re.findall(r"(?<!n)Elo:\s*([-\d.]+|inf|-inf)\s*\+/-\s*([\d.]+|nan)", text)
        g = re.findall(r"Games: (\d+), Wins: (\d+), Losses: (\d+), Draws: (\d+)", text)
        if not m or not g:
            print(f"{elo}: no result (see data/calibration/{stamp}-{elo}.log)", flush=True)
            continue
        diff, err = float(m[-1][0]), m[-1][1]
        n, w, l, d = g[-1]
        line = (f"setting {elo:4d} vs SF {sf_elo}: +{w} ={d} -{l} ({n} games)  "
                f"diff {diff:+.0f} +/- {err}  => ~{sf_elo + diff:.0f}")
        results.append(line)
        print(line, flush=True)
    print("\n" + "\n".join(results))


if __name__ == "__main__":
    main()
