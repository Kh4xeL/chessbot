"""
SPSA tuning of the engine's search parameters (the tune:: options in src/main.cpp).

Each iteration perturbs every parameter by +-c_k (random signs), plays a batch of game
pairs between the "plus" and "minus" settings with fastchess (fixed nodes, both colours
per opening), and moves each parameter towards the side that scored better:

    theta_i += a_k / c_k * (wins - losses) * delta_i

with the usual schedules c_k = c / k^0.101, a_k = a / (A + k)^0.602. As in OpenBench,
c is set from the final step c_end (default: 1/20 of the option's range) and a from
r_end, the final learning rate. State is saved after every iteration, so a run can be
stopped and resumed with the same command.

    python scripts/spsa.py --engine cmake-build-release/engine.exe --iterations 2500
    python scripts/spsa.py --engine ... --params rfp_margin,lmr_div   # tune a subset

The tuned values are printed at the end (and in data/spsa/state.json) to paste into
the TUNE(...) defaults; check them against the old defaults with test_change.py.
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_change import FASTCHESS, ROOT  # noqa: E402

ALPHA, GAMMA = 0.602, 0.101


def engine_params(exe):
    """Read the tunable spin options (lowercase names) from the engine's `uci` reply."""
    out = subprocess.run([exe], input="uci\nquit\n", capture_output=True, text=True, cwd=ROOT).stdout
    params = {}
    for m in re.finditer(r"option name (\S+) type spin default (-?\d+) min (-?\d+) max (-?\d+)", out):
        name, d, lo, hi = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        if name.islower():
            params[name] = {"value": float(d), "min": lo, "max": hi, "default": d}
    return params


def play(exe, plus, minus, pairs, nodes, concurrency):
    """Play `pairs` game pairs plus vs minus; return (wins, losses, draws) for plus."""
    def opts(vals):
        return [f"option.{k}={int(round(v))}" for k, v in vals.items()]
    cmd = [FASTCHESS[0],
           "-engine", f"cmd={exe}", "name=plus", f"dir={ROOT}", *opts(plus),
           "-engine", f"cmd={exe}", "name=minus", f"dir={ROOT}", *opts(minus),
           "-each", "proto=uci", "tc=inf", f"nodes={nodes}",
           "-openings", f"file={os.path.join(ROOT, 'tests', 'openings.epd')}", "format=epd", "order=random",
           "-repeat", "-games", "2", "-rounds", str(pairs), "-concurrency", str(concurrency),
           "-draw", "movenumber=40", "movecount=8", "score=10",
           "-resign", "movecount=3", "score=1000", "-recover"]
    text = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).stdout
    m = re.findall(r"Games: \d+, Wins: (\d+), Losses: (\d+), Draws: (\d+)", text)
    if not m:
        raise RuntimeError("no result from fastchess:\n" + text[-2000:])
    return tuple(int(x) for x in m[-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True, help="engine .exe exposing the tune options")
    ap.add_argument("--params", default="", help="comma-separated subset to tune (default: all)")
    ap.add_argument("--iterations", type=int, default=2500)
    ap.add_argument("--pairs", type=int, default=12, help="game pairs per iteration")
    ap.add_argument("--nodes", type=int, default=20000)
    ap.add_argument("--concurrency", type=int, default=6, help="games at once per batch")
    ap.add_argument("--workers", type=int, default=1, help="batches played at the same time")
    ap.add_argument("--r-end", type=float, default=0.002)
    ap.add_argument("--c-frac", type=float, default=0.05, help="c_end as a fraction of each option's range")
    ap.add_argument("--state", default=os.path.join(ROOT, "data", "spsa", "state.json"))
    args = ap.parse_args()

    exe = os.path.abspath(args.engine)
    os.makedirs(os.path.dirname(args.state), exist_ok=True)
    if os.path.exists(args.state):
        state = json.load(open(args.state))
        print(f"resuming at iteration {state['k']}")
    else:
        params = engine_params(exe)
        if args.params:
            keep = args.params.split(",")
            params = {k: v for k, v in params.items() if k in keep}
        state = {"k": 0, "params": params, "history": []}
    params = state["params"]
    if not params:
        sys.exit("no tunable options found")

    n, big_a = args.iterations, 0.1 * args.iterations
    for p in params.values():
        # at least 1: integer options only see a perturbation once it survives rounding
        c_end = max(1.0, args.c_frac * (p["max"] - p["min"]))
        p["c"] = c_end * n ** GAMMA
        p["a"] = args.r_end * c_end ** 2 * (big_a + n) ** ALPHA

    # Several batches can run at once (--workers), each from the current parameters, as
    # on a distributed tuner; finished batches are applied one at a time under a lock.
    lock = threading.Lock()
    started = [state["k"]]

    def worker():
        while True:
            with lock:
                if started[0] >= n:
                    return
                started[0] += 1
                k = started[0]
                plus, minus, delta = {}, {}, {}
                for name, p in params.items():
                    ck = p["c"] / k ** GAMMA
                    delta[name] = random.choice((-1, 1))
                    plus[name] = min(p["max"], max(p["min"], p["value"] + ck * delta[name]))
                    minus[name] = min(p["max"], max(p["min"], p["value"] - ck * delta[name]))
            w, l, d = play(exe, plus, minus, args.pairs, args.nodes, args.concurrency)
            with lock:
                for name, p in params.items():
                    ak, ck = p["a"] / (big_a + k) ** ALPHA, p["c"] / k ** GAMMA
                    p["value"] = min(p["max"], max(p["min"], p["value"] + ak / ck * (w - l) * delta[name]))
                state["k"] += 1
                done = state["k"]
                state["history"].append({"k": done, "w": w, "l": l, "d": d,
                                         "values": {name: round(p["value"], 2) for name, p in params.items()}})
                tmp = args.state + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(state, f, indent=1)
                os.replace(tmp, args.state)
                if done % 10 == 0 or done == n:
                    vals = "  ".join(f"{name}={p['value']:.1f}" for name, p in params.items())
                    print(f"iter {done}/{n}  +{w} -{l} ={d}  {vals}", flush=True)

    threads = [threading.Thread(target=worker) for _ in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("\nTuned values (default -> tuned):")
    for name, p in params.items():
        print(f"  {name}: {p['default']} -> {int(round(p['value']))}")


if __name__ == "__main__":
    main()
