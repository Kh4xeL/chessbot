"""
SPRT test: does a change gain Elo? Plays a new engine build against a base build
with fastchess at a fixed node count over tests/openings.epd (each opening with both
colours), stops as soon as the sequential test decides, and logs the result to
data/tests.csv.

Engines can be given as an .exe path or as a git ref; refs are built in a temporary
worktree (data/sprt/<ref>) with the same CMake setup. Each engine runs with a working
directory whose assets/ it loads, so networks can be compared too (--new-dir/--base-dir).

    python scripts/test_change.py --new my-branch                 # branch vs main
    python scripts/test_change.py --new cmake-build-release/engine.exe --base main --name lmr-tweak
    python scripts/test_change.py --new HEAD --base main --elo0 -5 --elo1 0   # non-regression test

Bounds are in normalized Elo: [0, 5] asks "is it better?" (gainer); [-5, 0] asks
"is it no worse?" (simplifications, refactors).
"""

import argparse
import csv
import datetime
import glob
import os
import re
import shutil
import subprocess
import sys
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FASTCHESS = [p for p in glob.glob(os.path.join(ROOT, "tools", "fastchess", "**", "fastchess*"), recursive=True)
             if os.path.isfile(p)]
EXE = "engine.exe" if os.name == "nt" else "engine"


def toolchain_env():
    """Put CMake/Ninja/MinGW from a CLion install on PATH when they aren't already."""
    env = dict(os.environ)
    if shutil.which("cmake") and shutil.which("ninja"):
        return env, "cmake"
    for clion in sorted(glob.glob(r"C:\Program Files\JetBrains\CLion*\bin"), reverse=True):
        paths = [os.path.join(clion, "mingw", "bin"), os.path.join(clion, "ninja", "win", "x64")]
        env["PATH"] = os.pathsep.join(paths + [env.get("PATH", "")])
        return env, os.path.join(clion, "cmake", "win", "x64", "bin", "cmake.exe")
    return env, "cmake"


def resolve(spec, label):
    """Return an engine executable for an .exe path or a git ref (built in a worktree)."""
    if os.path.isfile(spec):
        return os.path.abspath(spec)
    sha = subprocess.check_output(["git", "rev-parse", "--short", spec], cwd=ROOT, text=True).strip()
    tree = os.path.join(ROOT, "data", "sprt", f"{label}-{sha}")
    exe = os.path.join(tree, "build", EXE)
    if os.path.isfile(exe):
        return exe
    if not os.path.isdir(tree):
        subprocess.check_call(["git", "worktree", "add", "--detach", tree, sha], cwd=ROOT)
    env, cmake = toolchain_env()
    gen = ["-G", "Ninja"] if shutil.which("ninja", path=env["PATH"]) else []
    subprocess.check_call([cmake, "-S", tree, "-B", os.path.join(tree, "build"), "-DCMAKE_BUILD_TYPE=Release", *gen],
                          env=env, stdout=subprocess.DEVNULL)
    subprocess.check_call([cmake, "--build", os.path.join(tree, "build")], env=env, stdout=subprocess.DEVNULL)
    return exe


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", required=True, help="engine .exe or git ref")
    ap.add_argument("--base", default="main", help="engine .exe or git ref (default: main)")
    ap.add_argument("--new-dir", default=ROOT, help="working dir of the new engine (its assets/)")
    ap.add_argument("--base-dir", default=ROOT, help="working dir of the base engine")
    ap.add_argument("--new-options", default="", help='UCI options for the new engine, e.g. "Threads=1 MultiPV=1"')
    ap.add_argument("--base-options", default="", help='UCI options for the base engine, e.g. "UCI_LimitStrength=true UCI_Elo=2600"')
    ap.add_argument("--name", default="", help="label for the log")
    ap.add_argument("--nodes", type=int, default=20000, help="nodes per move (fixed-node games)")
    ap.add_argument("--tc", default="", help="time control instead of nodes, e.g. 8+0.08")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--elo0", type=float, default=0.0)
    ap.add_argument("--elo1", type=float, default=5.0)
    ap.add_argument("--max-games", type=int, default=30000)
    args = ap.parse_args()

    if not FASTCHESS:
        sys.exit("fastchess not found under tools/fastchess (download it from github.com/Disservin/fastchess)")
    new_exe = resolve(args.new, "new")
    base_exe = resolve(args.base, "base")
    name = args.name or f"{args.new} vs {args.base}"
    os.makedirs(os.path.join(ROOT, "data", "sprt"), exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    pgn = os.path.join(ROOT, "data", "sprt", f"{stamp}.pgn")

    limit = [f"tc={args.tc}"] if args.tc else ["tc=inf", f"nodes={args.nodes}"]
    def opts(spec):
        return [f"option.{kv}" for kv in spec.split()]

    cmd = [FASTCHESS[0],
           "-engine", f"cmd={new_exe}", "name=new", f"dir={os.path.abspath(args.new_dir)}", *opts(args.new_options),
           "-engine", f"cmd={base_exe}", "name=base", f"dir={os.path.abspath(args.base_dir)}", *opts(args.base_options),
           "-each", "proto=uci", *limit,
           "-openings", f"file={os.path.join(ROOT, 'tests', 'openings.epd')}", "format=epd", "order=random",
           "-repeat", "-games", "2", "-rounds", str(args.max_games // 2),
           "-concurrency", str(args.concurrency),
           "-sprt", f"elo0={args.elo0}", f"elo1={args.elo1}", "alpha=0.05", "beta=0.05", "model=normalized",
           "-draw", "movenumber=40", "movecount=8", "score=10",
           "-resign", "movecount=3", "score=1000",
           "-ratinginterval", "200", "-recover",
           "-pgnout", f"file={pgn}"]
    print(f"{name}: {new_exe}\n  vs {base_exe}\n  {'tc ' + args.tc if args.tc else str(args.nodes) + ' nodes/move'}, "
          f"SPRT [{args.elo0}, {args.elo1}], concurrency {args.concurrency}", flush=True)

    out = []
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    watchdog = None
    for line in proc.stdout:
        out.append(line)
        if re.search(r"Elo:|LLR|Games:|SPRT|H0|H1", line):
            print(line.rstrip(), flush=True)
        if "was accepted" in line and watchdog is None:
            # fastchess sometimes hangs on exit with idle engines after the SPRT ends (Windows)
            watchdog = threading.Timer(60, proc.kill)
            watchdog.start()
    proc.wait()
    if watchdog:
        watchdog.cancel()
    text = "".join(out)
    with open(pgn[:-4] + ".log", "w", encoding="utf-8") as f:
        f.write(text)
    if proc.returncode != 0:
        print(f"fastchess exited with code {proc.returncode}; full output in {pgn[:-4]}.log")

    def last(pattern):
        m = re.findall(pattern, text)
        return m[-1] if m else ""
    elo = last(r"(?<!n)Elo:\s*([-\d.]+\s*\+/-\s*[\d.]+)")
    nelo = last(r"nElo:\s*([-\d.]+\s*\+/-\s*[\d.]+)")
    llr = last(r"LLR:\s*([-\d.]+)")
    games = last(r"Games:\s*(\d+)")
    verdict = "H1 (gain)" if "H1 was accepted" in text else "H0 (no gain)" if "H0 was accepted" in text else "inconclusive"
    print(f"\nRESULT {name}: {verdict}, Elo {elo}, LLR {llr}, games {games}")

    log = os.path.join(ROOT, "data", "tests.csv")
    new_file = not os.path.exists(log)
    with open(log, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["date", "name", "new", "base", "limit", "bounds", "games", "elo", "nelo", "llr", "verdict", "pgn"])
        w.writerow([stamp, name, args.new, args.base, args.tc or f"{args.nodes}n", f"[{args.elo0},{args.elo1}]",
                    games, elo, nelo, llr, verdict, os.path.relpath(pgn, ROOT)])


if __name__ == "__main__":
    main()
