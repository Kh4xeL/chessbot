"""
Profile-guided build of the engine (GCC/Clang): build an instrumented engine, run it on
the bench and the self-tests (search, NNUE, CNN policy, tablebases), then rebuild using
those profiles. The result goes to the usual build directory.

    python scripts/build_pgo.py                         # -> cmake-build-release/engine.exe
    python scripts/build_pgo.py --build-dir build --portable

Compare speed with a fixed-depth bench, which searches the same tree every run:
    engine --bench d16
"""

import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from test_change import EXE, toolchain_env  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-dir", default=os.path.join(ROOT, "cmake-build-release"))
    ap.add_argument("--portable", action="store_true", help="x86-64-v3 instead of -march=native")
    args = ap.parse_args()

    env, cmake = toolchain_env()
    build = os.path.abspath(args.build_dir)
    profiles = os.path.join(build, "pgo")
    exe = os.path.join(build, EXE)
    gen = ["-G", "Ninja"] if shutil.which("ninja", path=env["PATH"]) and not os.path.isdir(build) else []
    common = [cmake, "-S", ROOT, "-B", build, "-DCMAKE_BUILD_TYPE=Release", *gen,
              f"-DENGINE_NATIVE={'OFF' if args.portable else 'ON'}", f"-DENGINE_PGO_DIR={profiles}"]

    def step(msg, cmd, **kw):
        print(msg, flush=True)
        subprocess.check_call(cmd, env=env, **kw)

    shutil.rmtree(profiles, ignore_errors=True)
    step("1/4 instrumented build", common + ["-DENGINE_PGO=GEN"], stdout=subprocess.DEVNULL)
    step("    ...", [cmake, "--build", build], stdout=subprocess.DEVNULL)
    step("2/4 profiling: bench", [exe, "--bench", "2000"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    step("3/4 profiling: self-tests", [exe, "--test"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    step("4/4 optimised build", common + ["-DENGINE_PGO=USE"], stdout=subprocess.DEVNULL)
    step("    ...", [cmake, "--build", build], stdout=subprocess.DEVNULL)
    print(f"done: {exe}")


if __name__ == "__main__":
    main()
