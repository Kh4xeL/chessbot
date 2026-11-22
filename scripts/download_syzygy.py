"""
Downloads the 3-4-5 piece Syzygy endgame tablebases (WDL + DTZ, ~940 MB, 290 files)
from the Lichess mirror into tools/syzygy, where the engine finds them.

Usage:  python scripts/download_syzygy.py [--dest tools/syzygy] [--parallel 8]
"""

import argparse
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://tablebase.lichess.ovh/tables/standard/"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def listing(kind, ext):
    html = urllib.request.urlopen(f"{BASE}3-4-5-{kind}/", timeout=60).read().decode()
    return sorted(set(re.findall(r"[A-Za-z]+v[A-Za-z]+\." + ext, html)))


def fetch(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return 0
    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)
    return os.path.getsize(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=os.path.join(ROOT, "tools", "syzygy"))
    ap.add_argument("--parallel", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.dest, exist_ok=True)
    jobs = [(f"{BASE}3-4-5-{kind}/{name}", os.path.join(args.dest, name))
            for kind, ext in (("wdl", "rtbw"), ("dtz", "rtbz")) for name in listing(kind, ext)]
    print(f"{len(jobs)} files -> {args.dest}")
    with ThreadPoolExecutor(args.parallel) as ex:
        total = sum(ex.map(lambda j: fetch(*j), jobs))
    print(f"done, {total / 1e6:.0f} MB downloaded")


if __name__ == "__main__":
    sys.exit(main())
