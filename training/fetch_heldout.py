"""
Streams the start of a Lichess rated-games archive from a month the policy never saw
and keeps a rating-balanced set of whole games, for checking the rating estimate
(scripts/estimate_elo.py --calibrate).

Kept: standard rated games, 3 min base or more, normal end or time forfeit, 30+ plies,
both ratings known. At most --per-band games per 200-point band of the average rating,
so weak and strong players are represented, not just the 1400-2000 crowd.

    python training/fetch_heldout.py                      # -> data/heldout_2026-09.pgn
"""

import argparse
import io
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADER_RE = re.compile(r'^\[(\w+) "(.*)"\]')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--month", default="2026-09")
    ap.add_argument("--per-band", type=int, default=500)
    ap.add_argument("--max-read", type=int, default=150_000, help="stop after reading this many games")
    args = ap.parse_args()

    import requests
    import zstandard

    url = f"https://database.lichess.org/standard/lichess_db_standard_rated_{args.month}.pgn.zst"
    out_path = os.path.join(ROOT, "data", f"heldout_{args.month}.pgn")
    bands = {b: 0 for b in range(600, 2800, 200)}          # band start -> games kept
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(resp.raw)
    text = io.TextIOWrapper(io.BufferedReader(reader, buffer_size=1 << 20), encoding="utf-8", errors="replace")

    read = kept = 0
    headers = []
    with open(out_path, "w", encoding="utf-8", newline="\n") as out:
        for line in text:
            if line.startswith("["):
                headers.append(line.rstrip("\n"))
                continue
            if not line.strip() or not headers:
                continue
            read += 1
            h = dict(HEADER_RE.match(l).groups() for l in headers if HEADER_RE.match(l))
            hdr, headers = headers, []
            try:
                welo, belo = int(h["WhiteElo"]), int(h["BlackElo"])
                base = int(h.get("TimeControl", "-").split("+")[0])
            except (KeyError, ValueError):
                continue
            plies = len(re.findall(r"\d+\.", line))  * 2
            band = min(2600, max(600, (welo + belo) // 2 // 200 * 200))
            if (base < 180 or h.get("Termination") not in ("Normal", "Time forfeit") or plies < 30
                    or h.get("Result") not in ("1-0", "0-1", "1/2-1/2") or bands[band] >= args.per_band):
                pass
            else:
                bands[band] += 1
                kept += 1
                out.write("\n".join(hdr) + "\n\n" + line.rstrip("\n") + "\n\n")
            if read % 10_000 == 0:
                print(f"read {read:,}  kept {kept:,}  " + " ".join(f"{b}:{n}" for b, n in bands.items()), flush=True)
            if read >= args.max_read or all(n >= args.per_band for n in bands.values()):
                break
    resp.close()
    print(f"DONE: read {read:,}, kept {kept:,} -> {out_path}")
    print(" ".join(f"{b}:{n}" for b, n in bands.items()))


if __name__ == "__main__":
    main()
