"""
Builds assets/guess_games.pgn for "Guess the move": strong Lichess games (both players 2400+,
3 min base or more, 40+ plies, decisive or drawn normally) from the held-out month fetched by
training/fetch_heldout.py, without player names. The Lichess database is CC0.

    python scripts/make_guess_games.py --count 150
"""

import argparse
import os
import random

import chess.pgn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pgn", default=os.path.join(ROOT, "data", "heldout_2026-09.pgn"))
    ap.add_argument("--out", default=os.path.join(ROOT, "assets", "guess_games.pgn"))
    ap.add_argument("--min-elo", type=int, default=2400)
    ap.add_argument("--count", type=int, default=150)
    args = ap.parse_args()

    keep = []
    # Player names and game links are left out: the trainer only needs the moves and ratings.
    keys = ["Event", "Date", "Result", "WhiteElo", "BlackElo", "TimeControl", "Opening"]
    with open(args.pgn, encoding="utf-8") as f:
        while (g := chess.pgn.read_game(f)) is not None:
            h = g.headers
            if min(int(h["WhiteElo"]), int(h["BlackElo"])) < args.min_elo:
                continue
            if len(list(g.mainline_moves())) < 40 or h.get("Termination") != "Normal":
                continue
            out = chess.pgn.Game()
            out.headers["White"] = f"White ({h['WhiteElo']})"
            out.headers["Black"] = f"Black ({h['BlackElo']})"
            out.headers["Site"] = "lichess.org"
            for k in keys:
                if k in h:
                    out.headers[k] = h[k]
            out.add_line(g.mainline_moves())
            keep.append(out)
    random.Random(7).shuffle(keep)
    keep = keep[:args.count]
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for g in keep:
            print(g, file=f, end="\n\n")
    print(f"wrote {len(keep)} games to {args.out}")


if __name__ == "__main__":
    main()
