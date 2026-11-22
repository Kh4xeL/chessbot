"""
Builds tests/openings.epd: balanced start positions for engine-vs-engine testing.

Takes positions 4-8 moves into real Lichess games (data/games_fens.txt from
training/prepare_games.py), keeps those Stockfish rates within +-50 cp, and
de-duplicates. Each opening is played twice (both colours) by the match runner.

Usage:  python scripts/make_openings.py [--count 5000] [--workers 6]
"""

import argparse
import os
import random
import sys
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_sf = None


def init(path):
    import chess.engine
    global _sf
    _sf = chess.engine.SimpleEngine.popen_uci(path)
    _sf.configure({"Threads": 1, "Hash": 16})


def balanced(fen):
    import chess
    import chess.engine
    board = chess.Board(fen)
    if board.is_game_over() or board.is_check():
        return None
    info = _sf.analyse(board, chess.engine.Limit(nodes=40000))
    score = info["score"].white()
    if score.is_mate() or abs(score.score()) > 50:
        return None
    return " ".join(fen.split()[:4])          # EPD: no move counters


def main():
    from python.semantic_analysis import STOCKFISH_PATH

    ap = argparse.ArgumentParser()
    ap.add_argument("--fens", default=os.path.join(ROOT, "data", "games_fens.txt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "tests", "openings.epd"))
    ap.add_argument("--count", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    rng = random.Random(7)
    pool_fens, seen = [], set()
    with open(args.fens, encoding="utf-8") as f:
        for line in f:
            fen = line.split(";")[0]
            parts = fen.split()
            if len(parts) < 6 or not 5 <= int(parts[5]) <= 8:
                continue
            key = " ".join(parts[:4])
            if key in seen:
                continue
            seen.add(key)
            # reservoir sample of 4x the target, so the result isn't biased to the file's start
            if len(pool_fens) < args.count * 4:
                pool_fens.append(fen)
            else:
                j = rng.randrange(len(seen))
                if j < len(pool_fens):
                    pool_fens[j] = fen
    print(f"{len(seen):,} distinct candidate positions, evaluating {len(pool_fens):,}")

    with Pool(args.workers, initializer=init, initargs=(STOCKFISH_PATH,)) as pool:
        kept = [e for e in pool.imap(balanced, pool_fens, chunksize=50) if e]
    rng.shuffle(kept)
    kept = kept[: args.count]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(kept) + "\n")
    print(f"wrote {len(kept):,} openings to {args.out}")


if __name__ == "__main__":
    main()
