"""
Measures engine strength by playing games over a set of balanced openings,
alternating colours, several games in parallel.

Opponents:
  * Stockfish, limited with UCI_Elo (--elo N) or at full strength (--elo 0)
  * another build of our engine (--opponent path/to/engine.exe), e.g. the
    previous version, to measure what a change is worth

Usage:
    python training/match.py --elo 2400 --games 40 --ms 300
    python training/match.py --opponent data/base/engine.exe --games 100 --ms 200
"""

import argparse
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import chess
import chess.engine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from python.semantic_analysis import STOCKFISH_PATH  # noqa: E402
from python.api import ENGINE_PATH  # noqa: E402

OPENINGS = [  # (name, moves) - common, roughly balanced lines
    ("Italian", "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5"),
    ("Sicilian Najdorf", "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6"),
    ("French", "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6"),
    ("Caro-Kann", "e2e4 c7c6 d2d4 d7d5 e4e5 c8f5"),
    ("QGD", "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6"),
    ("Slav", "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6"),
    ("KID", "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6"),
    ("English", "c2c4 e7e5 b1c3 g8f6 g1f3 b8c6"),
    ("Ruy Lopez", "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6"),
    ("London", "d2d4 d7d5 c1f4 g8f6 e2e3 e7e6"),
    ("Nimzo-Indian", "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4"),
    ("Grunfeld", "d2d4 g8f6 c2c4 g7g6 b1c3 d7d5"),
    ("Scotch", "e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4"),
    ("Sicilian Alapin", "e2e4 c7c5 c2c3 g8f6 e4e5 f6d5"),
    ("Dutch", "d2d4 f7f5 g2g3 g8f6 f1g2 e7e6"),
]


class EnginePlayer:
    """Our engine in --server mode. Gets the full game so it can see repetitions."""

    def __init__(self, path, ms, elo=3000, threads=1, cwd=ROOT):
        self.ms, self.elo = ms, elo
        self.p = subprocess.Popen([path, "--server", "--threads", str(threads)], cwd=cwd, text=True, bufsize=1,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        assert self.p.stdout.readline().strip() == "READY"

    def move(self, board):
        start = board.root().fen()
        moves = " ".join(m.uci() for m in board.move_stack)
        pos = f"{start} moves {moves}" if moves else start
        self.p.stdin.write(f"go {self.elo} 99 {self.ms} {pos}\n")
        self.p.stdin.flush()
        best = None
        while True:
            line = self.p.stdout.readline().strip()
            if line.startswith("BEST_MOVE:"):
                best = line.split(":", 1)[1]
            if line == "END" or not line and self.p.poll() is not None:
                return chess.Move.from_uci(best)

    def close(self):
        self.p.kill()


class StockfishPlayer:
    def __init__(self, ms, elo):
        self.ms = ms
        self.sf = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        opts = {"Threads": 1, "Hash": 64}
        if elo > 0:
            opts.update({"UCI_LimitStrength": True, "UCI_Elo": elo})
        self.sf.configure(opts)

    def move(self, board):
        return self.sf.play(board, chess.engine.Limit(time=self.ms / 1000)).move

    def close(self):
        self.sf.quit()


def play(game_idx, args):
    name, line = OPENINGS[(game_idx // 2) % len(OPENINGS)]
    ours_white = game_idx % 2 == 0
    board = chess.Board()
    for u in line.split():
        board.push_uci(u)
    ours = EnginePlayer(ENGINE_PATH, args.ms, args.our_elo, args.threads)
    if args.opponent:
        opp = EnginePlayer(os.path.abspath(args.opponent), args.opp_ms or args.ms, 3000, args.opp_threads,
                           os.path.abspath(args.opp_cwd) if args.opp_cwd else ROOT)
    else:
        opp = StockfishPlayer(args.sf_ms or args.ms, args.elo)
    try:
        while not board.is_game_over(claim_draw=True) and board.fullmove_number < 200:
            player = ours if (board.turn == chess.WHITE) == ours_white else opp
            board.push(player.move(board))
    finally:
        ours.close()
        opp.close()
    res = board.result(claim_draw=True)
    if res == "*":
        res = "1/2-1/2"
    score = {"1-0": 1.0, "0-1": 0.0}.get(res, 0.5)
    if not ours_white:
        score = 1.0 - score
    print(f"game {game_idx + 1:3d} {name:16s} ours={'W' if ours_white else 'B'} {res:8s} "
          f"({len(board.move_stack)} plies)", flush=True)
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elo", type=int, default=2000, help="Stockfish UCI_Elo (0 = full strength)")
    ap.add_argument("--opponent", default="", help="path to another engine build to play instead of Stockfish")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--ms", type=int, default=300, help="our time per move")
    ap.add_argument("--sf-ms", type=int, default=0, help="Stockfish time per move (default: same as --ms)")
    ap.add_argument("--opp-ms", type=int, default=0, help="opponent engine time per move (default: same as --ms)")
    ap.add_argument("--threads", type=int, default=1, help="our search threads")
    ap.add_argument("--opp-threads", type=int, default=1, help="opponent engine search threads")
    ap.add_argument("--opp-cwd", default="", help="working directory for the opponent (its assets/ = its networks)")
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--our-elo", type=int, default=3000, help="our engine's Elo setting (3000 = full strength)")
    args = ap.parse_args()

    with ThreadPoolExecutor(args.parallel) as ex:
        scores = list(ex.map(lambda i: play(i, args), range(args.games)))
    n = len(scores)
    s = sum(scores) / n
    w = sum(1 for x in scores if x == 1.0)
    d = sum(1 for x in scores if x == 0.5)
    l = n - w - d
    diff = -400 * math.log10(1 / s - 1) if 0 < s < 1 else (float("inf") if s == 1 else float("-inf"))
    # 95% error bar on the score, converted to Elo around the measured value.
    var = sum((x - s) ** 2 for x in scores) / max(1, n - 1)
    margin = 1.96 * math.sqrt(var / n)
    lo, hi = max(1e-3, s - margin), min(1 - 1e-3, s + margin)
    err = (-400 * math.log10(1 / hi - 1) - -400 * math.log10(1 / lo - 1)) / 2 if 0 < s < 1 else float("nan")
    record = f"+{w} ={d} -{l}  score {s:.1%}"
    if args.opponent:
        print(f"vs {os.path.basename(os.path.dirname(args.opponent))}/{os.path.basename(args.opponent)}: "
              f"{record}  => {diff:+.0f} +/- {err:.0f} Elo ({n} games)")
    elif args.elo > 0:
        print(f"vs Stockfish UCI_Elo {args.elo}: {record}  => ~{args.elo + diff:.0f} +/- {err:.0f} Elo ({n} games)")
    else:
        print(f"vs full-strength Stockfish ({args.sf_ms or args.ms} ms/move): {record}")


if __name__ == "__main__":
    main()
