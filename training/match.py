"""
Measures engine strength: plays our engine (full strength, fixed time per move)
against Stockfish limited with UCI_LimitStrength / UCI_Elo, alternating colours
over a set of balanced openings, several games in parallel.

Usage:
    python training/match.py --elo 2000 --games 20 --ms 300
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
]


class Ours:
    def __init__(self):
        self.p = subprocess.Popen([ENGINE_PATH, "--server"], cwd=ROOT, text=True, bufsize=1,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        assert self.p.stdout.readline().strip() == "READY"

    def move(self, fen, ms, elo=3000):
        self.p.stdin.write(f"go {elo} 99 {ms} {fen}\n")
        self.p.stdin.flush()
        best = None
        while True:
            line = self.p.stdout.readline().strip()
            if line.startswith("BEST_MOVE:"):
                best = line.split(":", 1)[1]
            if line == "END" or not line and self.p.poll() is not None:
                return best

    def close(self):
        self.p.kill()


def play(game_idx, sf_elo, ms, our_elo=3000, sf_ms=None):
    name, line = OPENINGS[(game_idx // 2) % len(OPENINGS)]
    ours_white = game_idx % 2 == 0
    board = chess.Board()
    for u in line.split():
        board.push_uci(u)
    ours = Ours()
    sf = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    if sf_elo > 0:
        sf.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo, "Threads": 1, "Hash": 64})
    else:
        sf.configure({"Threads": 1, "Hash": 64})  # full-strength Stockfish
    sf_ms = sf_ms or ms
    try:
        while not board.is_game_over(claim_draw=True) and board.fullmove_number < 200:
            if (board.turn == chess.WHITE) == ours_white:
                # Our engine only receives a FEN, so it can't see repetitions from game history.
                mv = chess.Move.from_uci(ours.move(board.fen(), ms, our_elo))
            else:
                mv = sf.play(board, chess.engine.Limit(time=sf_ms / 1000)).move
            board.push(mv)
    finally:
        ours.close()
        sf.quit()
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
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--ms", type=int, default=300)
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--sf-ms", type=int, default=0, help="Stockfish time per move (default: same as --ms)")
    ap.add_argument("--our-elo", type=int, default=3000, help="our engine's Elo setting (3000 = full strength)")
    args = ap.parse_args()

    with ThreadPoolExecutor(args.parallel) as ex:
        scores = list(ex.map(lambda i: play(i, args.elo, args.ms, args.our_elo, args.sf_ms), range(args.games)))
    s = sum(scores) / len(scores)
    w = sum(1 for x in scores if x == 1.0)
    d = sum(1 for x in scores if x == 0.5)
    l = len(scores) - w - d
    diff = -400 * math.log10(1 / s - 1) if 0 < s < 1 else (float("inf") if s == 1 else float("-inf"))
    if args.elo > 0:
        print(f"vs Stockfish UCI_Elo {args.elo}: +{w} ={d} -{l}  score {s:.1%}  => ~{args.elo + diff:.0f} Elo "
              f"(very rough, {len(scores)} games)")
    else:
        print(f"vs full-strength Stockfish ({args.sf_ms or args.ms} ms/move): +{w} ={d} -{l}  score {s:.1%}")


if __name__ == "__main__":
    main()
