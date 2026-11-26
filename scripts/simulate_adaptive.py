"""
Simulates the adaptive opponent (python/adaptive.py) against stand-in players: the engine at
a fixed Elo setting plays a series of games against the engine at the adaptive Elo (base +
in-game adjustment), with the base updated after each game as in the app. Reports how the
adaptive Elo moves and the player's score once it has settled.

    python scripts/simulate_adaptive.py --players 1000 1400 1800 2200 --games 30

Each player runs in its own process (2 engines, one thinking at a time).
"""

import argparse
import os
import subprocess
import sys
from multiprocessing import Pool

import chess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from python import adaptive  # noqa: E402

EXE = os.path.join(ROOT, "cmake-build-release", "engine.exe" if os.name == "nt" else "engine")


class Engine:
    def __init__(self):
        self.p = subprocess.Popen([EXE, "--server"], cwd=ROOT, text=True, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=1)
        assert self.p.stdout.readline().strip() == "READY"

    def go(self, board, elo, ms):
        moves = " ".join(m.uci() for m in board.move_stack)
        start = board.root().fen()
        self.p.stdin.write(f"go {elo} 99 {ms} {start}" + (f" moves {moves}" if moves else "") + "\n")
        out = {}
        while (line := self.p.stdout.readline().strip()) != "END":
            if ":" in line and not line.startswith(("RANK", "USER_RANK")):
                k, v = line.split(":", 1)
                out[k] = v
        return chess.Move.from_uci(out["BEST_MOVE"]), float(out.get("EVAL", 0))

    def newgame(self):
        self.p.stdin.write("newgame\n")
        self.p.stdout.readline()

    def close(self):
        self.p.stdin.write("quit\n")
        self.p.wait()


def play(player, bot, player_elo, bot_base, player_white, ms):
    """One game; returns the player's score (1, 0.5, 0)."""
    board = chess.Board()
    player.newgame(); bot.newgame()
    player_eval = None                       # pawns, player's view, from the bot's last search
    calm = 0
    while not board.is_game_over(claim_draw=True) and board.ply() < 240:
        if (board.turn == chess.WHITE) == player_white:
            mv, _ = player.go(board, player_elo, ms)
        else:
            elo = adaptive.clamp(bot_base + adaptive.in_game_adjust(player_eval))
            mv, ev = bot.go(board, elo, ms)
            player_eval = ev if player_white else -ev
            # Adjudicate clearly decided games, as a human would resign.
            calm = calm + 1 if abs(ev) > 8 else 0
            if calm >= 3:
                return 1.0 if player_eval > 0 else 0.0
        board.push(mv)
    res = board.result(claim_draw=True)
    if res == "*":
        return 0.5
    return adaptive.score_from_result(res, "w" if player_white else "b")


def run_player(args):
    player_elo, games, ms = args
    player, bot = Engine(), Engine()
    base, n, log = adaptive.initial_elo(None), 0, []
    try:
        for g in range(games):
            s = play(player, bot, player_elo, base, g % 2 == 0, ms)
            log.append((base, s))
            base = adaptive.update(base, n, s)
            n += 1
    finally:
        player.close(); bot.close()
    return player_elo, log, base


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--players", type=int, nargs="+", default=[1000, 1400, 1800, 2200])
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--ms", type=int, default=200, help="time per move (depth caps usually stop sooner)")
    ap.add_argument("--settle", type=int, default=10, help="games before the score is counted")
    args = ap.parse_args()

    with Pool(len(args.players)) as pool:
        for player_elo, log, final in pool.imap_unordered(run_player, [(p, args.games, args.ms) for p in args.players]):
            later = [s for _, s in log[args.settle:]]
            path = " ".join(str(b) for b, _ in log)
            print(f"player {player_elo}: final {final}, score after game {args.settle}: "
                  f"{sum(later) / max(1, len(later)):.0%} ({len(later)} games), "
                  f"overall {sum(s for _, s in log) / len(log):.0%}\n  Elo path: {path}", flush=True)


if __name__ == "__main__":
    main()
