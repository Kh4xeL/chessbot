"""
Guess the move: step through strong games (assets/guess_games.pgn, Lichess, both players 2400+)
and guess each move of one side. A guess is scored by the engine and the human policy:
  * the game move itself: 10 points;
  * a move the engine rates as good as the game move: 8 points;
  * otherwise fewer points the more expected score it gives up against the game move;
and the answer shows how many 2400 players would play the guess and the game move.
"""

import os
import random

import chess
import chess.engine
import chess.pgn

from . import elo_estimate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAMES_PATH = os.path.join(ROOT, "assets", "guess_games.pgn")
POLICY_LEVEL = 2400
EQUAL_LOSS = 0.02           # expected score: a guess this close to the game move counts as "as good"
ZERO_POINTS_LOSS = 0.25     # ... and one this much worse scores nothing

_games = None


def games():
    global _games
    if _games is None:
        _games = []
        with open(GAMES_PATH, encoding="utf-8") as f:
            while (g := chess.pgn.read_game(f)) is not None:
                _games.append(g)
    return _games


def get_game(index=None):
    gs = games()
    i = random.randrange(len(gs)) if index is None else index % len(gs)
    g = gs[i]
    board = g.board()
    moves, sans = [], []
    for mv in g.mainline_moves():
        sans.append(board.san(mv))
        moves.append(mv.uci())
        board.push(mv)
    h = g.headers
    return {"id": i, "start_fen": g.board().fen(), "moves": moves, "san": sans, "result": h.get("Result", "*"),
            "white_elo": int(h.get("WhiteElo", 0)), "black_elo": int(h.get("BlackElo", 0)),
            "opening": h.get("Opening", ""), "event": h.get("Event", ""), "date": h.get("Date", "")}


def points_for(diff):
    """diff: expected score the guess gives up against the game move (negative = better)."""
    if diff <= EQUAL_LOSS:
        return 8
    return max(0, round(8 * (1 - (diff - EQUAL_LOSS) / (ZERO_POINTS_LOSS - EQUAL_LOSS))))


def score(game_id, ply, guess_uci, engine, probs, nodes=60000):
    """Scores a guess for the move at `ply` (0-based) of game `game_id`.
    probs: [(uci, share)] for that position at POLICY_LEVEL (engine humanprobs)."""
    g = games()[game_id]
    moves = list(g.mainline_moves())
    if not 0 <= ply < len(moves):
        raise ValueError("no move at that ply")
    board = g.board()
    for mv in moves[:ply]:
        board.push(mv)
    actual = moves[ply]
    guess = chess.Move.from_uci(guess_uci)
    if guess not in board.legal_moves:
        raise ValueError(f"illegal move {guess_uci}")
    share = dict(probs)
    out = {"actual": actual.uci(), "actual_san": board.san(actual), "guess": guess.uci(), "guess_san": board.san(guess),
           "share_actual": round(share.get(actual.uci(), 0.0), 4), "share_guess": round(share.get(guess.uci(), 0.0), 4),
           "level": POLICY_LEVEL}
    info = engine.analyse(board, chess.engine.Limit(nodes=nodes))
    best = info["pv"][0]
    out["best"], out["best_san"] = best.uci(), board.san(best)
    if guess == actual:
        out.update(match=True, points=10, diff=0.0)
        return out
    after_actual = elo_estimate.move_loss(engine, board, actual, nodes)[2]
    after_guess = elo_estimate.move_loss(engine, board, guess, nodes)[2]
    diff = float(elo_estimate.expected_score(after_actual) - elo_estimate.expected_score(after_guess))
    out.update(match=False, points=points_for(diff), diff=round(diff, 3),
               eval_actual=round(after_actual / 100, 2), eval_guess=round(after_guess / 100, 2))
    return out
