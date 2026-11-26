"""
Opening explorer: for a position, the candidate moves with
  * the opening book's share (assets/book.bin, master-level Polyglot book),
  * the share of players at the given rating who play it (rating-aware policy),
  * the engine's verdict: its evaluation after the move (White's view, short fixed-node search).
"""

import os

import chess
import chess.engine
import chess.polyglot

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOK_PATH = os.path.join(ROOT, "assets", "book.bin")
MAX_CANDIDATES = 8
LEVEL_MIN_SHARE = 0.03


def book_shares(board, path=BOOK_PATH):
    if not os.path.exists(path):
        return {}
    weights = {}
    with chess.polyglot.open_reader(path) as reader:
        for e in reader.find_all(board):
            weights[e.move.uci()] = weights.get(e.move.uci(), 0) + e.weight
    total = sum(weights.values())
    return {u: w / total for u, w in weights.items()} if total else {}


def _white_pawns(score):
    s = score.white()
    if s.is_mate():
        return None, s.mate()
    return round(s.score() / 100, 2), None


def explore(board, level, probs, engine, nodes=80000):
    """probs: [(uci, share)] from the engine's humanprobs at `level`."""
    if board.is_game_over():
        return {"level": level, "moves": [], "eval": None, "best": None}
    book = book_shares(board)
    level_share = dict(probs)
    root = engine.analyse(board, chess.engine.Limit(nodes=nodes))
    best = root["pv"][0].uci()
    cands = sorted(book, key=book.get, reverse=True)[:6]
    cands += [u for u, p in probs[:6] if p >= LEVEL_MIN_SHARE and u not in cands]
    if best not in cands:
        cands.append(best)
    rows = []
    for u in cands[:MAX_CANDIDATES] if best in cands[:MAX_CANDIDATES] else cands[:MAX_CANDIDATES - 1] + [best]:
        mv = chess.Move.from_uci(u)
        b = board.copy()
        b.push(mv)
        if b.is_checkmate():
            ev, mate = None, (1 if board.turn == chess.WHITE else -1)
        elif b.is_game_over(claim_draw=True):
            ev, mate = 0.0, None
        else:
            ev, mate = _white_pawns(engine.analyse(b, chess.engine.Limit(nodes=nodes))["score"])
        rows.append({"move": u, "san": board.san(mv), "book": round(book.get(u, 0.0), 4),
                     "level": round(level_share.get(u, 0.0), 4), "eval": ev, "mate": mate, "best": u == best})
    rows.sort(key=lambda r: (-r["book"], -r["level"]))
    ev, mate = _white_pawns(root["score"])
    return {"level": level, "moves": rows, "eval": ev, "mate": mate, "best": best,
            "in_book": bool(book)}
