"""
Puzzles from the player's own mistakes, repeated on a spaced schedule until solved.

When a finished game is rated, each of the player's moves that lost at least MISTAKE_LOSS of
expected score becomes a puzzle: the position before the move, "find the better move". An
answer counts as solved when it loses less than SOLVED_LOSS against the engine's best, so
positions with several good moves aren't ambiguous.

Schedule (Leitner boxes): solved -> next box, due after INTERVALS[box]; missed -> box 0, due
again in 10 minutes. Solving it in the last box retires the puzzle.
"""

import time

import chess

from . import elo_estimate, rating_store

MISTAKE_LOSS = 0.15                  # expected-score loss that makes a move a puzzle
SOLVED_LOSS = 0.05                   # an answer this close to the best move is accepted
RETRY_SECONDS = 10 * 60
DAY = 24 * 3600
INTERVALS = [DAY, 3 * DAY, 7 * DAY, 16 * DAY, 35 * DAY]   # after solving from box 0, 1, ...

SCHEMA = """
CREATE TABLE IF NOT EXISTS puzzles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created REAL NOT NULL,
    fen TEXT NOT NULL,
    played TEXT NOT NULL,
    best TEXT NOT NULL,
    line TEXT NOT NULL,
    loss REAL NOT NULL,
    box INTEGER NOT NULL DEFAULT 0,
    due REAL NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    solved_count INTEGER NOT NULL DEFAULT 0,
    retired INTEGER NOT NULL DEFAULT 0,
    UNIQUE (fen, played)
)"""


def _con(path=None):
    con = rating_store._connect(path)
    con.execute(SCHEMA)
    return con


def add_from_game(start_fen, moves, plies, losses, pvs, path=None):
    """Stores the player's mistakes from a rated game. plies: 1-based plies of the player's
    moves; losses/pvs: their expected-score loss and the engine's line before each."""
    board = chess.Board(start_fen)
    at = {}
    for i, u in enumerate(moves):
        at[i + 1] = board.fen()
        board.push_uci(u)
    added = 0
    now = time.time()
    with rating_store._lock, _con(path) as con:
        for p, loss, pv in zip(plies, losses, pvs):
            if loss < MISTAKE_LOSS or not pv:
                continue
            cur = con.execute("INSERT OR IGNORE INTO puzzles (created, fen, played, best, line, loss, due)"
                              " VALUES (?, ?, ?, ?, ?, ?, ?)",
                              (now, at[p], moves[p - 1], pv[0].uci(), " ".join(m.uci() for m in pv[:6]),
                               float(loss), now))
            added += cur.rowcount
    return added


def counts(path=None):
    now = time.time()
    with rating_store._lock, _con(path) as con:
        total, due, retired = con.execute(
            "SELECT COUNT(*), COALESCE(SUM(retired = 0 AND due <= ?), 0), COALESCE(SUM(retired), 0) FROM puzzles",
            (now,)).fetchone()
        nxt = con.execute("SELECT MIN(due) FROM puzzles WHERE retired = 0").fetchone()[0]
    return {"total": total, "due": due, "retired": retired, "next_due": nxt}


def next_puzzle(path=None):
    """The most overdue puzzle, or None when nothing is due."""
    with rating_store._lock, _con(path) as con:
        row = con.execute("SELECT id, fen, played, loss, box, attempts, solved_count FROM puzzles"
                          " WHERE retired = 0 AND due <= ? ORDER BY due LIMIT 1", (time.time(),)).fetchone()
    if not row:
        return None
    pid, fen, played, loss, box, attempts, solved = row
    board = chess.Board(fen)
    return {"id": pid, "fen": fen, "played": played, "played_san": board.san(chess.Move.from_uci(played)),
            "loss": round(loss, 3), "box": box, "attempts": attempts, "solved_count": solved}


def answer(pid, move_uci, engine, path=None, nodes=60000):
    """Checks an answer, moves the puzzle on its schedule and returns the verdict."""
    with rating_store._lock, _con(path) as con:
        row = con.execute("SELECT fen, played, best, line, box FROM puzzles WHERE id = ?", (pid,)).fetchone()
    if not row:
        raise KeyError(pid)
    fen, played, best, line, box = row
    board = chess.Board(fen)
    move = chess.Move.from_uci(move_uci)
    if move not in board.legal_moves:
        raise ValueError(f"illegal move {move_uci}")
    if move_uci == best:
        loss = 0.0
    else:
        # Compare the position after the answer with the one after the stored best move.
        after_answer = elo_estimate.move_loss(engine, board, move, nodes)[2]
        after_best = elo_estimate.move_loss(engine, board, chess.Move.from_uci(best), nodes)[2]
        loss = float(max(0.0, elo_estimate.expected_score(after_best) - elo_estimate.expected_score(after_answer)))
    solved = loss < SOLVED_LOSS
    now = time.time()
    if solved:
        retired = box >= len(INTERVALS) - 1
        new_box = min(box + 1, len(INTERVALS) - 1)
        due = now + INTERVALS[box]
    else:
        retired, new_box, due = False, 0, now + RETRY_SECONDS
    with rating_store._lock, _con(path) as con:
        con.execute("UPDATE puzzles SET box = ?, due = ?, attempts = attempts + 1, solved_count = solved_count + ?,"
                    " retired = ? WHERE id = ?", (new_box, due, int(solved), int(retired), pid))
    line_moves = [chess.Move.from_uci(u) for u in line.split()]
    return {"solved": solved, "loss": round(loss, 3), "best": best, "best_san": board.san(chess.Move.from_uci(best)),
            "line_san": board.variation_san(line_moves) if line_moves else "", "played": played,
            "played_san": board.san(chess.Move.from_uci(played)), "box": new_box, "retired": retired, "due": due}
