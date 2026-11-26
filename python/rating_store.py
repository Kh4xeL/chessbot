"""
Stores rated games in SQLite (data/tutor.db) so the rating estimate can combine several
games and show a trend. Each row keeps the game's summary (policy log-likelihood per rating
and loss-bin counts), so combined estimates can be recomputed if the model is refitted.
"""

import json
import os
import sqlite3
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.environ.get("TUTOR_DB", os.path.join(ROOT, "data", "tutor.db"))
_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS rated_games (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    side TEXT NOT NULL,
    start_fen TEXT,
    moves TEXT NOT NULL,
    bot_elo INTEGER,
    result TEXT,
    n_moves INTEGER NOT NULL,
    rating INTEGER NOT NULL,
    low INTEGER NOT NULL,
    high INTEGER NOT NULL,
    llp TEXT NOT NULL,
    counts TEXT NOT NULL
)"""


def _connect(path=None):
    path = path or DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = sqlite3.connect(path)
    con.execute(SCHEMA)
    return con


def add_game(side, start_fen, moves, bot_elo, result, est, llp, counts, path=None):
    with _lock, _connect(path) as con:
        cur = con.execute(
            "INSERT INTO rated_games (ts, side, start_fen, moves, bot_elo, result, n_moves, rating, low, high, llp, counts)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (time.time(), side, start_fen, " ".join(moves), bot_elo, result, est["moves"], est["rating"], est["low"],
             est["high"], json.dumps([round(float(x), 4) for x in llp]), json.dumps([int(c) for c in counts])))
        return cur.lastrowid


def recent_games(limit=10, path=None):
    """Most recent first: dicts with ts, rating, low, high, n_moves, bot_elo, result, llp, counts."""
    with _lock, _connect(path) as con:
        rows = con.execute("SELECT id, ts, rating, low, high, n_moves, bot_elo, result, llp, counts FROM rated_games"
                           " ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    keys = ["id", "ts", "rating", "low", "high", "n_moves", "bot_elo", "result", "llp", "counts"]
    out = [dict(zip(keys, r)) for r in rows]
    for g in out:
        g["llp"], g["counts"] = json.loads(g["llp"]), json.loads(g["counts"])
    return out


def clear(path=None):
    with _lock, _connect(path) as con:
        con.execute("DELETE FROM rated_games")
