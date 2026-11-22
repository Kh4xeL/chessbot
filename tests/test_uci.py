"""UCI protocol smoke tests (the interface match runners and GUIs use)."""

import os
import subprocess
import sys

import chess
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from python.api import ENGINE_PATH  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.isfile(ENGINE_PATH), reason="engine not built")


def session(commands):
    """Send commands one by one; after each "go", wait for its bestmove before continuing
    (an early "quit" would stop the search)."""
    p = subprocess.Popen([ENGINE_PATH], cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    out = []
    try:
        for c in commands:
            p.stdin.write(c + "\n")
            p.stdin.flush()
            if c.startswith("go"):
                while True:
                    line = p.stdout.readline()
                    if not line:
                        break
                    out.append(line.strip())
                    if line.startswith("bestmove"):
                        break
        p.stdin.close()
        out += [l.strip() for l in p.stdout.read().splitlines()]
        p.wait(timeout=30)
    finally:
        p.kill()
    return out


def test_handshake_and_options():
    out = session(["uci", "isready", "quit"])
    assert "uciok" in out and "readyok" in out
    assert any(l.startswith("option name Threads") for l in out)


def test_fixed_nodes_search_returns_legal_move():
    out = session(["uci", "position startpos moves e2e4 e7e5 g1f3", "go nodes 30000", "isready", "quit"])
    best = [l for l in out if l.startswith("bestmove")]
    assert best
    board = chess.Board()
    for u in ("e2e4", "e7e5", "g1f3"):
        board.push_uci(u)
    assert chess.Move.from_uci(best[0].split()[1]) in board.legal_moves
    assert any(l.startswith("info depth") and " pv " in l for l in out)


def test_fixed_nodes_is_deterministic():
    cmds = ["uci", "position fen r2q1rk1/pp2bppp/2n1bn2/3p4/3N4/2N1B3/PPP1BPPP/R2Q1RK1 w - - 4 10",
            "go nodes 50000", "isready", "quit"]
    a = [l for l in session(cmds) if l.startswith("bestmove")]
    b = [l for l in session(cmds) if l.startswith("bestmove")]
    assert a == b


def test_clock_and_mate():
    out = session(["uci", "position fen 6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
                   "go wtime 10000 btime 10000 winc 100 binc 100", "isready", "quit"])
    assert "bestmove d1d8" in out
    assert any("score mate 1" in l for l in out)
