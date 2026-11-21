"""
API tests: run with `pytest tests/` from the project root after building the engine.
Tests that need Stockfish are skipped when it isn't installed.
"""

import os
import sys

import chess
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fastapi.testclient import TestClient  # noqa: E402

import python.api as api  # noqa: E402
from python.semantic_analysis import STOCKFISH_PATH  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.isfile(api.ENGINE_PATH), reason="engine not built")
needs_stockfish = pytest.mark.skipif(not os.path.isfile(STOCKFISH_PATH), reason="Stockfish not installed")

START = chess.STARTING_FEN
MIDDLEGAME = "r2q1rk1/pp2bppp/2n1bn2/3p4/3N4/2N1B3/PPP1BPPP/R2Q1RK1 w - - 4 10"


@pytest.fixture(scope="module")
def client():
    with TestClient(api.app) as c:
        yield c
    api.ENGINE.close()


def post(client, path, **body):
    r = client.post(path, json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_engine_move_and_tutor_lines_are_legal(client):
    r = post(client, "/get_move", fen=MIDDLEGAME, elo=3000, time_limit=500)
    board = chess.Board(MIDDLEGAME)
    assert chess.Move.from_uci(r["move"]) in board.legal_moves
    assert r["depth_reached"] >= 6
    assert 1 <= len(r["top_3"]) <= 3
    for line in r["top_3"]:
        b = board.copy()
        for u in line["idea"].split():
            mv = chess.Move.from_uci(u)
            assert mv in b.legal_moves, line
            b.push(mv)
        assert line["tactics"]
    after = board.copy()
    after.push_uci(r["move"])
    for line in r["user_top_3"]:
        assert chess.Move.from_uci(line["move"]) in after.legal_moves


def test_tutor_lines_carry_final_evals(client):
    r = post(client, "/get_move", fen=MIDDLEGAME, elo=3000, time_limit=500, use_book=False)
    for line in r["top_3"]:
        assert isinstance(line["eval_pawns"], float)
    evals = [line["eval_pawns"] for line in r["top_3"]]
    assert evals == sorted(evals, reverse=True)   # White to move: best line first


def test_book_rules(client):
    assert post(client, "/get_move", fen=START, elo=2500, time_limit=300)["engine"] == "PolyGlot Book"
    assert post(client, "/get_move", fen=START, elo=1200, time_limit=300)["engine"] != "PolyGlot Book"
    hints = post(client, "/get_move", fen=START, elo=3000, time_limit=300, use_book=False)
    assert len(hints["top_3"]) == 3


def test_mate_is_found_and_reported(client):
    r = post(client, "/get_move", fen="6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", elo=3000, time_limit=500)
    assert r["move"] == "d1d8"
    assert r["mate"] == 1
    assert r["top_3"][0]["tactics"] == "🏆 Checkmate!"


def test_mate_threat_is_detected(client):
    scholar = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR b KQkq - 3 3"
    r = post(client, "/get_threats", fen=scholar)
    assert r["is_threat"] and r["threat_move"] == "f3f7"
    assert "MATE" in r["threat"]


def test_game_history_is_accepted(client):
    moves = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"]
    b = chess.Board()
    for u in moves:
        b.push_uci(u)
    r = post(client, "/get_move", fen=b.fen(), elo=3000, time_limit=300, use_book=False,
             start_fen=START, moves=moves)
    assert chess.Move.from_uci(r["move"]) in b.legal_moves


def test_low_elo_still_plays_legal_moves(client):
    for _ in range(3):
        r = post(client, "/get_move", fen=MIDDLEGAME, elo=800, time_limit=300)
        assert chess.Move.from_uci(r["move"]) in chess.Board(MIDDLEGAME).legal_moves


def test_new_game(client):
    assert post(client, "/new_game") == {"ok": True}


@needs_stockfish
def test_explore_move(client):
    r = post(client, "/explore_move", fen=MIDDLEGAME, move="d4e6")
    assert r["san"] == "Nxe6"
    assert r["classification"]


@needs_stockfish
def test_analyze_pgn(client):
    pgn = "1. e4 e5 2. Nf3 Nc6 3. Bc4 Nd4 4. Nxe5 Qg5 5. Nxf7 Qxg2 6. Rf1 Qxe4+ 7. Be2 Nf3#"
    r = post(client, "/analyze", pgn=pgn)
    assert len(r["moves"]) == 14


@needs_stockfish
def test_analyze_position_lines(client):
    fen = "r2q1rk1/pp2bppp/2n1bn2/3p4/3N4/2N1B3/PPP1BPPP/R2Q1RK1 b - - 4 10"
    r = post(client, "/analyze_position", fen=fen, multipv=3, time_ms=300)
    assert r["depth"] > 0 and len(r["lines"]) == 3
    board = chess.Board(fen)
    for line in r["lines"]:
        assert chess.Move.from_uci(line["uci"][0]) in board.legal_moves
        assert len(line["san"]) == len(line["uci"])
