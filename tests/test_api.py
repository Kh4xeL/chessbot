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


TABLEBASES = os.path.isfile(os.path.join(ROOT, "tools", "syzygy", "KQvK.rtbw"))


@pytest.mark.skipif(not TABLEBASES, reason="Syzygy tablebases not downloaded")
def test_tablebase_result_reported(client):
    won = post(client, "/get_move", fen="8/8/8/4k3/8/8/8/R3K3 w - - 0 1", elo=3000, time_limit=300)
    assert won["tablebase"] == 1
    drawn = post(client, "/get_move", fen="8/8/8/8/8/5k2/8/KN6 w - - 0 1", elo=3000, time_limit=300)
    assert drawn["tablebase"] == 0
    middlegame = post(client, "/get_move", fen=MIDDLEGAME, elo=3000, time_limit=200)
    assert middlegame["tablebase"] is None


# A short game: 1.e4 e5 2.Nf3 Nc6 3.Bc4 Nf6 4.Ng5 d5 5.exd5 Na5 6.Bb5+ c6 7.dxc6 bxc6
# 8.Be2 h6 9.Nf3 e4 10.Ne5 Bd6 11.d4 exd3 12.Nxd3 Qc7
RATED_GAME = ("e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 f3g5 d7d5 e4d5 c6a5 c4b5 c7c6 d5c6 b7c6 b5e2 h7h6 g5f3 e5e4 "
              "f3e5 f8d6 d2d4 e4d3 e5d3 d8c7").split()


def test_rate_game(client, tmp_path, monkeypatch):
    monkeypatch.setattr(api.rating_store, "DB_PATH", str(tmp_path / "tutor.db"))
    r = post(client, "/rate_game", moves=RATED_GAME, side="w", save=True, bot_elo=1500)
    assert r["rated"], r
    g = r["game"]
    assert 600 <= g["low"] <= g["rating"] <= g["high"] <= 2800
    assert g["moves"] >= 10
    assert r["overall"]["games"] == 1 and len(r["history"]) == 1
    # A preview isn't stored; a second saved game joins the overall estimate.
    post(client, "/rate_game", moves=RATED_GAME, side="b", save=False)
    assert client.get("/rating_history").json()["overall"]["games"] == 1
    r2 = post(client, "/rate_game", moves=RATED_GAME, side="b", save=True)
    assert r2["overall"]["games"] == 2 and len(r2["history"]) == 2


def test_rate_game_rejects_short_and_illegal(client, tmp_path, monkeypatch):
    monkeypatch.setattr(api.rating_store, "DB_PATH", str(tmp_path / "tutor.db"))
    r = post(client, "/rate_game", moves=RATED_GAME[:6], side="w")
    assert not r["rated"] and "at least" in r["reason"]
    bad = client.post("/rate_game", json={"moves": ["e2e4", "e2e4"], "side": "w"})
    assert bad.status_code == 400


def test_adaptive_opponent(client, tmp_path, monkeypatch):
    monkeypatch.setattr(api.rating_store, "DB_PATH", str(tmp_path / "tutor.db"))
    start = client.get("/rating_history").json()["adaptive"]
    assert start == {"elo": 1200, "games": 0}            # no history yet: the default
    # A won game against the adaptive opponent makes it stronger, a lost one weaker.
    won = post(client, "/rate_game", moves=RATED_GAME, side="w", save=True, adaptive=True, result="1-0")
    assert won["adaptive_change"]["from"] == 1200 and won["adaptive"]["games"] == 1
    up = won["adaptive"]["elo"]
    assert up > 1200
    lost = post(client, "/rate_game", moves=RATED_GAME[:6], side="w", save=True, adaptive=True, result="0-1")
    assert not lost["rated"] and lost["adaptive"]["elo"] < up and lost["adaptive"]["games"] == 2
    # Previews, unfinished games and non-adaptive games leave it alone.
    for body in ({"save": False, "adaptive": True, "result": "1-0"}, {"adaptive": True}, {"result": "1-0"}):
        r = post(client, "/rate_game", moves=RATED_GAME, side="w", **body)
        assert r["adaptive"] == lost["adaptive"]
    assert client.post("/adaptive/reset").json()["adaptive"]["games"] == 0


def test_adaptive_rules():
    from python import adaptive
    assert adaptive.initial_elo(None) == 1200 and adaptive.initial_elo(3500) == 2800
    assert adaptive.update(1500, 0, 1.0) == 1650 and adaptive.update(1500, 10, 0.0) == 1430
    assert adaptive.update(1500, 10, 0.5) == 1500
    assert adaptive.in_game_adjust(0.5) == 0 and adaptive.in_game_adjust(9) == 250 and adaptive.in_game_adjust(-9) == -150
    assert adaptive.score_from_result("0-1", "b") == 1.0 and adaptive.score_from_result("*", "w") is None


# Legal's trap: after 1.e4 e5 2.Nf3 d6 3.Bc4 Bg4 4.Nc3 g6 5.Nxe5, taking the queen (Bxd1) allows mate.
LEGAL_TRAP = "rn1qkbnr/ppp2p1p/3p2p1/4N3/2B1P1b1/2N5/PPPP1PPP/R1BQK2R b KQkq - 0 5"


def test_insights_trap_and_level_choice(client):
    r = post(client, "/insights", fen=LEGAL_TRAP, level=1000)
    assert r["level"] == 1000 and r["best"]["move"] == "d6e5"
    assert r["popular"][0]["move"] == "g4d1" and r["popular"][0]["share"] > 0.3
    kinds = {i["type"]: i for i in r["insights"]}
    assert kinds["trap"]["move"] == "g4d1" and "forced loss" in kinds["trap"]["text"]
    assert "Bxd1" in kinds["level_choice"]["text"] and "dxe5" in kinds["level_choice"]["text"]


def test_insights_quiet_start(client):
    r = post(client, "/insights", fen=START, level=1500)
    assert [i["type"] for i in r["insights"]] == ["level_choice"]       # no traps, nothing hard to find
    assert r["popular"][0]["share"] > 0.3
    assert post(client, "/insights", fen="7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", level=1500)["insights"] == []   # stalemate


def test_insights_use_the_tutor_lines(client):
    # Fried Liver after 4.Ng5 d5 5.exd5: the tutor's own lines (White's view) decide "the engine's move".
    fen = "r1bqkb1r/ppp2ppp/2n2n2/3Pp1N1/2B5/8/PPPP1PPP/RNBQK2R b KQkq - 0 5"
    known = [{"move": "c6a5", "eval_pawns": -0.1, "mate": None}, {"move": "f6d5", "eval_pawns": 1.6, "mate": None}]
    r = post(client, "/insights", fen=fen, level=1200, known=known)
    assert r["best"]["move"] == "c6a5" and r["best"]["eval"] == 0.1
    nxd5 = next(p for p in r["popular"] if p["move"] == "f6d5")
    assert nxd5["eval"] == -1.6 and nxd5["loss"] == 1.7
    assert any(i["type"] == "trap" and i["move"] == "f6d5" for i in r["insights"])


def test_puzzles_from_mistakes(client, tmp_path, monkeypatch):
    monkeypatch.setattr(api.rating_store, "DB_PATH", str(tmp_path / "tutor.db"))
    assert client.get("/puzzles/next").json()["puzzle"] is None
    moves = "e2e4 e7e5 g1f3 d7d6 f1c4 c8g4 b1c3 g7g6 f3e5 g4d1".split()
    pv = [chess.Move.from_uci(u) for u in ("d6e5", "d1g4")]
    assert api.puzzles.add_from_game(START, moves, [10, 8], [0.9, 0.01], [pv, pv]) == 1   # only the real mistake
    p = client.get("/puzzles/next").json()["puzzle"]
    assert p["fen"] == LEGAL_TRAP and p["played_san"] == "Bxd1"
    wrong = post(client, "/puzzles/answer", id=p["id"], move="g4d1")["verdict"]
    assert not wrong["solved"] and wrong["box"] == 0 and wrong["best_san"] == "dxe5"
    right = post(client, "/puzzles/answer", id=p["id"], move="d6e5")
    assert right["verdict"]["solved"] and right["verdict"]["box"] == 1 and right["counts"]["due"] == 0
    assert client.post("/puzzles/answer", json={"id": p["id"], "move": "e1e2"}).status_code == 400


def test_endgame_trainer(client):
    pos = post(client, "/endgame/new", ending="KQvK", color="w")
    board = chess.Board(pos["fen"])
    assert board.turn == chess.WHITE and pos["par"] >= 1
    # A tablebase-perfect move keeps the win and gets the longest defence as a reply.
    mv = api.endgames.attacker_move(board)
    r = post(client, "/endgame/move", start_fen=pos["fen"], moves=[mv.uci()])
    assert r["status"]["state"] in ("winning", "mate") and r["player_moves"] == 1
    # Giving the queen away throws the win away.
    king = board.king(chess.BLACK)
    hang = next(m for m in board.legal_moves if board.piece_type_at(m.from_square) == chess.QUEEN
                and chess.square_distance(m.to_square, king) == 1
                and not board.attackers(chess.WHITE, m.to_square) - chess.SquareSet([m.from_square]))
    r = post(client, "/endgame/move", start_fen=pos["fen"], moves=[hang.uci()])
    assert r["status"]["state"] == "lost_win" and r["reply"] is None


def test_opening_explorer(client):
    r = post(client, "/explorer", fen="rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", level=1500)
    first = r["moves"][0]
    assert first["san"] == "Nf3" and first["book"] > 0.5 and first["level"] > 0.3 and first["eval"] is not None
    assert any(m["best"] for m in r["moves"]) and r["in_book"]


def test_guess_the_move(client):
    g = client.get("/guess/game", params={"index": 0}).json()
    assert min(g["white_elo"], g["black_elo"]) >= 2400 and len(g["moves"]) >= 40
    exact = post(client, "/guess/score", game_id=0, ply=4, move=g["moves"][4])
    assert exact["match"] and exact["points"] == 10
    board = chess.Board(g["start_fen"])
    for u in g["moves"][:4]:
        board.push_uci(u)
    other = next(m.uci() for m in board.legal_moves if m.uci() != g["moves"][4])
    r = post(client, "/guess/score", game_id=0, ply=4, move=other)
    assert not r["match"] and 0 <= r["points"] <= 8 and 0 <= r["share_guess"] <= 1
