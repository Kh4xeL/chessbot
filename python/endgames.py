"""
Endgame trainer: convert a won endgame (5 pieces or fewer) against perfect tablebase defence.

Positions are random but checked with the Syzygy tables: a win for the player, not mate in one,
and not too short. The defender always picks the move that resists longest by DTZ (distance
to the next capture or pawn move), which is what tablebase defence means in practice. "Par"
is the number of moves a tablebase-perfect attacker (shortest DTZ, mate when it can) needs
against that defence, from playing the position out once.
"""

import os
import random

import chess
import chess.syzygy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYZYGY_DIR = os.environ.get("SYZYGY_PATH", os.path.join(ROOT, "tools", "syzygy"))

# name -> (title, attacker pieces besides the king, defender pieces besides the king, min par moves)
ENDINGS = {
    "KQvK": ("Queen vs king", "Q", "", 6),
    "KRvK": ("Rook vs king", "R", "", 10),
    "KBBvK": ("Two bishops vs king", "BB", "", 12),
    "KBNvK": ("Bishop and knight vs king", "BN", "", 20),
    "KPvK": ("King and pawn vs king", "P", "", 8),
    "KQvKR": ("Queen vs rook", "Q", "R", 12),
    "KRPvKR": ("Rook and pawn vs rook", "RP", "R", 10),
    "KQvKP": ("Queen vs advanced pawn", "Q", "P", 8),
}
MAX_PAR = 60

_tb = None


def tablebase():
    global _tb
    if _tb is None:
        if not os.path.isdir(SYZYGY_DIR):
            raise RuntimeError("Syzygy tablebases not found (scripts/download_syzygy.py)")
        _tb = chess.syzygy.open_tablebase(SYZYGY_DIR)
    return _tb


def defender_move(board):
    """The tablebase defence: keep the best result, and among losing moves resist longest."""
    tb = tablebase()
    best, key = None, None
    for mv in board.legal_moves:
        board.push(mv)
        if board.is_checkmate():
            k = (2, 0)                               # mating the attacker: best possible
        elif board.is_game_over(claim_draw=False):
            k = (1, 0)                               # stalemate / insufficient material: a draw
        else:
            wdl = -tb.probe_wdl(board)               # from the defender's view
            dtz = abs(tb.probe_dtz(board))
            k = (0 if wdl < 0 else (1 if wdl == 0 else 2), dtz if wdl < 0 else -dtz)
        board.pop()
        if key is None or k > key:
            best, key = mv, k
    return best


def attacker_move(board):
    """A tablebase-perfect attacking move: mate if possible, else the shortest winning DTZ."""
    tb = tablebase()
    best, key = None, None
    for mv in board.legal_moves:
        board.push(mv)
        if board.is_checkmate():
            board.pop()
            return mv
        if board.is_game_over(claim_draw=False):
            k = (1, 0)
        else:
            wdl = -tb.probe_wdl(board)
            dtz = abs(tb.probe_dtz(board))
            zeroing = board.halfmove_clock == 0           # a capture or pawn move that keeps the win
            k = (0 if wdl > 0 else (1 if wdl == 0 else 2), 0 if zeroing else dtz)
        board.pop()
        if key is None or k < key:
            best, key = mv, k
    return best


def par(board, limit=150):
    """Attacker moves to mate with perfect play on both sides (by DTZ), or None if it doesn't end."""
    b = board.copy(stack=False)
    moves = 0
    for _ in range(limit):
        if b.is_checkmate():
            return moves
        if b.is_game_over(claim_draw=False):
            return None
        b.push(attacker_move(b))
        moves += 1
        if b.is_checkmate():
            return moves
        if b.is_game_over(claim_draw=False):
            return None
        b.push(defender_move(b))
    return None


def _random_position(ending, white_attacks, rng):
    _, att, dfn, _ = ENDINGS[ending]
    attacker = chess.WHITE if white_attacks else chess.BLACK
    board = chess.Board(None)
    squares = rng.sample(range(64), 2 + len(att) + len(dfn))
    board.set_piece_at(squares[0], chess.Piece(chess.KING, attacker))
    board.set_piece_at(squares[1], chess.Piece(chess.KING, not attacker))
    pieces = [(sym, attacker) for sym in att] + [(sym, not attacker) for sym in dfn]
    for sq, (sym, color) in zip(squares[2:], pieces):
        board.set_piece_at(sq, chess.Piece(chess.Piece.from_symbol(sym).piece_type, color))
    board.turn = attacker
    return board


def new_position(ending, white_attacks=True, seed=None, tries=5000):
    """A random won position for the attacker (to move), with its par."""
    if ending not in ENDINGS:
        raise ValueError(f"unknown ending {ending}")
    rng = random.Random(seed)
    tb = tablebase()
    title, att, dfn, min_par = ENDINGS[ending]
    for _ in range(tries):
        board = _random_position(ending, white_attacks, rng)
        if not board.is_valid() or board.is_game_over():
            continue
        # Pawns: not on the first or last rank (is_valid checks), and for KQvKP the pawn should be advanced.
        if ending == "KQvKP":
            p = next(iter(board.pieces(chess.PAWN, not board.turn)))
            if chess.square_rank(p) not in ((1, 2) if board.turn == chess.WHITE else (5, 6)):
                continue
        if tb.probe_wdl(board) != 2:
            continue
        p = par(board)
        if p is None or not (min_par <= p <= MAX_PAR):
            continue
        return {"ending": ending, "title": title, "fen": board.fen(), "par": p}
    raise RuntimeError(f"no {ending} position found")


def status(board):
    """The attacker's situation after their move (defender to move)."""
    if board.is_checkmate():
        return {"state": "mate"}
    if board.is_game_over(claim_draw=True):
        return {"state": "draw", "reason": "stalemate" if board.is_stalemate() else "draw"}
    wdl = -tablebase().probe_wdl(board)        # attacker's view
    if wdl <= 0:
        return {"state": "lost_win", "wdl": wdl}
    return {"state": "winning", "dtz": abs(tablebase().probe_dtz(board))}
