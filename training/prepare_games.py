"""
Streams a Lichess rated-games archive and builds two datasets in one pass:

1. Policy data (for the rating-aware CNN): positions with the move the human
   actually played and both players' ratings. Ratings are balanced by capping how
   many positions each 100-point band may contribute, so rare ratings (< 1000,
   > 2400) aren't drowned out by the 1400-1900 crowd.

       data/policy/boards.u8   N x 32   nibble-packed board, a1 = 0 (codes as prepare_data.py)
       data/policy/meta.npy    N records: stm, castling, move (from*64+to, a1-based),
                               promo, mover_elo, opp_elo, base_time

2. Positions to label with Stockfish (more NNUE training data, from real games
   rather than analysis positions): one FEN per line plus the game result from
   White's view, sampled sparsely.

       data/games_fens.txt     "<fen>;<result>"   result in {1, 0.5, 0}

Bullet (< 3 min base), correspondence and abandoned games are skipped, as Maia did.

Usage:  python training/prepare_games.py --games 6000000
"""

import argparse
import io
import os
import re
import sys
import time
from multiprocessing import Pool

import numpy as np

URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2026-10.pgn.zst"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIECE_CODE = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
              "p": 7, "n": 8, "b": 9, "r": 10, "q": 11, "k": 12}
META = np.dtype([("stm", "u1"), ("castling", "u1"), ("move", "<u2"), ("promo", "u1"),
                 ("mover_elo", "<i2"), ("opp_elo", "<i2"), ("base_time", "<i2")])

HEADER_RE = re.compile(r'^\[(\w+) "(.*)"\]')
COMMENT_RE = re.compile(r"\{[^}]*\}")
TOKEN_SKIP = re.compile(r"^(\d+\.+|\$\d+|1-0|0-1|1/2-1/2|\*)$")

POLICY_SAMPLE = 0.35     # fraction of positions kept for the policy set (before band caps)
FEN_SAMPLE = 0.04        # fraction of positions written for Stockfish labelling


def pack(board):
    sq = np.zeros(64, dtype=np.uint8)
    for s, p in board.piece_map().items():
        sq[s] = PIECE_CODE[p.symbol()]
    return sq[0::2] | (sq[1::2] << 4)


def parse_games(chunk):
    import random
    import chess

    rnd = random.Random(hash(chunk[0][:64]) if chunk else 0)
    boards, metas, fens = [], [], []
    for headers_text, moves_text in chunk:
        h = dict(HEADER_RE.match(l).groups() for l in headers_text.split("\n") if HEADER_RE.match(l))
        try:
            welo, belo = int(h.get("WhiteElo", "?")), int(h.get("BlackElo", "?"))
            base = int(h.get("TimeControl", "-").split("+")[0])
        except ValueError:
            continue
        if base < 180 or h.get("Termination") == "Abandoned" or h.get("Variant", "Standard") != "Standard":
            continue
        result = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(h.get("Result"))
        if result is None:
            continue
        tokens = [t for t in COMMENT_RE.sub(" ", moves_text).split() if not TOKEN_SKIP.match(t)]
        if len(tokens) < 10:
            continue
        board = chess.Board()
        try:
            for ply, san in enumerate(tokens):
                san = san.rstrip("!?")
                move = board.parse_san(san)
                if rnd.random() < POLICY_SAMPLE:
                    white = board.turn == chess.WHITE
                    to = move.to_square
                    castling = (board.has_kingside_castling_rights(chess.WHITE) |
                                board.has_queenside_castling_rights(chess.WHITE) << 1 |
                                board.has_kingside_castling_rights(chess.BLACK) << 2 |
                                board.has_queenside_castling_rights(chess.BLACK) << 3)
                    boards.append(pack(board))
                    metas.append((1 if white else 0, castling, move.from_square * 64 + to,
                                  move.promotion or 0, welo if white else belo, belo if white else welo,
                                  min(base, 32000)))
                if ply >= 8 and rnd.random() < FEN_SAMPLE and not board.is_check():
                    fens.append(f"{board.fen()};{result}")
                board.push(move)
        except ValueError:
            continue
    if not boards:
        return np.zeros((0, 32), np.uint8), np.zeros(0, META), fens
    return np.stack(boards), np.array(metas, dtype=META), fens


def game_chunks(max_games, per_chunk=2000):
    import requests
    import zstandard

    resp = requests.get(URL, stream=True, timeout=60)
    resp.raise_for_status()
    reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(resp.raw)
    text = io.TextIOWrapper(io.BufferedReader(reader, buffer_size=1 << 22), encoding="utf-8", errors="replace")
    chunk, headers, n = [], [], 0
    for line in text:
        if line.startswith("["):
            headers.append(line.rstrip("\n"))
        elif line.strip() and headers:
            chunk.append(("\n".join(headers), line))
            headers = []
            n += 1
            if len(chunk) >= per_chunk:
                yield chunk
                chunk = []
            if n >= max_games:
                break
    if chunk:
        yield chunk
    resp.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=6_000_000)
    ap.add_argument("--band-cap", type=int, default=6_000_000, help="max policy positions per 100-Elo band")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    out = os.path.join(ROOT, "data", "policy")
    os.makedirs(out, exist_ok=True)
    fb = open(os.path.join(out, "boards.u8"), "wb")
    fm = open(os.path.join(out, "meta.bin"), "wb")
    ff = open(os.path.join(ROOT, "data", "games_fens.txt"), "w", encoding="utf-8")
    band_counts = np.zeros(40, dtype=np.int64)
    kept = games = fen_count = 0
    t0 = time.time()
    with Pool(args.workers) as pool:
        for b, m, fens in pool.imap(parse_games, game_chunks(args.games), chunksize=1):
            games += 2000
            if len(m):
                band = np.clip(m["mover_elo"] // 100, 0, 39)
                keep = np.zeros(len(m), dtype=bool)
                for i, bd in enumerate(band):        # enforce per-band caps
                    if band_counts[bd] < args.band_cap:
                        band_counts[bd] += 1
                        keep[i] = True
                fb.write(b[keep].tobytes())
                fm.write(m[keep].tobytes())
                kept += int(keep.sum())
            ff.write("\n".join(fens) + ("\n" if fens else ""))
            fen_count += len(fens)
            if games % 100_000 == 0:
                print(f"games {games:,}  policy positions {kept:,}  fens {fen_count:,}  "
                      f"{games / (time.time() - t0):,.0f} games/s", flush=True)
    for f in (fb, fm, ff):
        f.close()
    print("band counts (Elo band: positions):")
    for i, c in enumerate(band_counts):
        if c:
            print(f"  {i * 100:4d}: {c:,}")
    print(f"DONE: {games:,} games, {kept:,} policy positions, {fen_count:,} FENs in {time.time() - t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
