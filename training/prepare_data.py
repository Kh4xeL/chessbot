"""
Streams the Lichess Stockfish evaluation database and converts it into a compact
binary training set for the NNUE evaluator.

Source: https://database.lichess.org/lichess_db_eval.jsonl.zst
Each line is one unique position with one or more Stockfish evaluations. We keep
the deepest evaluation's best line and store:

    data/boards.u8   N x 32 bytes   64 squares, 4 bits each (a1 = square 0)
                                    0 = empty, 1..6 = white P N B R Q K, 7..12 = black
    data/scores.i16  N int16        Stockfish centipawns, WHITE's point of view
    data/stm.u8      N uint8        1 = white to move, 0 = black to move

Filters (standard for NNUE "quiet position" training): positions where the side
to move is in check, or where the best move is a capture or promotion, are
skipped because a static evaluation cannot be expected to match them.

The file is decompressed on the fly, so only the part we read is downloaded.

Usage:  python training/prepare_data.py --max-lines 100000000
"""

import argparse
import io
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
SCORE_CLAMP = 2000          # centipawns; mates map to +-SCORE_CLAMP
PIECE_CODE = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
              "p": 7, "n": 8, "b": 9, "r": 10, "q": 11, "k": 12}


def parse_lines(lines):
    import chess  # imported in the worker

    boards, scores, stms = [], [], []
    for line in lines:
        try:
            rec = json.loads(line)
            evals = rec["evals"]
            best = max(evals, key=lambda e: e.get("depth", 0))
            pv = best["pvs"][0]
            if "mate" in pv:
                score = SCORE_CLAMP if pv["mate"] > 0 else -SCORE_CLAMP
            else:
                score = max(-SCORE_CLAMP, min(SCORE_CLAMP, int(pv["cp"])))

            board = chess.Board(rec["fen"])
            if board.is_check():
                continue
            first = pv.get("line", "").split(" ")[0]
            if first:
                mv = chess.Move.from_uci(first)
                if mv.promotion or board.is_capture(mv):
                    continue

            sq = np.zeros(64, dtype=np.uint8)
            placement = rec["fen"].split(" ", 1)[0]
            rank, file = 7, 0
            for ch in placement:
                if ch == "/":
                    rank -= 1
                    file = 0
                elif ch.isdigit():
                    file += int(ch)
                else:
                    sq[rank * 8 + file] = PIECE_CODE[ch]
                    file += 1
            boards.append(sq[0::2] | (sq[1::2] << 4))
            scores.append(score)
            stms.append(1 if board.turn == chess.WHITE else 0)
        except Exception:
            continue

    if not boards:
        return (np.zeros((0, 32), np.uint8), np.zeros(0, np.int16), np.zeros(0, np.uint8))
    return (np.stack(boards).astype(np.uint8),
            np.asarray(scores, dtype=np.int16),
            np.asarray(stms, dtype=np.uint8))


def line_batches(max_lines, batch=20000):
    import requests
    import zstandard

    resp = requests.get(URL, stream=True, timeout=60)
    resp.raise_for_status()
    reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(resp.raw)
    text = io.TextIOWrapper(io.BufferedReader(reader, buffer_size=1 << 22), encoding="utf-8")
    buf, n = [], 0
    for line in text:
        buf.append(line)
        n += 1
        if len(buf) >= batch:
            yield buf
            buf = []
        if n >= max_lines:
            break
    if buf:
        yield buf
    resp.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-lines", type=int, default=100_000_000)
    ap.add_argument("--out", default="data")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    fb = open(os.path.join(args.out, "boards.u8"), "wb")
    fs = open(os.path.join(args.out, "scores.i16"), "wb")
    ft = open(os.path.join(args.out, "stm.u8"), "wb")

    kept, read, t0 = 0, 0, time.time()
    with Pool(args.workers) as pool:
        for (b, s, t), n_in in pool.imap(_wrap, _counted(line_batches(args.max_lines)), chunksize=1):
            fb.write(b.tobytes()); fs.write(s.tobytes()); ft.write(t.tobytes())
            kept += len(s); read += n_in
            if read % 1_000_000 < 20000:
                rate = read / (time.time() - t0)
                print(f"read {read:,} lines, kept {kept:,} ({kept / max(1, read):.0%}), {rate:,.0f} lines/s",
                      flush=True)
    for f in (fb, fs, ft):
        f.close()
    print(f"DONE: kept {kept:,} of {read:,} positions in {time.time() - t0:.0f}s", flush=True)


def _counted(gen):
    for lines in gen:
        yield lines


def _wrap(lines):
    return parse_lines(lines), len(lines)


if __name__ == "__main__":
    sys.exit(main())
