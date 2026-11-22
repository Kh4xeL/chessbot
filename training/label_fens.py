"""
Labels the FENs collected by prepare_games.py with Stockfish, producing more NNUE
training data from real games (the Lichess eval DB is mostly analysis positions).

Each worker runs its own single-threaded Stockfish at a fixed node budget
(~depth 9-11). Positions whose best move is a capture or promotion are dropped, as
in prepare_data.py. The game result is kept so training can blend eval and outcome.

Output (same layout as prepare_data.py, plus results):
    data/games_labeled/boards.u8, scores.i16 (White's view), stm.u8,
    results.u8 (0 = Black won, 1 = draw, 2 = White won)

Usage:  python training/label_fens.py --nodes 6000 --limit 12000000 [--workers 6] [--skip N]
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
SCORE_CLAMP = 2000
PIECE_CODE = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
              "p": 7, "n": 8, "b": 9, "r": 10, "q": 11, "k": 12}

_sf = None
_nodes = 6000


def init_worker(sf_path, nodes):
    import chess.engine
    global _sf, _nodes
    _sf = chess.engine.SimpleEngine.popen_uci(sf_path)
    _sf.configure({"Threads": 1, "Hash": 16})
    _nodes = nodes


def label(lines):
    import chess
    import chess.engine

    boards, scores, stms, results = [], [], [], []
    for line in lines:
        try:
            fen, res = line.rsplit(";", 1)
            board = chess.Board(fen)
            info = _sf.analyse(board, chess.engine.Limit(nodes=_nodes))
            pv = info.get("pv")
            if not pv or board.is_capture(pv[0]) or pv[0].promotion:
                continue
            sc = info["score"].white()
            cp = (SCORE_CLAMP if sc.mate() > 0 else -SCORE_CLAMP) if sc.is_mate() else \
                max(-SCORE_CLAMP, min(SCORE_CLAMP, sc.score()))
            sq = np.zeros(64, dtype=np.uint8)
            for s, p in board.piece_map().items():
                sq[s] = PIECE_CODE[p.symbol()]
            boards.append(sq[0::2] | (sq[1::2] << 4))
            scores.append(cp)
            stms.append(1 if board.turn == chess.WHITE else 0)
            results.append(int(round(float(res) * 2)))
        except Exception:
            continue
    if not boards:
        return (np.zeros((0, 32), np.uint8), np.zeros(0, np.int16), np.zeros(0, np.uint8), np.zeros(0, np.uint8))
    return (np.stack(boards), np.asarray(scores, np.int16), np.asarray(stms, np.uint8), np.asarray(results, np.uint8))


def batches(path, limit, skip=0, size=500):
    buf, n = [], 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < skip:
                continue
            line = line.strip()
            if not line:
                continue
            buf.append(line)
            n += 1
            if len(buf) >= size:
                yield buf
                buf = []
            if n >= limit:
                break
    if buf:
        yield buf


def main():
    from python.semantic_analysis import STOCKFISH_PATH

    ap = argparse.ArgumentParser()
    ap.add_argument("--fens", default=os.path.join(ROOT, "data", "games_fens.txt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "games_labeled"))
    ap.add_argument("--nodes", type=int, default=6000)
    ap.add_argument("--limit", type=int, default=12_000_000)
    # Each worker is a Stockfish process running flat out; keep it well below the core
    # count so the machine stays usable and cool.
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--skip", type=int, default=0, help="skip the first N FENs (resume)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    mode = "ab" if args.skip else "wb"   # resuming appends to the existing output
    files = [open(os.path.join(args.out, n), mode) for n in ("boards.u8", "scores.i16", "stm.u8", "results.u8")]
    done = kept = 0
    t0 = time.time()
    with Pool(args.workers, initializer=init_worker, initargs=(STOCKFISH_PATH, args.nodes)) as pool:
        for arrays in pool.imap_unordered(label, batches(args.fens, args.limit, args.skip)):
            for f, a in zip(files, arrays):
                f.write(a.tobytes())
            done += 500
            kept += len(arrays[1])
            if done % 100_000 == 0:
                rate = done / (time.time() - t0)
                print(f"labelled {done:,}  kept {kept:,}  {rate:,.0f} pos/s", flush=True)
    for f in files:
        f.close()
    print(f"DONE: kept {kept:,} of {done:,} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
