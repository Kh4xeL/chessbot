"""
Per-move features of held-out rated games (training/fetch_heldout.py), for fitting and
checking the rating estimator (python/elo_estimate.py):

  * logp[i, r]: log-probability of the played move for a player rated RATINGS[r], from the
    rating-aware policy (assets/policy_elo.bin, run on the GPU; the same numbers as the
    engine's humangame command);
  * loss[i]:    expected-score loss of the played move, from the engine's own evaluation of
    the position before and after it (UCI, fixed nodes, --procs processes).

    python training/heldout_features.py              # -> data/heldout_features.npz
"""

import argparse
import os
import struct
import subprocess
import sys
import time
from multiprocessing import Pool

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from python.elo_estimate import RATINGS, expected_score, move_loss  # noqa: E402

ENGINE = os.path.join(ROOT, "cmake-build-release", "engine.exe" if os.name == "nt" else "engine")


class FoldedPolicy(torch.nn.Module):
    """The rating-aware CNN with BatchNorm folded in, loaded from the engine's .bin file."""

    def __init__(self, path):
        super().__init__()
        raw = open(path, "rb").read()
        assert raw[:4] == b"POL2"
        planes, blocks, ch = struct.unpack("<iii", raw[4:16])
        w = np.frombuffer(raw[16:], dtype="<f4")
        pos = 0

        def take(*shape):
            nonlocal pos
            n = int(np.prod(shape))
            t = torch.from_numpy(w[pos:pos + n].copy()).view(*shape)
            pos += n
            return torch.nn.Parameter(t, requires_grad=False)

        self.stem_w, self.stem_b = take(ch, planes, 3, 3), take(ch)
        self.blk = torch.nn.ParameterList()
        for _ in range(blocks * 2):
            self.blk.append(take(ch, ch, 3, 3))
            self.blk.append(take(ch))
        self.head_w, self.head_b = take(ch, ch, 3, 3), take(ch)
        self.out_w, self.out_b = take(64, ch, 1, 1), take(64)
        assert pos == len(w)
        self.blocks = blocks

    def forward(self, x):
        h = F.relu(F.conv2d(x, self.stem_w, self.stem_b, padding=1))
        for b in range(self.blocks):
            t = F.relu(F.conv2d(h, self.blk[4 * b], self.blk[4 * b + 1], padding=1))
            t = F.conv2d(t, self.blk[4 * b + 2], self.blk[4 * b + 3], padding=1)
            h = F.relu(h + t)
        t = F.relu(F.conv2d(h, self.head_w, self.head_b, padding=1))
        y = F.conv2d(t, self.out_w, self.out_b)          # [B, 64(to), 8, 8(from)]
        return y.flatten(2).transpose(1, 2).reshape(-1, 4096)


def planes(board):
    """[17, 64] input planes without the mover-rating plane (index 16 is filled per rating)."""
    x = np.zeros((18, 64), np.float32)
    white = board.turn == chess.WHITE
    for sq, pc in board.piece_map().items():
        own = pc.color == board.turn
        x[(0 if own else 6) + pc.piece_type - 1, sq if white else sq ^ 56] = 1
    us, them = board.turn, not board.turn
    x[12] = board.has_kingside_castling_rights(us)
    x[13] = board.has_queenside_castling_rights(us)
    x[14] = board.has_kingside_castling_rights(them)
    x[15] = board.has_queenside_castling_rights(them)
    return x


def move_index(m, white):
    flip = 0 if white else 56
    return (m.from_square ^ flip) * 64 + (m.to_square ^ flip)


def read_games(path):
    games = []
    with open(path, encoding="utf-8") as f:
        while (g := chess.pgn.read_game(f)) is not None:
            games.append(g)
    return games


def policy_logp(games, dev, batch=4096):
    """Per position: log P(played move | rating) for every rating, normalised over legal moves."""
    net = FoldedPolicy(os.path.join(ROOT, "assets", "policy_elo.bin")).to(dev).eval()
    xs, legal, played, meta, fens = [], [], [], [], []
    for gi, g in enumerate(games):
        board = g.board()
        welo, belo = int(g.headers["WhiteElo"]), int(g.headers["BlackElo"])
        for ply, mv in enumerate(g.mainline_moves()):
            white = board.turn == chess.WHITE
            idx = [move_index(m, white) for m in board.legal_moves]
            xs.append(planes(board))
            legal.append(idx)
            played.append(move_index(mv, white))
            meta.append((gi, ply, white, welo if white else belo, belo if white else welo))
            fens.append(board.fen() + "|" + mv.uci())
            board.push(mv)
    n = len(xs)
    print(f"{len(games):,} games, {n:,} positions", flush=True)

    # Legal-move mask (promotions share one index, so count duplicates like the engine does).
    maxl = max(len(l) for l in legal)
    li = np.zeros((n, maxl), np.int64)
    lm = np.zeros((n, maxl), bool)
    for i, l in enumerate(legal):
        li[i, :len(l)] = l
        lm[i, :len(l)] = True
    logp = np.zeros((n, len(RATINGS)), np.float32)
    x_all = torch.from_numpy(np.stack(xs))
    with torch.no_grad():
        for s in range(0, n, batch):
            x = x_all[s:s + batch].to(dev)
            idx = torch.from_numpy(li[s:s + batch]).to(dev)
            mask = torch.from_numpy(lm[s:s + batch]).to(dev)
            pl = torch.tensor(played[s:s + batch], device=dev)
            for r, rating in enumerate(RATINGS):
                x[:, 16] = (rating - 1500) / 500
                x[:, 17] = (rating - 1500) / 500     # the engine assumes an opponent of the same rating
                logits = net(x.view(-1, 18, 8, 8))
                lg = logits.gather(1, idx).masked_fill(~mask, -1e9)
                lse = torch.logsumexp(lg, dim=1)
                logp[s:s + batch, r] = (logits.gather(1, pl[:, None])[:, 0] - lse).cpu().numpy()
    return logp, np.array(meta, dtype=np.int32), fens


def eval_chunk(args):
    """Engine score (mover's view, cp) before and after each move, for a chunk of positions;
    the same code the API uses (python/elo_estimate.move_loss)."""
    fens, nodes = args
    eng = chess.engine.SimpleEngine.popen_uci(ENGINE, cwd=ROOT, stderr=subprocess.DEVNULL)
    out = np.zeros((len(fens), 2), np.float32)
    try:
        for i, fm in enumerate(fens):
            fen, mv = fm.split("|")
            _, before, after = move_loss(eng, chess.Board(fen), chess.Move.from_uci(mv), nodes)
            out[i] = (before, after)
    finally:
        eng.quit()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pgn", default=os.path.join(ROOT, "data", "heldout_2026-09.pgn"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "heldout_features.npz"))
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--nodes", type=int, default=20000)
    args = ap.parse_args()

    t0 = time.time()
    games = read_games(args.pgn)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    logp, meta, fens = policy_logp(games, dev)
    print(f"policy done in {time.time() - t0:.0f}s", flush=True)

    chunks = [(fens[i:i + 2000], args.nodes) for i in range(0, len(fens), 2000)]
    evals = []
    with Pool(args.procs) as pool:
        for k, e in enumerate(pool.imap(eval_chunk, chunks)):
            evals.append(e)
            if k % 10 == 0:
                print(f"evals {min(len(fens), (k + 1) * 2000):,}/{len(fens):,}  {time.time() - t0:.0f}s", flush=True)
    ev = np.concatenate(evals)
    loss = np.clip(expected_score(ev[:, 0]) - expected_score(ev[:, 1]), 0, 1).astype(np.float32)
    np.savez_compressed(args.out, logp=logp, meta=meta, evals=ev, loss=loss, ratings=np.array(RATINGS))
    print(f"wrote {args.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
