"""
Trains the rating-aware move-prediction CNN (Maia-style): given a position and the
players' ratings, predict the move a human of that rating actually played.

Input planes (18 x 8 x 8), always from the side to move's point of view (the board
is flipped vertically and colours swapped when Black is to move):
    0-5   own pawns, knights, bishops, rooks, queens, king
    6-11  opponent's pieces
    12-15 castling rights: own O-O, own O-O-O, opponent O-O, opponent O-O-O (constant planes)
    16    mover's rating     (elo - 1500) / 500   (constant plane)
    17    opponent's rating  (elo - 1500) / 500
    square index = rank * 8 + file with a1 = 0 (after the flip)

Network: 3x3 conv stem -> N residual blocks (C channels) -> policy head:
    3x3 conv C->C, ReLU, 1x1 conv C->64. Output channel t at square f is the logit of
    the move from f to t, i.e. logits[from * 64 + to] in the flipped frame.

Data: data/policy/ from prepare_games.py. Writes assets/policy_elo.bin for the
engine (format "POL2", BatchNorm folded) and reports top-1 accuracy per rating band.

Usage:  python training/train_policy.py --epochs 2
"""

import argparse
import math
import os
import struct
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META = np.dtype([("stm", "u1"), ("castling", "u1"), ("move", "<u2"), ("promo", "u1"),
                 ("mover_elo", "<i2"), ("opp_elo", "<i2"), ("base_time", "<i2")])
PLANES = 18


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(c)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class PolicyNet(nn.Module):
    def __init__(self, blocks=6, channels=96):
        super().__init__()
        self.stem = nn.Conv2d(PLANES, channels, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList([Block(channels) for _ in range(blocks)])
        self.head = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.head_bn = nn.BatchNorm2d(channels)
        self.out = nn.Conv2d(channels, 64, 1)

    def forward(self, x):
        x = F.relu(self.stem_bn(self.stem(x)))
        for b in self.blocks:
            x = b(x)
        x = F.relu(self.head_bn(self.head(x)))
        y = self.out(x)                        # [B, 64(to), 8, 8(from)]
        return y.flatten(2).transpose(1, 2).reshape(-1, 4096)   # [B, from*64 + to]


def make_input(packed, meta_stm, castling, mover_elo, opp_elo):
    """GPU tensors -> [B,18,8,8] planes and the flip flag (1 when Black to move)."""
    lo = (packed & 15).long()
    hi = (packed >> 4).long()
    pc = torch.stack([lo, hi], dim=2).reshape(-1, 64)                # a1 = 0
    black = (meta_stm == 0)
    # Flip squares vertically for Black to move: new[s] = old[s ^ 56].
    idx = torch.arange(64, device=pc.device)
    pc = torch.where(black.unsqueeze(1), pc[:, idx ^ 56], pc)
    # Swap colours for Black: own pieces become codes 1-6.
    own = torch.where(black.unsqueeze(1), torch.where(pc > 6, pc - 6, torch.where(pc > 0, pc + 6, pc)), pc)
    B = pc.shape[0]
    x = torch.zeros(B, PLANES, 64, device=pc.device)
    occ = own > 0
    x.scatter_(1, (own - 1).clamp(min=0).unsqueeze(1), occ.float().unsqueeze(1))
    c = castling.long()
    wk, wq, bk, bq = c & 1, (c >> 1) & 1, (c >> 2) & 1, (c >> 3) & 1
    ok = torch.where(black, bk, wk); oq = torch.where(black, bq, wq)
    tk = torch.where(black, wk, bk); tq = torch.where(black, wq, bq)
    for i, v in enumerate((ok, oq, tk, tq)):
        x[:, 12 + i] = v.float().unsqueeze(1)
    x[:, 16] = ((mover_elo.float() - 1500) / 500).unsqueeze(1)
    x[:, 17] = ((opp_elo.float() - 1500) / 500).unsqueeze(1)
    return x.view(B, PLANES, 8, 8), black


def target_index(move, black):
    f = (move // 64).long()
    t = (move % 64).long()
    flip = black.long() * 56
    return (f ^ flip) * 64 + (t ^ flip)


def fold(conv, bn):
    w = conv.weight.detach().double()
    b = conv.bias.detach().double() if conv.bias is not None else torch.zeros(w.shape[0], dtype=torch.float64, device=w.device)
    if bn is not None:
        scale = bn.weight.detach().double() / torch.sqrt(bn.running_var.detach().double() + bn.eps)
        w = w * scale.view(-1, 1, 1, 1)
        b = (b - bn.running_mean.detach().double()) * scale + bn.bias.detach().double()
    return w.reshape(w.shape[0], -1).float(), b.float()


def export(model, path, blocks, channels):
    """'POL2', int32 planes, blocks, channels, then float32 tensors (BatchNorm folded):
       stem W[C][18*9] b[C]; per block conv1 W[C][C*9] b[C], conv2 W b;
       head W[C][C*9] b[C]; out W[64][C] b[64]."""
    model = model.float().eval()
    tensors = list(fold(model.stem, model.stem_bn))
    for blk in model.blocks:
        tensors += fold(blk.c1, blk.b1)
        tensors += fold(blk.c2, blk.b2)
    tensors += fold(model.head, model.head_bn)
    tensors += fold(model.out, None)
    with open(path, "wb") as f:
        f.write(b"POL2")
        f.write(struct.pack("<iii", PLANES, blocks, channels))
        for t in tensors:
            f.write(t.contiguous().cpu().numpy().astype("<f4").tobytes())


REF_CASES = [
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 1200, 1200),
    ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", 1500, 1600),
    ("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 5", 2400, 2300),
    ("8/5pk1/6p1/8/3R4/6P1/5PK1/1r6 b - - 0 40", 900, 1000),
]


def write_reference(model, path):
    """Reference logits for `engine --selftest` (checks the C++ forward pass)."""
    import chess
    dev = next(model.parameters()).device
    code = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6, "p": 7, "n": 8, "b": 9, "r": 10, "q": 11, "k": 12}
    model.eval()
    tf32 = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False   # full fp32 reference
    with open(path, "w") as f, torch.no_grad():
        for fen, me, opp in REF_CASES:
            b = chess.Board(fen)
            sq = np.zeros(64, np.uint8)
            for s_, pc in b.piece_map().items():
                sq[s_] = code[pc.symbol()]
            packed = torch.from_numpy((sq[0::2] | (sq[1::2] << 4))[None]).to(dev)
            cast = (b.has_kingside_castling_rights(chess.WHITE) | b.has_queenside_castling_rights(chess.WHITE) << 1 |
                    b.has_kingside_castling_rights(chess.BLACK) << 2 | b.has_queenside_castling_rights(chess.BLACK) << 3)
            x, _ = make_input(packed, torch.tensor([1 if b.turn else 0], device=dev),
                              torch.tensor([cast], device=dev), torch.tensor([me], device=dev),
                              torch.tensor([opp], device=dev))
            logits = model(x)[0].float().cpu().numpy()
            top = np.argsort(-logits)[:5]
            f.write(f"{fen}|{me}|{opp}\n" + " ".join(f"{i}:{logits[i]:.5f}" for i in top) + "\n")
    torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = tf32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "policy"))
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--channels", type=int, default=96)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val", type=int, default=400_000)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N positions")
    ap.add_argument("--export-only", action="store_true", help="re-export the checkpoint and exit")
    ap.add_argument("--out", default=os.path.join(ROOT, "assets", "policy_elo.bin"))
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "data", "policy_ckpt.pt"))
    args = ap.parse_args()

    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    mp, bp = os.path.join(args.data, "meta.bin"), os.path.join(args.data, "boards.u8")
    n = min(os.path.getsize(mp) // META.itemsize, os.path.getsize(bp) // 32)
    if args.limit:
        n = min(n, args.limit)
    meta = np.fromfile(mp, META, count=n)
    boards = np.fromfile(os.path.join(args.data, "boards.u8"), np.uint8, count=n * 32).reshape(n, 32)
    print(f"{n:,} positions")
    perm = np.random.default_rng(1).permutation(n)
    boards_g = torch.from_numpy(boards[perm]).to(dev)
    meta = meta[perm]
    stm_g = torch.from_numpy(meta["stm"].copy()).to(dev)
    cast_g = torch.from_numpy(meta["castling"].copy()).to(dev)
    move_g = torch.from_numpy(meta["move"].astype(np.int32)).to(dev)
    me_g = torch.from_numpy(meta["mover_elo"].copy()).to(dev)
    oe_g = torch.from_numpy(meta["opp_elo"].copy()).to(dev)
    del boards, perm, meta

    nv = args.val
    train_n = n - nv
    model = PolicyNet(args.blocks, args.channels).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = train_n // args.batch
    total = int(args.epochs * steps_per_epoch)
    warm = 1000
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))))

    step = 0
    if os.path.exists(args.ckpt):
        st = torch.load(args.ckpt, map_location=dev)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        step = st["step"]
        print(f"resumed at step {step}")
    if args.export_only:
        export(model, args.out, args.blocks, args.channels)
        write_reference(model, os.path.splitext(args.out)[0] + "_test.txt")
        print(f"exported {args.out}")
        return

    def batch(sl):
        x, black = make_input(boards_g[sl], stm_g[sl], cast_g[sl], me_g[sl], oe_g[sl])
        return x.contiguous(memory_format=torch.channels_last), target_index(move_g[sl], black)

    def validate():
        model.eval()
        bands = {}
        correct = total_n = 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(train_n, n, 8192):
                sl = slice(i, min(n, i + 8192))
                x, y = batch(sl)
                hit = (model(x).argmax(1) == y)
                band = (me_g[sl].long() // 200) * 200
                for b in band.unique().tolist():
                    m = band == b
                    c, t = bands.get(b, (0, 0))
                    bands[b] = (c + int(hit[m].sum()), t + int(m.sum()))
                correct += int(hit.sum()); total_n += len(y)
        model.train()
        return correct / total_n, {b: c / t for b, (c, t) in sorted(bands.items()) if t > 2000}

    print(f"{args.blocks} blocks x {args.channels} ch; {total:,} steps ({args.epochs} epochs of {steps_per_epoch:,})")
    t0 = time.time()
    order = torch.randperm(train_n, device=dev)
    run = 0.0
    while step < total:
        k = step % steps_per_epoch
        if k == 0 and step > 0:
            order = torch.randperm(train_n, device=dev)
        idx = order[k * args.batch:(k + 1) * args.batch]
        x, y = batch(idx)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(x).float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        step += 1
        if step % 50 == 0:
            run = 0.98 * run + 0.02 * loss.item() if run else loss.item()
        if step % 5000 == 0 or step == total:
            acc, bands = validate()
            band_txt = " ".join(f"{b}:{a:.1%}" for b, a in bands.items())
            print(f"step {step:,}/{total:,}  loss {run:.3f}  val top-1 {acc:.2%}  [{band_txt}]  "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step}, args.ckpt)
            export(model, args.out, args.blocks, args.channels)
            write_reference(model, os.path.splitext(args.out)[0] + "_test.txt")
            model.train()
    print(f"done; wrote {args.out}")


if __name__ == "__main__":
    main()
