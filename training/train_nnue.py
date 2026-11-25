"""
Trains the NNUE evaluator.

Network ("NNU2"):
    feature transformer: king-bucketed piece-square inputs -> H, shared weights,
        evaluated from the side to move's and the opponent's perspective. Each
        perspective sees the board with its own king; if that king is on files e-h
        the board is mirrored left-right, and the king's square picks one of
        `n_kb` input buckets (KB_LAYOUT), so the same piece placement can mean
        different things depending on where the king stands.
    output: SCReLU(concat(us, them)) -> 1, with one of 8 output buckets chosen by
        the number of pieces on the board (opening / middlegame / endgame get their
        own final layer).
    `--arch v1` gives the original 768 -> 2x512 -> 1 net (1 bucket, no mirroring).

Targets: sigmoid(stockfish_cp / 400), blended with the game result where known
(positions from real games, see label_fens.py):  t = L * eval + (1 - L) * result.

The whole dataset lives on the GPU (32 bytes per position).

Usage:  python training/train_nnue.py --epochs 15
Writes: data/nnue_ckpt.pt (resumable) and assets/nnue.bin (quantized, for C++)
"""

import argparse
import math
import os
import struct
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCALE = 400.0
QA = 255          # feature transformer quantization
CLIP = 1.98       # keeps quantized accumulators far from int16 overflow

# King-bucket layout over the 32 squares a1-d8 (after mirroring), index = rank * 4 + file.
KB_LAYOUTS = {
    "v1": [0] * 32,
    "v2": [0, 0, 1, 1,          # rank 1: castled-queenside side | centre
           2, 2, 3, 3,          # rank 2
           4, 4, 4, 4, 4, 4, 4, 4,          # ranks 3-4
           5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],  # ranks 5-8
}


class NNUE(torch.nn.Module):
    def __init__(self, hidden, n_kb, n_ob):
        super().__init__()
        self.n_feat = n_kb * 768
        self.ft = torch.nn.EmbeddingBag(self.n_feat + 1, hidden, mode="sum", padding_idx=self.n_feat)
        self.ft_bias = torch.nn.Parameter(torch.zeros(hidden))
        self.out = torch.nn.Linear(2 * hidden, n_ob)
        with torch.no_grad():
            self.ft.weight.normal_(0, 0.05)
            self.ft.weight[self.n_feat].zero_()

    def forward(self, idx_us, idx_them, obucket):
        us = torch.clamp(self.ft(idx_us) + self.ft_bias, 0, 1) ** 2
        them = torch.clamp(self.ft(idx_them) + self.ft_bias, 0, 1) ** 2
        out = self.out(torch.cat([us, them], dim=1))
        ob = obucket.clamp(max=out.shape[1] - 1)   # 1-bucket (v1) nets: always bucket 0
        return out.gather(1, ob.unsqueeze(1)).squeeze(1)

    def clip(self):
        for p in self.parameters():
            p.data.clamp_(-CLIP, CLIP)
        self.ft.weight.data[self.n_feat].zero_()


def make_features(packed, stm, layout, mirror, n_feat):
    """packed [B,32] uint8 on GPU, stm [B] (1 = white). Returns feature indices [B,64]
    for the side to move and the opponent, and the output bucket per sample."""
    lo = (packed & 15).long()
    hi = (packed >> 4).long()
    pc = torch.stack([lo, hi], dim=2).reshape(-1, 64)               # [B,64], a1 = 0
    sq = torch.arange(64, device=packed.device).expand_as(pc)
    occ = pc > 0
    t = (pc - 1).clamp(min=0) % 6
    col = (pc - 1).clamp(min=0) // 6                                # 0 white, 1 black
    wk = (pc == 6).float().argmax(1, keepdim=True)
    bk = (pc == 12).float().argmax(1, keepdim=True)

    def persp(is_white):
        ks = wk if is_white else bk ^ 56
        s = sq if is_white else sq ^ 56
        rel = col if is_white else 1 - col
        if mirror:
            m = ((ks & 7) >= 4).long()
            ks = ks ^ (7 * m)
            s = s ^ (7 * m)
        bucket = layout[(ks >> 3) * 4 + (ks & 3)]   # files e-h only occur unmirrored (1 bucket)
        idx = bucket * 768 + rel * 384 + t * 64 + s
        return torch.where(occ, idx, n_feat)

    iw, ib = persp(True), persp(False)
    white = (stm == 1).unsqueeze(1)
    ob = ((occ.sum(1) - 2) // 4).clamp(0, 7)
    return torch.where(white, iw, ib), torch.where(white, ib, iw), ob


def export(model, path, hidden, layout, mirror, n_ob):
    """Little-endian layout read by src/nnue.hpp:
       'NNU2', int32 hidden, int32 n_kb, int32 n_ob, uint8 mirror, uint8 layout[32],
       int16 ft_weight[n_kb*768][hidden] (x QA), int16 ft_bias[hidden] (x QA),
       float out_weight[n_ob][2*hidden], float out_bias[n_ob]."""
    n_kb = max(layout) + 1
    ftw = model.ft.weight.detach().cpu()[: n_kb * 768]
    with open(path, "wb") as f:
        f.write(b"NNU2")
        f.write(struct.pack("<iiiB", hidden, n_kb, n_ob, 1 if mirror else 0))
        f.write(bytes(layout))
        f.write(torch.round(ftw * QA).to(torch.int16).numpy().astype("<i2").tobytes())
        f.write(torch.round(model.ft_bias.detach().cpu() * QA).to(torch.int16).numpy().astype("<i2").tobytes())
        f.write(model.out.weight.detach().cpu().numpy().astype("<f4").tobytes())
        f.write(model.out.bias.detach().cpu().numpy().astype("<f4").tobytes())


def load_dir(d, limit=0):
    stm = np.fromfile(os.path.join(d, "stm.u8"), np.uint8)
    # a generator stopped mid-write can leave the files a record apart: use the common length
    sizes = [len(stm), os.path.getsize(os.path.join(d, "boards.u8")) // 32,
             os.path.getsize(os.path.join(d, "scores.i16")) // 2]
    rp = os.path.join(d, "results.u8")
    if os.path.exists(rp):
        sizes.append(os.path.getsize(rp))
    n = min(sizes) if not limit else min(limit, *sizes)
    boards = np.fromfile(os.path.join(d, "boards.u8"), np.uint8, count=n * 32).reshape(n, 32)
    scores = np.fromfile(os.path.join(d, "scores.i16"), np.int16, count=n)
    rp = os.path.join(d, "results.u8")
    results = np.fromfile(rp, np.uint8, count=n) if os.path.exists(rp) else np.full(n, 255, np.uint8)
    return boards, scores, stm[:n], results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=[os.path.join(ROOT, "data"), os.path.join(ROOT, "data", "games_labeled")])
    ap.add_argument("--arch", choices=["v1", "v2"], default="v2")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=float, default=15)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wdl", type=float, default=0.3, help="weight of the game result where known")
    ap.add_argument("--val", type=int, default=500_000)
    ap.add_argument("--out", default=os.path.join(ROOT, "assets", "nnue.bin"))
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "data", "nnue_ckpt.pt"))
    args = ap.parse_args()

    dev = torch.device("cuda")
    layout_list = KB_LAYOUTS[args.arch]
    mirror = args.arch != "v1"
    n_ob = 8 if args.arch != "v1" else 1
    n_kb = max(layout_list) + 1
    layout = torch.tensor(layout_list, device=dev)

    parts = [load_dir(d) for d in args.data if os.path.exists(os.path.join(d, "stm.u8"))]
    boards = np.concatenate([p[0] for p in parts])
    scores = np.concatenate([p[1] for p in parts])
    stm = np.concatenate([p[2] for p in parts])
    results = np.concatenate([p[3] for p in parts])
    n_src = len(parts)
    del parts
    n = len(stm)
    print(f"{n:,} positions from {n_src} source(s); {int((results != 255).sum()):,} with game results")

    # Targets are built on the CPU; the GPU holds only boards, stm and targets (37 bytes per
    # position) plus the validation slice's raw scores.
    perm = np.random.default_rng(0).permutation(n)
    stm = stm[perm]
    white = stm == 1
    cp = scores[perm].astype(np.float32)
    cp_stm_np = np.where(white, cp, -cp)
    eval_np = (1 / (1 + np.exp(-cp_stm_np / SCALE))).astype(np.float32)
    res = results[perm]
    res_stm = np.where(white, res / 2, 1 - res / 2).astype(np.float32)
    target_np = np.where(res != 255, (1 - args.wdl) * eval_np + args.wdl * res_stm, eval_np).astype(np.float32)
    del cp, res, res_stm, scores, results

    nv = args.val
    train_n = n - nv
    boards_g = torch.from_numpy(boards[perm]).to(dev)
    del boards, perm
    stm_g = torch.from_numpy(stm).to(dev)
    target_g = torch.from_numpy(target_np).to(dev)
    eval_t = torch.from_numpy(eval_np[train_n:]).to(dev)      # validation slice only
    cp_stm = torch.from_numpy(cp_stm_np[train_n:]).to(dev)
    del stm, target_np, eval_np, cp_stm_np
    model = NNUE(args.hidden, n_kb, n_ob).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps_per_epoch = train_n // args.batch
    total_steps = int(args.epochs * steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))

    step = 0
    if os.path.exists(args.ckpt):
        st = torch.load(args.ckpt, map_location=dev)
        if st.get("arch") == args.arch:
            model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
            step = st["step"]
            print(f"resumed at step {step}")

    def batch(i, j=None, idx=None):
        b = boards_g[idx] if idx is not None else boards_g[i:j]
        s = stm_g[idx] if idx is not None else stm_g[i:j]
        return make_features(b, s, layout, mirror, model.n_feat)

    def validate():
        model.eval()
        tot, mae = 0.0, 0.0
        with torch.no_grad():
            for i in range(train_n, n, 65536):
                j = min(n, i + 65536)
                us, them, ob = batch(i, j)
                p = model(us, them, ob)
                tot += ((torch.sigmoid(p) - eval_t[i - train_n:j - train_n]) ** 2).sum().item()
                mae += (p * SCALE - cp_stm[i - train_n:j - train_n]).clamp(-SCALE * 4, SCALE * 4).abs().sum().item()
        model.train()
        return tot / nv, mae / nv

    print(f"arch {args.arch}: {n_kb} king buckets, {n_ob} output buckets, hidden {args.hidden}; "
          f"{total_steps:,} steps ({args.epochs} epochs of {steps_per_epoch:,})")
    t0 = time.time()
    run, run_n = 0.0, 0
    order = torch.randperm(train_n, device=dev, dtype=torch.int32)
    while step < total_steps:
        k = step % steps_per_epoch
        if k == 0 and step > 0:
            order = torch.randperm(train_n, device=dev, dtype=torch.int32)
        idx = order[k * args.batch:(k + 1) * args.batch]
        us, them, ob = batch(0, idx=idx)
        pred = model(us, them, ob)
        loss = ((torch.sigmoid(pred) - target_g[idx]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        model.clip()
        if step % 100 == 0:
            run += loss.item(); run_n += 1
        step += 1

        if step % 2000 == 0 or step == total_steps:
            vl, err = validate()
            print(f"step {step:,}/{total_steps:,}  epoch {step / steps_per_epoch:.2f}  "
                  f"train {run / max(1, run_n):.5f}  val(eval) {vl:.5f}  |err| {err:.0f}cp  "
                  f"lr {sched.get_last_lr()[0]:.2e}  {(time.time() - t0) / 60:.1f} min", flush=True)
            run, run_n = 0.0, 0
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step, "arch": args.arch}, args.ckpt)
            export(model, args.out, args.hidden, layout_list, mirror, n_ob)

    print(f"done; wrote {args.out}")


if __name__ == "__main__":
    main()
