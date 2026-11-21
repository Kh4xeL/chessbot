"""
Trains the NNUE evaluator on the data produced by prepare_data.py.

Network ("768 -> 2x512 -> 1", SCReLU), the same shape modern hobby engines use:
    feature transformer: 768 one-hot piece-square inputs -> 512, shared weights,
                         evaluated from both the side-to-move's and the
                         opponent's perspective (board mirrored for black)
    output:              concat(us, them) = 1024 -> 1
Prediction is in units of (centipawns / 400). Loss is MSE between
sigmoid(prediction) and sigmoid(stockfish_cp / 400), i.e. on win-probability
scale, so +1500 vs +2000 matters far less than +0 vs +500.

The whole dataset is loaded onto the GPU (32 bytes per position), so training is
compute bound rather than I/O bound.

Usage:  python training/train_nnue.py --epochs 12
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
HIDDEN = 512
SCALE = 400.0
QA = 255          # feature transformer quantization
CLIP = 1.98       # keeps quantized accumulators far from int16 overflow


class NNUE(torch.nn.Module):
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.ft = torch.nn.Linear(768, hidden)
        self.out = torch.nn.Linear(2 * hidden, 1)
        with torch.no_grad():
            self.ft.weight.mul_(0.5)

    def forward(self, oh_us, oh_them):
        us = torch.clamp(self.ft(oh_us), 0, 1) ** 2
        them = torch.clamp(self.ft(oh_them), 0, 1) ** 2
        return self.out(torch.cat([us, them], dim=1)).squeeze(1)

    def clip(self):
        for p in self.parameters():
            p.data.clamp_(-CLIP, CLIP)


def features(packed, stm):
    """packed: [B,32] uint8 on GPU, stm: [B] (1 = white). Returns dense one-hot
    inputs from the side-to-move's and the opponent's perspective."""
    lo = (packed & 15).long()
    hi = (packed >> 4).long()
    pc = torch.stack([lo, hi], dim=2).reshape(-1, 64)          # [B,64], a1 = 0
    sq = torch.arange(64, device=packed.device).expand_as(pc)
    occ = pc > 0
    t = (pc - 1).clamp(min=0) % 6
    col = (pc - 1).clamp(min=0) // 6                            # 0 white, 1 black
    idx_w = torch.where(occ, col * 384 + t * 64 + sq, 768)
    idx_b = torch.where(occ, (1 - col) * 384 + t * 64 + (sq ^ 56), 768)

    B = pc.shape[0]
    oh_w = torch.zeros(B, 769, device=packed.device)
    oh_b = torch.zeros(B, 769, device=packed.device)
    oh_w.scatter_(1, idx_w, 1.0)
    oh_b.scatter_(1, idx_b, 1.0)
    oh_w, oh_b = oh_w[:, :768], oh_b[:, :768]
    white = (stm == 1).unsqueeze(1)
    return torch.where(white, oh_w, oh_b), torch.where(white, oh_b, oh_w)


def export(model, path):
    """Quantized little-endian layout read by the C++ engine:
       magic 'NNU1', int32 hidden,
       int16 ft_weight[768][hidden] (x QA), int16 ft_bias[hidden] (x QA),
       float out_weight[2*hidden], float out_bias."""
    ftw = model.ft.weight.detach().cpu().t().contiguous()      # [768, hidden]
    ftb = model.ft.bias.detach().cpu()
    ow = model.out.weight.detach().cpu().view(-1)
    ob = model.out.bias.detach().cpu()
    with open(path, "wb") as f:
        f.write(b"NNU1")
        f.write(struct.pack("<i", ftw.shape[1]))
        f.write(torch.round(ftw * QA).to(torch.int16).numpy().astype("<i2").tobytes())
        f.write(torch.round(ftb * QA).to(torch.int16).numpy().astype("<i2").tobytes())
        f.write(ow.numpy().astype("<f4").tobytes())
        f.write(ob.numpy().astype("<f4").tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--epochs", type=float, default=12)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val", type=int, default=500_000)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N positions")
    args = ap.parse_args()

    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True

    stm = np.fromfile(os.path.join(args.data, "stm.u8"), np.uint8)
    n = len(stm) if not args.limit else min(args.limit, len(stm))
    boards = np.fromfile(os.path.join(args.data, "boards.u8"), np.uint8, count=n * 32).reshape(n, 32)
    scores = np.fromfile(os.path.join(args.data, "scores.i16"), np.int16, count=n)
    stm = stm[:n]
    print(f"{n:,} positions")

    # Shuffle once on the CPU so the validation split is a random sample.
    perm = np.random.default_rng(0).permutation(n)
    boards_g = torch.from_numpy(boards[perm]).to(dev)
    stm_g = torch.from_numpy(stm[perm]).to(dev)
    cp_white = torch.from_numpy(scores[perm].astype(np.float32)).to(dev)
    cp_stm = torch.where(stm_g == 1, cp_white, -cp_white)
    target_g = torch.sigmoid(cp_stm / SCALE)
    del boards, perm

    nv = args.val
    train_n = n - nv
    model = NNUE().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps_per_epoch = train_n // args.batch
    total_steps = int(args.epochs * steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))

    ckpt = os.path.join(args.data, "nnue_ckpt.pt")
    step = 0
    if os.path.exists(ckpt):
        st = torch.load(ckpt, map_location=dev)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        step = st["step"]
        print(f"resumed at step {step}")

    def validate():
        model.eval()
        tot, mae_cp = 0.0, 0.0
        with torch.no_grad():
            for i in range(train_n, n, 65536):
                j = min(n, i + 65536)
                us, them = features(boards_g[i:j], stm_g[i:j])
                p = model(us, them)
                tot += ((torch.sigmoid(p) - target_g[i:j]) ** 2).sum().item()
                mae_cp += (p * SCALE - cp_stm[i:j]).clamp(-SCALE * 4, SCALE * 4).abs().sum().item()
        model.train()
        return tot / nv, mae_cp / nv

    print(f"{total_steps:,} steps ({args.epochs} epochs of {steps_per_epoch:,})")
    t0 = time.time()
    run = 0.0
    order = torch.randperm(train_n, device=dev)
    while step < total_steps:
        k = step % steps_per_epoch
        if k == 0 and step > 0:
            order = torch.randperm(train_n, device=dev)
        idx = order[k * args.batch:(k + 1) * args.batch]
        us, them = features(boards_g[idx], stm_g[idx])
        pred = model(us, them)
        loss = ((torch.sigmoid(pred) - target_g[idx]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        model.clip()
        run += loss.item() if step % 100 == 0 else 0.0
        step += 1

        if step % 2000 == 0 or step == total_steps:
            vl, mae = validate()
            el = time.time() - t0
            print(f"step {step:,}/{total_steps:,}  epoch {step / steps_per_epoch:.2f}  "
                  f"train {run / 20:.5f}  val {vl:.5f}  val |err| {mae:.0f}cp  "
                  f"lr {sched.get_last_lr()[0]:.2e}  {el / 60:.1f} min", flush=True)
            run = 0.0
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "step": step}, ckpt)
            export(model, os.path.join(ROOT, "assets", "nnue.bin"))

    print("done; wrote assets/nnue.bin")


if __name__ == "__main__":
    main()
