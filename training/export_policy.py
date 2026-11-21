"""
Exports the policy half of the original CNN (assets/traced_alphazero.pt) into a
flat float32 file that the C++ engine reads directly, so the engine no longer
needs LibTorch. BatchNorm layers are folded into the preceding convolutions.

Layout of assets/policy_cnn.bin (all little-endian float32, in order):
    conv_input  W[128][13*9]  b[128]
    4 x res block: conv1 W[128][128*9] b[128], conv2 W[128][128*9] b[128]
    pol_conv    W[2][128]     b[2]
    pol_fc      W[4096][128]  b[4096]

Also writes assets/policy_cnn_test.txt: a few positions with their reference
logits so the C++ implementation can be checked against PyTorch.

Usage:  python training/export_policy.py
"""

import os
import struct

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPS = 1e-5


def fold(sd, conv, bn):
    w = sd[f"{conv}.weight"].double()
    b = sd.get(f"{conv}.bias")
    b = b.double() if b is not None else torch.zeros(w.shape[0], dtype=torch.float64)
    g, beta = sd[f"{bn}.weight"].double(), sd[f"{bn}.bias"].double()
    mean, var = sd[f"{bn}.running_mean"].double(), sd[f"{bn}.running_var"].double()
    scale = g / torch.sqrt(var + EPS)
    w = w * scale.view(-1, 1, 1, 1)
    b = (b - mean) * scale + beta
    return w.reshape(w.shape[0], -1).float(), b.float()


def board_tensor(fen):
    import chess
    board = chess.Board(fen)
    x = torch.zeros(1, 13, 8, 8)
    for sq, p in board.piece_map().items():
        ch = (p.piece_type - 1) + (0 if p.color == chess.WHITE else 6)
        rank, file = sq // 8, sq % 8
        x[0, ch, 7 - rank, file] = 1.0
    x[0, 12] = 1.0 if board.turn == chess.WHITE else 0.0
    return x


def main():
    model = torch.jit.load(os.path.join(ROOT, "assets", "traced_alphazero.pt")).eval()
    sd = {k: v for k, v in model.state_dict().items()}

    tensors = [*fold(sd, "conv_input", "bn_input")]
    for i in range(4):
        tensors += fold(sd, f"res_blocks.{i}.conv1", f"res_blocks.{i}.bn1")
        tensors += fold(sd, f"res_blocks.{i}.conv2", f"res_blocks.{i}.bn2")
    tensors += fold(sd, "pol_conv", "pol_bn")
    tensors += [sd["pol_fc.weight"].float(), sd["pol_fc.bias"].float()]

    out = os.path.join(ROOT, "assets", "policy_cnn.bin")
    with open(out, "wb") as f:
        for t in tensors:
            f.write(t.contiguous().numpy().astype("<f4").tobytes())
    print(f"wrote {out} ({os.path.getsize(out):,} bytes)")

    # Reference logits for the C++ self-test.
    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        "8/5pk1/6p1/8/3R4/6P1/5PK1/1r6 b - - 0 40",
    ]
    elo = torch.tensor([[2500 / 3000.0]])
    with torch.no_grad(), open(os.path.join(ROOT, "assets", "policy_cnn_test.txt"), "w") as f:
        for fen in fens:
            pol, _ = model(board_tensor(fen), elo)
            pol = pol[0].numpy()
            top = np.argsort(-pol)[:5]
            f.write(fen + "\n")
            f.write(" ".join(f"{i}:{pol[i]:.5f}" for i in top) + "\n")
    print("wrote assets/policy_cnn_test.txt")


if __name__ == "__main__":
    main()
