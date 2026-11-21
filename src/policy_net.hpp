#pragma once
/**
 * The original AlphaZero-style CNN, policy head only, in plain C++.
 *
 * Weights come from training/export_policy.py (BatchNorm folded into the convs),
 * so the engine no longer needs LibTorch. It runs once per move at the root
 * (~tens of ms): its "intuition" orders the root moves and, below full
 * strength, steers which reasonable move a human of the requested Elo would pick.
 *
 * Input encoding matches training: 13 planes x 64, plane index = piece
 * (white P..K = 0..5, black = 6..11) with position (7 - rank) * 8 + file, plane
 * 12 = 1 if white to move. Output move index = from' * 64 + to' where squares are
 * numbered from a8 (sq' = sq ^ 56).
 */

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "../include/chess.hpp"

namespace policy {

constexpr int CH = 128;
constexpr int BLOCKS = 4;

struct Net {
    std::vector<float> data;
    const float* in_w = nullptr;  const float* in_b = nullptr;
    const float* blk_w[BLOCKS][2] = {};  const float* blk_b[BLOCKS][2] = {};
    const float* pc_w = nullptr;  const float* pc_b = nullptr;
    const float* fc_w = nullptr;  const float* fc_b = nullptr;
    bool loaded = false;

    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) return false;
        const size_t bytes = static_cast<size_t>(f.tellg());
        const size_t expect = (CH * 13 * 9 + CH + BLOCKS * 2 * (CH * CH * 9 + CH) + 2 * CH + 2 + 4096 * 128 + 4096) * 4;
        if (bytes != expect) return false;
        data.resize(bytes / 4);
        f.seekg(0);
        f.read(reinterpret_cast<char*>(data.data()), bytes);
        const float* p = data.data();
        auto take = [&p](size_t n) { const float* r = p; p += n; return r; };
        in_w = take(CH * 13 * 9); in_b = take(CH);
        for (int b = 0; b < BLOCKS; b++)
            for (int k = 0; k < 2; k++) { blk_w[b][k] = take(CH * CH * 9); blk_b[b][k] = take(CH); }
        pc_w = take(2 * CH); pc_b = take(2);
        fc_w = take(4096 * 128); fc_b = take(4096);
        loaded = true;
        return true;
    }

    // 3x3 conv, padding 1, via im2col: out[o][p] = b[o] + sum_k W[o][k] * col[k][p]
    static void conv3x3(const float* in, int cin, const float* w, const float* b, float* out, int cout,
                        std::vector<float>& col) {
        col.assign(static_cast<size_t>(cin) * 9 * 64, 0.0f);
        for (int c = 0; c < cin; c++)
            for (int ky = 0; ky < 3; ky++)
                for (int kx = 0; kx < 3; kx++) {
                    float* dst = &col[(static_cast<size_t>(c) * 9 + ky * 3 + kx) * 64];
                    for (int y = 0; y < 8; y++) {
                        int sy = y + ky - 1;
                        if (sy < 0 || sy > 7) continue;
                        for (int x = 0; x < 8; x++) {
                            int sx = x + kx - 1;
                            if (sx < 0 || sx > 7) continue;
                            dst[y * 8 + x] = in[c * 64 + sy * 8 + sx];
                        }
                    }
                }
        const int K = cin * 9;
        for (int o = 0; o < cout; o++) {
            float acc[64];
            for (int p = 0; p < 64; p++) acc[p] = b[o];
            const float* wr = &w[static_cast<size_t>(o) * K];
            for (int k = 0; k < K; k++) {
                const float wk = wr[k];
                const float* cr = &col[static_cast<size_t>(k) * 64];
                for (int p = 0; p < 64; p++) acc[p] += wk * cr[p];
            }
            for (int p = 0; p < 64; p++) out[o * 64 + p] = acc[p];
        }
    }

    // Returns 4096 raw logits.
    std::vector<float> forward(const chess::Board& board) const {
        std::vector<float> x(13 * 64, 0.0f), h(CH * 64), t(CH * 64), t2(CH * 64), col;
        for (int c = 0; c < 2; c++)
            for (int ty = 0; ty < 6; ty++) {
                auto bb = board.pieces(chess::PieceType(static_cast<chess::PieceType::underlying>(ty)),
                                       c == 0 ? chess::Color::WHITE : chess::Color::BLACK);
                while (bb) {
                    int sq = bb.pop();
                    int rank = sq / 8, file = sq % 8;
                    x[(c * 6 + ty) * 64 + (7 - rank) * 8 + file] = 1.0f;
                }
            }
        const float turn = board.sideToMove() == chess::Color::WHITE ? 1.0f : 0.0f;
        for (int i = 0; i < 64; i++) x[12 * 64 + i] = turn;

        conv3x3(x.data(), 13, in_w, in_b, h.data(), CH, col);
        for (auto& v : h) v = v > 0 ? v : 0;
        for (int b = 0; b < BLOCKS; b++) {
            conv3x3(h.data(), CH, blk_w[b][0], blk_b[b][0], t.data(), CH, col);
            for (auto& v : t) v = v > 0 ? v : 0;
            conv3x3(t.data(), CH, blk_w[b][1], blk_b[b][1], t2.data(), CH, col);
            for (int i = 0; i < CH * 64; i++) { float v = t2[i] + h[i]; h[i] = v > 0 ? v : 0; }
        }
        float pf[128];
        for (int c = 0; c < 2; c++)
            for (int p = 0; p < 64; p++) {
                float s = pc_b[c];
                for (int k = 0; k < CH; k++) s += pc_w[c * CH + k] * h[k * 64 + p];
                pf[c * 64 + p] = s > 0 ? s : 0;
            }
        std::vector<float> logits(4096);
        for (int o = 0; o < 4096; o++) {
            float s = fc_b[o];
            const float* wr = &fc_w[static_cast<size_t>(o) * 128];
            for (int k = 0; k < 128; k++) s += wr[k] * pf[k];
            logits[o] = s;
        }
        return logits;
    }

    // Policy index for a chess.hpp move (castling is stored as king-takes-rook).
    static int move_index(const chess::Move& m) {
        int from = m.from().index();
        int to = m.to().index();
        if (m.typeOf() == chess::Move::CASTLING) to = (from & 56) + (to > from ? 6 : 2);
        return (from ^ 56) * 64 + (to ^ 56);
    }
};

inline Net NET;

}  // namespace policy
