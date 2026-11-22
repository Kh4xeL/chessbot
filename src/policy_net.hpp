#pragma once
/**
 * Move-prediction CNN ("policy"), evaluated in plain C++ once per move at the root.
 * Its probabilities order the root moves and, below full strength, steer which
 * reasonable move the bot picks, so weaker settings play like humans of that rating.
 *
 * Two weight formats (BatchNorm already folded into the convolutions):
 *
 * - assets/policy_elo.bin, "POL2" (training/train_policy.py): rating-aware net
 *   trained on Lichess games. Input is the board from the side to move's view
 *   (vertically flipped and colours swapped for Black), castling planes and both
 *   players' ratings; a convolutional head gives logits[from' * 64 + to'] in that
 *   flipped frame.
 *
 * - assets/policy_cnn.bin (headerless, training/export_policy.py): the original
 *   AlphaZero-style CNN's policy head, no ratings; squares numbered from a8.
 *
 * Square layout inside the planes: position = rank * 8 + file (row = rank).
 */

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "../include/chess.hpp"

namespace policy {

struct Net {
    int version = 0;          // 0 = not loaded, 1 = original CNN, 2 = rating-aware
    int planes = 13, blocks = 4, ch = 128;
    std::vector<float> data;
    const float* in_w = nullptr;  const float* in_b = nullptr;
    std::vector<const float*> blk_w, blk_b;     // 2 convs per block
    // v1 head
    const float* pc_w = nullptr;  const float* pc_b = nullptr;
    const float* fc_w = nullptr;  const float* fc_b = nullptr;
    // v2 head
    const float* hd_w = nullptr;  const float* hd_b = nullptr;
    const float* out_w = nullptr; const float* out_b = nullptr;
    bool loaded = false;

    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) return false;
        const size_t bytes = static_cast<size_t>(f.tellg());
        f.seekg(0);
        std::vector<char> raw(bytes);
        f.read(raw.data(), bytes);
        if (bytes > 16 && std::memcmp(raw.data(), "POL2", 4) == 0) {
            std::int32_t hdr[3];
            std::memcpy(hdr, raw.data() + 4, 12);
            planes = hdr[0]; blocks = hdr[1]; ch = hdr[2];
            if (planes != 18 || blocks < 1 || blocks > 40 || ch < 8 || ch > 512) return false;
            const size_t expect = (size_t)ch * planes * 9 + ch + (size_t)blocks * 2 * ((size_t)ch * ch * 9 + ch) +
                                  (size_t)ch * ch * 9 + ch + 64 * (size_t)ch + 64;
            if (bytes != 16 + expect * 4) return false;
            data.resize(expect);
            std::memcpy(data.data(), raw.data() + 16, expect * 4);
            version = 2;
        } else {
            planes = 13; blocks = 4; ch = 128;
            const size_t expect = (size_t)ch * 13 * 9 + ch + (size_t)blocks * 2 * ((size_t)ch * ch * 9 + ch) +
                                  2 * (size_t)ch + 2 + 4096 * 128 + 4096;
            if (bytes != expect * 4) return false;
            data.resize(expect);
            std::memcpy(data.data(), raw.data(), bytes);
            version = 1;
        }
        const float* p = data.data();
        auto take = [&p](size_t n) { const float* r = p; p += n; return r; };
        in_w = take((size_t)ch * planes * 9); in_b = take(ch);
        blk_w.clear(); blk_b.clear();
        for (int b = 0; b < blocks * 2; b++) { blk_w.push_back(take((size_t)ch * ch * 9)); blk_b.push_back(take(ch)); }
        if (version == 2) {
            hd_w = take((size_t)ch * ch * 9); hd_b = take(ch);
            out_w = take(64 * (size_t)ch); out_b = take(64);
        } else {
            pc_w = take(2 * (size_t)ch); pc_b = take(2);
            fc_w = take(4096 * 128); fc_b = take(4096);
        }
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

    static void relu(std::vector<float>& v) { for (auto& x : v) x = x > 0 ? x : 0; }

    std::vector<float> input(const chess::Board& board, int mover_elo, int opp_elo) const {
        std::vector<float> x(static_cast<size_t>(planes) * 64, 0.0f);
        const bool white = board.sideToMove() == chess::Color::WHITE;
        for (int c = 0; c < 2; c++)
            for (int ty = 0; ty < 6; ty++) {
                auto bb = board.pieces(chess::PieceType(static_cast<chess::PieceType::underlying>(ty)),
                                       c == 0 ? chess::Color::WHITE : chess::Color::BLACK);
                while (bb) {
                    const int sq = bb.pop();
                    if (version == 1) {
                        x[(c * 6 + ty) * 64 + (7 - sq / 8) * 8 + sq % 8] = 1.0f;
                    } else {
                        const bool own = (c == 0) == white;
                        x[((own ? 0 : 6) + ty) * 64 + (white ? sq : sq ^ 56)] = 1.0f;
                    }
                }
            }
        if (version == 1) {
            const float turn = white ? 1.0f : 0.0f;
            for (int i = 0; i < 64; i++) x[12 * 64 + i] = turn;
            return x;
        }
        const auto cr = board.castlingRights();
        using Side = chess::Board::CastlingRights::Side;
        const chess::Color us = board.sideToMove(), them = ~board.sideToMove();
        const float planes_v[6] = {
            cr.has(us, Side::KING_SIDE) ? 1.0f : 0.0f, cr.has(us, Side::QUEEN_SIDE) ? 1.0f : 0.0f,
            cr.has(them, Side::KING_SIDE) ? 1.0f : 0.0f, cr.has(them, Side::QUEEN_SIDE) ? 1.0f : 0.0f,
            (mover_elo - 1500) / 500.0f, (opp_elo - 1500) / 500.0f};
        for (int k = 0; k < 6; k++)
            for (int i = 0; i < 64; i++) x[(12 + k) * 64 + i] = planes_v[k];
        return x;
    }

    // Returns 4096 raw logits, indexed by move_index().
    std::vector<float> forward(const chess::Board& board, int mover_elo = 2000, int opp_elo = 2000) const {
        std::vector<float> x = input(board, mover_elo, opp_elo), col;
        std::vector<float> h(static_cast<size_t>(ch) * 64), t(h.size()), t2(h.size());
        conv3x3(x.data(), planes, in_w, in_b, h.data(), ch, col);
        relu(h);
        for (int b = 0; b < blocks; b++) {
            conv3x3(h.data(), ch, blk_w[2 * b], blk_b[2 * b], t.data(), ch, col);
            relu(t);
            conv3x3(t.data(), ch, blk_w[2 * b + 1], blk_b[2 * b + 1], t2.data(), ch, col);
            for (size_t i = 0; i < h.size(); i++) { float v = t2[i] + h[i]; h[i] = v > 0 ? v : 0; }
        }
        std::vector<float> logits(4096);
        if (version == 2) {
            conv3x3(h.data(), ch, hd_w, hd_b, t.data(), ch, col);
            relu(t);
            for (int to = 0; to < 64; to++)
                for (int from = 0; from < 64; from++) {
                    float s = out_b[to];
                    const float* wr = &out_w[static_cast<size_t>(to) * ch];
                    for (int k = 0; k < ch; k++) s += wr[k] * t[k * 64 + from];
                    logits[from * 64 + to] = s;
                }
            return logits;
        }
        float pf[128];
        for (int c = 0; c < 2; c++)
            for (int p = 0; p < 64; p++) {
                float s = pc_b[c];
                for (int k = 0; k < ch; k++) s += pc_w[c * ch + k] * h[k * 64 + p];
                pf[c * 64 + p] = s > 0 ? s : 0;
            }
        for (int o = 0; o < 4096; o++) {
            float s = fc_b[o];
            const float* wr = &fc_w[static_cast<size_t>(o) * 128];
            for (int k = 0; k < 128; k++) s += wr[k] * pf[k];
            logits[o] = s;
        }
        return logits;
    }

    // Logit index for a chess.hpp move (castling is stored as king-takes-rook).
    int move_index(const chess::Move& m, bool white_to_move) const {
        int from = m.from().index();
        int to = m.to().index();
        if (m.typeOf() == chess::Move::CASTLING) to = (from & 56) + (to > from ? 6 : 2);
        if (version == 1) return (from ^ 56) * 64 + (to ^ 56);
        const int flip = white_to_move ? 0 : 56;
        return (from ^ flip) * 64 + (to ^ flip);
    }
};

inline Net NET;

}  // namespace policy
