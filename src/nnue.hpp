#pragma once
/**
 * NNUE evaluator (format "NNU2", trained by training/train_nnue.py).
 *
 *   feature transformer: king-bucketed piece-square inputs -> H (int16, x QA)
 *   output:              SCReLU(concat(us, them)) -> 1, one of n_ob output buckets
 *                        chosen by the number of pieces on the board
 *
 * Each perspective (White's, Black's) sees the board from its own side (Black's
 * view is flipped vertically). If mirroring is on and that side's king stands on
 * files e-h, the board is also mirrored left-right, and the king's square selects
 * one of n_kb input buckets through `layout`. Feature index:
 *     bucket * 768 + (piece is the perspective's own ? 0 : 384) + type * 64 + square'
 *
 * Accumulators are updated incrementally along the search path. When a king move
 * changes its own side's bucket or mirroring, that perspective is recomputed from
 * scratch after the move (Stack::finish). The old "NNU1" format (one bucket, no
 * mirroring, one output) still loads.
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "../include/chess.hpp"

namespace nnue {

constexpr int MAX_HIDDEN = 1024;
constexpr int QA = 255;
constexpr float SCALE = 400.0f;
constexpr int MAX_PLY = 256;

struct Network {
    int hidden = 0;
    int n_kb = 1;          // king buckets
    int n_ob = 1;          // output buckets
    bool mirror = false;
    std::uint8_t layout[32] = {};
    std::vector<std::int16_t> ft_w;   // [n_kb * 768][hidden]
    std::vector<std::int16_t> ft_b;   // [hidden]
    std::vector<float> out_w;         // [n_ob][2 * hidden]
    std::vector<float> out_b;         // [n_ob]

    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        char magic[4];
        f.read(magic, 4);
        std::int32_t h = 0;
        if (std::memcmp(magic, "NNU1", 4) == 0) {
            f.read(reinterpret_cast<char*>(&h), 4);
            n_kb = n_ob = 1;
            mirror = false;
            std::memset(layout, 0, sizeof(layout));
        } else if (std::memcmp(magic, "NNU2", 4) == 0) {
            std::int32_t kb = 0, ob = 0;
            std::uint8_t mir = 0;
            f.read(reinterpret_cast<char*>(&h), 4);
            f.read(reinterpret_cast<char*>(&kb), 4);
            f.read(reinterpret_cast<char*>(&ob), 4);
            f.read(reinterpret_cast<char*>(&mir), 1);
            f.read(reinterpret_cast<char*>(layout), 32);
            if (kb < 1 || kb > 64 || ob < 1 || ob > 32) return false;
            n_kb = kb;
            n_ob = ob;
            mirror = mir != 0;
            for (auto b : layout)
                if (b >= n_kb) return false;
        } else {
            return false;
        }
        if (h <= 0 || h > MAX_HIDDEN) return false;
        hidden = h;
        ft_w.resize(static_cast<size_t>(n_kb) * 768 * hidden);
        ft_b.resize(hidden);
        out_w.resize(static_cast<size_t>(n_ob) * 2 * hidden);
        out_b.resize(n_ob);
        f.read(reinterpret_cast<char*>(ft_w.data()), ft_w.size() * 2);
        f.read(reinterpret_cast<char*>(ft_b.data()), ft_b.size() * 2);
        f.read(reinterpret_cast<char*>(out_w.data()), out_w.size() * 4);
        f.read(reinterpret_cast<char*>(out_b.data()), out_b.size() * 4);
        return static_cast<bool>(f);
    }
};

inline Network NET;

// How one perspective sees the board: its king bucket and the file flip (0 or 7).
struct View {
    int bucket = 0;
    int flip = 0;
    bool operator==(const View& o) const { return bucket == o.bucket && flip == o.flip; }
};

inline View view_for(int persp, int king_sq) {
    int ks = persp == 0 ? king_sq : king_sq ^ 56;
    View v;
    v.flip = (NET.mirror && (ks & 7) >= 4) ? 7 : 0;
    ks ^= v.flip;
    // The layout covers files a-d (mirrored nets never see e-h). Unmirrored nets have a
    // single bucket, so folding e-h onto a-d only keeps the index inside the table.
    v.bucket = NET.layout[(ks >> 3) * 4 + (ks & 3)];
    return v;
}

struct Accumulator {
    std::int16_t v[2][MAX_HIDDEN];  // [perspective: 0 white, 1 black][hidden]
    View view[2];
    bool dirty[2] = {false, false};
};

inline int feature(int persp, const View& view, int color, int type, int sq) {
    const int s = (persp == 0 ? sq : sq ^ 56) ^ view.flip;
    return view.bucket * 768 + (color == persp ? 0 : 384) + type * 64 + s;
}

inline void add_row(std::int16_t* acc, int feat) {
    const std::int16_t* w = &NET.ft_w[static_cast<size_t>(feat) * NET.hidden];
    for (int i = 0; i < NET.hidden; i++) acc[i] += w[i];
}
inline void sub_row(std::int16_t* acc, int feat) {
    const std::int16_t* w = &NET.ft_w[static_cast<size_t>(feat) * NET.hidden];
    for (int i = 0; i < NET.hidden; i++) acc[i] -= w[i];
}

inline void add_piece(Accumulator& a, int color, int type, int sq) {
    for (int p = 0; p < 2; p++)
        if (!a.dirty[p]) add_row(a.v[p], feature(p, a.view[p], color, type, sq));
}
inline void sub_piece(Accumulator& a, int color, int type, int sq) {
    for (int p = 0; p < 2; p++)
        if (!a.dirty[p]) sub_row(a.v[p], feature(p, a.view[p], color, type, sq));
}

inline chess::Color color_of(int c) { return c == 0 ? chess::Color::WHITE : chess::Color::BLACK; }

// Recompute one perspective from scratch.
inline void refresh_persp(Accumulator& a, const chess::Board& board, int p) {
    a.view[p] = view_for(p, board.kingSq(color_of(p)).index());
    a.dirty[p] = false;
    std::memcpy(a.v[p], NET.ft_b.data(), NET.hidden * 2);
    for (int c = 0; c < 2; c++) {
        for (int t = 0; t < 6; t++) {
            auto bb = board.pieces(chess::PieceType(static_cast<chess::PieceType::underlying>(t)), color_of(c));
            while (bb) add_row(a.v[p], feature(p, a.view[p], c, t, bb.pop()));
        }
    }
}

inline void refresh(Accumulator& a, const chess::Board& board) {
    refresh_persp(a, board, 0);
    refresh_persp(a, board, 1);
}

// Side-to-move relative evaluation in centipawns.
inline int output(const Accumulator& a, bool white_to_move, int piece_count) {
    const std::int16_t* us = a.v[white_to_move ? 0 : 1];
    const std::int16_t* them = a.v[white_to_move ? 1 : 0];
    const int H = NET.hidden;
    const int ob = std::clamp((piece_count - 2) / 4, 0, NET.n_ob - 1);
    const float* w = &NET.out_w[static_cast<size_t>(ob) * 2 * H];
    float s = 0.0f;
#ifdef __AVX2__
    if (H % 16 == 0) {
        // SCReLU dot product, 16 neurons per step with 4 independent accumulators.
        const __m256i lo = _mm256_setzero_si256(), hi = _mm256_set1_epi16(QA);
        __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
        for (int side = 0; side < 2; side++) {
            const std::int16_t* x = side == 0 ? us : them;
            const float* ws = w + side * H;
            for (int i = 0; i < H; i += 16) {
                __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i));
                v = _mm256_min_epi16(_mm256_max_epi16(v, lo), hi);
                const __m256i a = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v));
                const __m256i b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v, 1));
                const __m256 fa = _mm256_cvtepi32_ps(_mm256_mullo_epi32(a, a));
                const __m256 fb = _mm256_cvtepi32_ps(_mm256_mullo_epi32(b, b));
                const int k = (i >> 3) & 2;
                acc[k] = _mm256_fmadd_ps(fa, _mm256_loadu_ps(ws + i), acc[k]);
                acc[k + 1] = _mm256_fmadd_ps(fb, _mm256_loadu_ps(ws + i + 8), acc[k + 1]);
            }
        }
        const __m256 t = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
        __m128 r = _mm_add_ps(_mm256_castps256_ps128(t), _mm256_extractf128_ps(t, 1));
        r = _mm_add_ps(r, _mm_movehl_ps(r, r));
        r = _mm_add_ss(r, _mm_shuffle_ps(r, r, 1));
        s = _mm_cvtss_f32(r);
    } else
#endif
    {
        for (int i = 0; i < H; i++) {
            int x = std::clamp<int>(us[i], 0, QA);
            s += static_cast<float>(x * x) * w[i];
        }
        for (int i = 0; i < H; i++) {
            int x = std::clamp<int>(them[i], 0, QA);
            s += static_cast<float>(x * x) * w[H + i];
        }
    }
    float out = s / static_cast<float>(QA * QA) + NET.out_b[ob];
    // Keep far away from the search's mate-score range.
    return std::clamp(static_cast<int>(out * SCALE), -20000, 20000);
}

/**
 * Incrementally maintained accumulator stack:
 *   push(board, move) BEFORE board.makeMove(move), finish(board) right after it,
 *   pop() after board.unmakeMove(move).
 */
struct Stack {
    std::vector<Accumulator> acc = std::vector<Accumulator>(MAX_PLY + 4);
    int top = 0;

    void reset(const chess::Board& board) {
        top = 0;
        refresh(acc[0], board);
    }

    void push(const chess::Board& board, const chess::Move& m) {
        Accumulator& next = acc[top + 1];
        const Accumulator& cur = acc[top];
        std::memcpy(next.v[0], cur.v[0], NET.hidden * 2);
        std::memcpy(next.v[1], cur.v[1], NET.hidden * 2);
        next.view[0] = cur.view[0];
        next.view[1] = cur.view[1];
        next.dirty[0] = next.dirty[1] = false;
        top++;

        const int us = board.sideToMove() == chess::Color::WHITE ? 0 : 1;
        const int them = 1 - us;
        const int from = m.from().index();
        const int to = m.to().index();
        const int moved = static_cast<int>(board.at<chess::PieceType>(m.from()));

        // A king move that changes its own side's bucket or mirroring invalidates that
        // whole perspective: skip its deltas and recompute it in finish().
        if (moved == 5) {
            const int king_to = m.typeOf() == chess::Move::CASTLING ? (from & 56) + (to > from ? 6 : 2) : to;
            if (!(view_for(us, king_to) == cur.view[us])) next.dirty[us] = true;
        }

        if (m.typeOf() == chess::Move::CASTLING) {
            // chess.hpp encodes castling as "king takes own rook".
            const bool king_side = to > from;
            const int rank_base = from & 56;
            const int king_to = rank_base + (king_side ? 6 : 2);
            const int rook_to = rank_base + (king_side ? 5 : 3);
            sub_piece(next, us, 5, from);
            sub_piece(next, us, 3, to);
            add_piece(next, us, 5, king_to);
            add_piece(next, us, 3, rook_to);
            return;
        }
        if (m.typeOf() == chess::Move::ENPASSANT) {
            sub_piece(next, them, 0, to ^ 8);
        } else {
            const auto captured = board.at<chess::PieceType>(m.to());
            if (captured != chess::PieceType::NONE) sub_piece(next, them, static_cast<int>(captured), to);
        }
        sub_piece(next, us, moved, from);
        const int placed = m.typeOf() == chess::Move::PROMOTION ? static_cast<int>(m.promotionType()) : moved;
        add_piece(next, us, placed, to);
    }

    // Call after board.makeMove(): recomputes perspectives a king move invalidated.
    void finish(const chess::Board& board) {
        Accumulator& a = acc[top];
        for (int p = 0; p < 2; p++)
            if (a.dirty[p]) refresh_persp(a, board, p);
    }

    void push_null() {
        Accumulator& next = acc[top + 1];
        const Accumulator& cur = acc[top];
        std::memcpy(next.v[0], cur.v[0], NET.hidden * 2);
        std::memcpy(next.v[1], cur.v[1], NET.hidden * 2);
        next.view[0] = cur.view[0];
        next.view[1] = cur.view[1];
        next.dirty[0] = next.dirty[1] = false;
        top++;
    }

    void pop() { top--; }

    int evaluate(const chess::Board& board) const {
        return output(acc[top], board.sideToMove() == chess::Color::WHITE, board.occ().count());
    }
};

}  // namespace nnue
