#pragma once
/**
 * NNUE evaluator ("768 -> 2xH -> 1", SCReLU).
 *
 * Trained by training/train_nnue.py on Stockfish evaluations from the Lichess
 * evaluation database. Replaces the old CNN at the leaves of the search: the CNN
 * cost ~1.5 ms per position, this costs about a microsecond.
 *
 * Feature index for a piece of color c, type t (pawn..king = 0..5) on square s
 * (a1 = 0), seen from perspective p:
 *     white perspective: c * 384 + t * 64 + s
 *     black perspective: (1 - c) * 384 + t * 64 + (s ^ 56)    (board mirrored)
 *
 * Accumulators are updated incrementally along the search path (push/pop), so a
 * leaf evaluation only touches the handful of rows that changed.
 */

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

#include "../include/chess.hpp"

namespace nnue {

constexpr int MAX_HIDDEN = 1024;
constexpr int QA = 255;
constexpr float SCALE = 400.0f;
constexpr int MAX_PLY = 256;

struct Network {
    int hidden = 0;
    std::vector<std::int16_t> ft_w;   // [768][hidden]
    std::vector<std::int16_t> ft_b;   // [hidden]
    std::vector<float> out_w;         // [2*hidden]
    float out_b = 0.0f;

    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;
        char magic[4];
        f.read(magic, 4);
        if (std::memcmp(magic, "NNU1", 4) != 0) return false;
        std::int32_t h = 0;
        f.read(reinterpret_cast<char*>(&h), 4);
        if (h <= 0 || h > MAX_HIDDEN) return false;
        hidden = h;
        ft_w.resize(768 * hidden);
        ft_b.resize(hidden);
        out_w.resize(2 * hidden);
        f.read(reinterpret_cast<char*>(ft_w.data()), ft_w.size() * 2);
        f.read(reinterpret_cast<char*>(ft_b.data()), ft_b.size() * 2);
        f.read(reinterpret_cast<char*>(out_w.data()), out_w.size() * 4);
        f.read(reinterpret_cast<char*>(&out_b), 4);
        return static_cast<bool>(f);
    }
};

inline Network NET;

struct alignas(64) Accumulator {
    std::int16_t v[2][MAX_HIDDEN];  // [perspective: 0 white, 1 black][hidden]
};

inline int feature(int persp, int color, int type, int sq) {
    return persp == 0 ? color * 384 + type * 64 + sq
                      : (1 - color) * 384 + type * 64 + (sq ^ 56);
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
    add_row(a.v[0], feature(0, color, type, sq));
    add_row(a.v[1], feature(1, color, type, sq));
}
inline void sub_piece(Accumulator& a, int color, int type, int sq) {
    sub_row(a.v[0], feature(0, color, type, sq));
    sub_row(a.v[1], feature(1, color, type, sq));
}

inline void refresh(Accumulator& a, const chess::Board& board) {
    for (int p = 0; p < 2; p++) std::memcpy(a.v[p], NET.ft_b.data(), NET.hidden * 2);
    for (int c = 0; c < 2; c++) {
        for (int t = 0; t < 6; t++) {
            auto bb = board.pieces(chess::PieceType(static_cast<chess::PieceType::underlying>(t)),
                                   c == 0 ? chess::Color::WHITE : chess::Color::BLACK);
            while (bb) add_piece(a, c, t, bb.pop());
        }
    }
}

// Side-to-move relative evaluation in centipawns.
inline int output(const Accumulator& a, bool white_to_move) {
    const std::int16_t* us = a.v[white_to_move ? 0 : 1];
    const std::int16_t* them = a.v[white_to_move ? 1 : 0];
    const int H = NET.hidden;
    const float* w = NET.out_w.data();
    float s = 0.0f;
    for (int i = 0; i < H; i++) {
        int x = std::clamp<int>(us[i], 0, QA);
        s += static_cast<float>(x * x) * w[i];
    }
    for (int i = 0; i < H; i++) {
        int x = std::clamp<int>(them[i], 0, QA);
        s += static_cast<float>(x * x) * w[H + i];
    }
    float out = s / static_cast<float>(QA * QA) + NET.out_b;
    // Keep far away from the search's mate-score range.
    return std::clamp(static_cast<int>(out * SCALE), -20000, 20000);
}

/**
 * Incrementally maintained accumulator stack. Call push(board, move) BEFORE
 * board.makeMove(move), and pop() after board.unmakeMove(move).
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
        top++;

        const int us = board.sideToMove() == chess::Color::WHITE ? 0 : 1;
        const int them = 1 - us;
        const int from = m.from().index();
        const int to = m.to().index();
        const int moved = static_cast<int>(board.at<chess::PieceType>(m.from()));

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
            const int cap_sq = to ^ 8;
            sub_piece(next, them, 0, cap_sq);
        } else {
            const auto captured = board.at<chess::PieceType>(m.to());
            if (captured != chess::PieceType::NONE) sub_piece(next, them, static_cast<int>(captured), to);
        }
        sub_piece(next, us, moved, from);
        const int placed = m.typeOf() == chess::Move::PROMOTION ? static_cast<int>(m.promotionType()) : moved;
        add_piece(next, us, placed, to);
    }

    void push_null() {
        Accumulator& next = acc[top + 1];
        std::memcpy(next.v[0], acc[top].v[0], NET.hidden * 2);
        std::memcpy(next.v[1], acc[top].v[1], NET.hidden * 2);
        top++;
    }

    void pop() { top--; }

    int evaluate(const chess::Board& board) const {
        return output(acc[top], board.sideToMove() == chess::Color::WHITE);
    }
};

}  // namespace nnue
