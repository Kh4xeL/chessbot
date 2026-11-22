#pragma once
/**
 * Syzygy endgame tablebases through Fathom (include/fathom, MIT).
 *
 * - In the search: win/draw/loss probes (tb_probe_wdl), which Fathom only answers
 *   right after a capture or pawn move (50-move counter at zero), the standard
 *   way engines use WDL tables.
 * - At the root: distance-to-zero probe (tb_probe_root) for every legal move, so
 *   the engine keeps only moves that preserve the theoretical result and, when
 *   winning, plays the move that makes progress fastest.
 *
 * Tables are looked up in --syzygy <dir>, $SYZYGY_PATH or tools/syzygy.
 */

#include <cstdint>
#include <string>
#include <vector>

#include "../include/chess.hpp"
#include "../include/fathom/tbprobe.h"

namespace tb {

inline bool enabled = false;

inline bool init(const std::string& path) {
    enabled = !path.empty() && tb_init(path.c_str()) && TB_LARGEST > 0;
    return enabled;
}

inline int largest() { return enabled ? static_cast<int>(TB_LARGEST) : 0; }

struct Bits {
    std::uint64_t white, black, kings, queens, rooks, bishops, knights, pawns;
    unsigned ep;
    bool turn;
};

inline Bits bits(const chess::Board& b) {
    using chess::PieceType;
    Bits x;
    x.white = b.us(chess::Color::WHITE).getBits();
    x.black = b.us(chess::Color::BLACK).getBits();
    x.kings = b.pieces(PieceType::KING).getBits();
    x.queens = b.pieces(PieceType::QUEEN).getBits();
    x.rooks = b.pieces(PieceType::ROOK).getBits();
    x.bishops = b.pieces(PieceType::BISHOP).getBits();
    x.knights = b.pieces(PieceType::KNIGHT).getBits();
    x.pawns = b.pieces(PieceType::PAWN).getBits();
    x.ep = b.enpassantSq() == chess::Square::NO_SQ ? 0u : static_cast<unsigned>(b.enpassantSq().index());
    x.turn = b.sideToMove() == chess::Color::WHITE;
    return x;
}

inline bool probeable(const chess::Board& b) {
    return enabled && b.occ().count() <= static_cast<int>(TB_LARGEST) &&
           !b.castlingRights().has(chess::Color::WHITE) && !b.castlingRights().has(chess::Color::BLACK);
}

// Search probe: TB_LOSS..TB_WIN from the side to move's view, or TB_RESULT_FAILED.
inline unsigned probe_wdl(const chess::Board& b) {
    if (!probeable(b) || b.halfMoveClock() != 0) return TB_RESULT_FAILED;
    const Bits x = bits(b);
    return tb_probe_wdl(x.white, x.black, x.kings, x.queens, x.rooks, x.bishops, x.knights, x.pawns, 0, 0, x.ep,
                        x.turn);
}

struct RootMove {
    chess::Move move;
    unsigned wdl;   // TB_LOSS..TB_WIN, side to move's view after playing it is already accounted for
    unsigned dtz;
};

struct RootInfo {
    std::vector<RootMove> moves;
    chess::Move best;     // Fathom's choice: keeps the result, fastest progress when winning
    unsigned wdl = TB_DRAW;
};

inline chess::Move to_move(const chess::Board& b, unsigned res) {
    const int from = TB_GET_FROM(res), to = TB_GET_TO(res), promo = TB_GET_PROMOTES(res);
    chess::Movelist legal;
    chess::movegen::legalmoves(legal, b);
    for (const auto& m : legal) {
        if (m.from().index() != from || m.to().index() != to) continue;
        if (m.typeOf() == chess::Move::PROMOTION) {
            static const chess::PieceType map[5] = {chess::PieceType::NONE, chess::PieceType::QUEEN,
                                                    chess::PieceType::ROOK, chess::PieceType::BISHOP,
                                                    chess::PieceType::KNIGHT};
            if (promo == TB_PROMOTES_NONE || m.promotionType() != map[promo]) continue;
        }
        return m;
    }
    return chess::Move(chess::Move::NO_MOVE);
}

// Root probe (not thread safe: call once per search, from the main thread).
inline bool probe_root(const chess::Board& b, RootInfo& info) {
    if (!probeable(b)) return false;
    const Bits x = bits(b);
    unsigned results[TB_MAX_MOVES];
    const unsigned res = tb_probe_root(x.white, x.black, x.kings, x.queens, x.rooks, x.bishops, x.knights, x.pawns,
                                       b.halfMoveClock(), 0, x.ep, x.turn, results);
    if (res == TB_RESULT_FAILED || res == TB_RESULT_CHECKMATE || res == TB_RESULT_STALEMATE) return false;
    info.wdl = TB_GET_WDL(res);
    info.best = to_move(b, res);
    info.moves.clear();
    for (int i = 0; i < TB_MAX_MOVES && results[i] != TB_RESULT_FAILED; i++) {
        const chess::Move m = to_move(b, results[i]);
        if (m != chess::Move(chess::Move::NO_MOVE)) info.moves.push_back({m, TB_GET_WDL(results[i]), TB_GET_DTZ(results[i])});
    }
    return info.best != chess::Move(chess::Move::NO_MOVE);
}

}  // namespace tb
