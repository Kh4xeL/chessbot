#pragma once
/**
 * Engine self-checks, run with `engine --test` (also in CI).
 * Included at the end of main.cpp, so it can use the search internals directly.
 */

#include <cstring>
#include <iostream>
#include <random>
#include <string>

namespace tests {

inline int fails = 0;

inline void check(bool ok, const std::string& name) {
    std::cout << (ok ? "PASS  " : "FAIL  ") << name << "\n";
    if (!ok) fails++;
}

// The incrementally updated accumulators must always equal a full recompute,
// through captures, castling, en passant, promotions and back again.
inline void nnue_incremental() {
    const char* fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  // castling, ep, captures
        "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",                               // promotions
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",                             // en passant
    };
    std::mt19937 rng(12345);
    auto* st = new nnue::Stack();
    nnue::Accumulator ref;
    const size_t bytes = static_cast<size_t>(nnue::NET.hidden) * 2;
    int positions = 0;
    bool ok = true;
    for (const char* fen : fens) {
        for (int game = 0; game < 40 && ok; game++) {
            Board b(fen);
            st->reset(b);
            std::vector<Move> played;
            for (int ply = 0; ply < 120; ply++) {
                Movelist ml;
                chess::movegen::legalmoves(ml, b);
                if (ml.empty() || b.isHalfMoveDraw()) break;
                const Move m = ml[std::uniform_int_distribution<int>(0, ml.size() - 1)(rng)];
                st->push(b, m);
                b.makeMove(m);
                played.push_back(m);
                nnue::refresh(ref, b);
                positions++;
                if (std::memcmp(ref.v[0], st->acc[st->top].v[0], bytes) != 0 ||
                    std::memcmp(ref.v[1], st->acc[st->top].v[1], bytes) != 0) {
                    ok = false;
                    std::cout << "  mismatch after " << chess::uci::moveToUci(m) << " in " << b.getFen() << "\n";
                    break;
                }
            }
            while (ok && !played.empty()) {   // unmaking must restore the start position's accumulator
                b.unmakeMove(played.back());
                played.pop_back();
                st->pop();
            }
            if (ok) {
                nnue::refresh(ref, b);
                ok = std::memcmp(ref.v[0], st->acc[0].v[0], bytes) == 0;
            }
        }
    }
    delete st;
    check(ok, "NNUE incremental update == full refresh (" + std::to_string(positions) + " random positions)");
}

inline void see() {
    struct Case { const char* fen; const char* uci; int threshold; bool expect; const char* name; };
    const Case cases[] = {
        {"4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5", 100, true, "free pawn wins 100"},
        {"4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5", 101, false, "free pawn wins no more than 100"},
        {"4k3/8/2p5/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5", 0, true, "pawn trade is even"},
        {"4k3/8/2p5/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5", 1, false, "pawn trade gains nothing"},
        {"4k3/8/2p5/3p4/8/8/3Q4/4K3 w - - 0 1", "d2d5", 0, false, "queen takes defended pawn loses"},
        {"4k3/8/2p5/3p4/8/8/3Q4/4K3 w - - 0 1", "d2d5", -800, true, "queen for pawn is -800"},
        {"4k3/8/8/3r4/8/8/3R4/3RK3 w - - 0 1", "d2d5", 500, true, "rook wins undefended rook"},
    };
    for (const auto& c : cases) {
        Board b(c.fen);
        const Move m = chess::uci::uciToMove(b, c.uci);
        check(Search::see_ge(b, m, c.threshold) == c.expect, std::string("SEE: ") + c.name);
    }
}

inline void mates() {
    struct Case { const char* fen; const char* best; int mate_plies; const char* name; };
    const Case cases[] = {
        {"6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", "d1d8", 1, "back-rank mate in 1"},
        {"r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", "h5f7", 1, "scholar's mate in 1"},
        {"kbK5/pp6/1P6/8/8/8/8/R7 w - - 0 1", "a1a6", 3, "Morphy's mate in 2 (Ra6!)"},
    };
    for (const auto& c : cases) {
        tt_clear();
        RootResult r = search_root(Board(c.fen), 20, 3000, 1, false);
        const bool ok = !r.moves.empty() && chess::uci::moveToUci(r.moves[0].move) == c.best &&
                        r.moves[0].score == MATE - c.mate_plies;
        check(ok, std::string("mate: ") + c.name);
    }
}

inline void repetition() {
    // After Nf3 Nf6 Ng1 Ng8 the start position has occurred before, so it scores as a draw.
    Board b;
    for (const char* u : {"g1f3", "g8f6", "f3g1", "f6g8"}) b.makeMove(chess::uci::uciToMove(b, u));
    S.nn.reset(b);
    g_stop = false;
    S.time_limit_ms = 10000;
    S.start = std::chrono::steady_clock::now();
    check(S.negamax(b, 4, -INF, INF, 1, true) == 0, "repeated position scores as a draw");
}

inline int run(const char* argv0) {
    nnue_incremental();
    see();
    mates();
    repetition();
    check(selftest(argv0) == 0, "CNN policy matches PyTorch reference");
    std::cout << (fails ? "TESTS FAILED (" + std::to_string(fails) + ")" : std::string("ALL TESTS PASSED")) << "\n";
    return fails ? 1 : 0;
}

}  // namespace tests
