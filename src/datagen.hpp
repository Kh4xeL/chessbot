#pragma once
// Self-play data generation for NNUE training:
//
//     engine --datagen <out_dir> [games] [nodes] [seed]
//
// Plays games against itself at a fixed node count (default 5000 per move) and appends
// quiet positions to <out_dir> in the trainer's format (see training/prepare_data.py):
//     boards.u8   32 bytes per position, 4 bits per square (a1 = low nibble of byte 0),
//                 pieces 1-6 = white PNBRQK, 7-12 = black
//     scores.i16  search score, White's point of view, centipawns
//     stm.u8      1 = White to move
//     results.u8  game result: 0 = Black won, 1 = draw, 2 = White won
//
// Each game opens with 8 random plies (games starting more than 10 pawns from equal are
// dropped), then both sides play the search's best move. Games are cut short by the
// tablebases, by 4 plies in a row with a score beyond +-10 pawns, or by 10 plies in a row
// within +-0.1 pawns after move 40. Run one process per core with different seeds and
// output directories (scripts/datagen.py does that).

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>

namespace datagen {

struct Pos {
    std::uint8_t board[32];
    std::int16_t score;   // White's view
    std::uint8_t stm;
};

inline void encode(const Board& b, Pos& p) {
    std::uint8_t sq[64] = {};
    for (int c = 0; c < 2; c++)
        for (int t = 0; t < 6; t++) {
            std::uint64_t bb = b.pieces(static_cast<chess::PieceType::underlying>(t),
                                        c == 0 ? chess::Color::WHITE : chess::Color::BLACK)
                                   .getBits();
            while (bb) {
                const int s = __builtin_ctzll(bb);
                bb &= bb - 1;
                sq[s] = static_cast<std::uint8_t>(1 + t + 6 * c);
            }
        }
    for (int i = 0; i < 32; i++) p.board[i] = static_cast<std::uint8_t>(sq[2 * i] | (sq[2 * i + 1] << 4));
}

inline int run(const std::string& out_dir, long long games, std::uint64_t nodes, unsigned seed) {
    std::mt19937 rng(seed);
    {   // a run stopped mid-write can leave the files a record apart: trim to the common length
        namespace fs = std::filesystem;
        const char* names[] = {"boards.u8", "scores.i16", "stm.u8", "results.u8"};
        const std::uintmax_t width[] = {32, 2, 1, 1};
        std::uintmax_t n = UINTMAX_MAX;
        bool any = false;
        for (int i = 0; i < 4; i++) {
            const fs::path p = fs::path(out_dir) / names[i];
            if (fs::exists(p)) { n = std::min(n, fs::file_size(p) / width[i]); any = true; }
            else n = 0;
        }
        if (any)
            for (int i = 0; i < 4; i++) {
                const fs::path p = fs::path(out_dir) / names[i];
                if (fs::exists(p)) fs::resize_file(p, n * width[i]);
            }
    }
    std::ofstream f_board(out_dir + "/boards.u8", std::ios::binary | std::ios::app);
    std::ofstream f_score(out_dir + "/scores.i16", std::ios::binary | std::ios::app);
    std::ofstream f_stm(out_dir + "/stm.u8", std::ios::binary | std::ios::app);
    std::ofstream f_res(out_dir + "/results.u8", std::ios::binary | std::ios::app);
    if (!f_board || !f_score || !f_stm || !f_res) {
        std::cerr << "cannot write to " << out_dir << "\n";
        return 1;
    }
    policy::NET.loaded = false;   // root ordering by the CNN costs more than a 5k-node search
    g_threads = 1;

    long long total = 0, played = 0;
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<Pos> buf;
    while (played < games) {
        Board b;
        bool ok = true;
        for (int i = 0; i < 8 && ok; i++) {
            Movelist ml;
            chess::movegen::legalmoves(ml, b);
            if (ml.empty()) ok = false;
            else b.makeMove(ml[std::uniform_int_distribution<int>(0, ml.size() - 1)(rng)]);
        }
        if (!ok) continue;

        buf.clear();
        int result = -1;   // 0 / 1 / 2 from White's view
        int win_run = 0, draw_run = 0;
        for (int ply = 0;; ply++) {
            const auto over = b.isGameOver();
            if (over.second != chess::GameResult::NONE) {
                if (over.second == chess::GameResult::DRAW) result = 1;
                else result = b.sideToMove() == chess::Color::WHITE ? 0 : 2;   // side to move is mated
                break;
            }
            if (ply >= 400) { result = 1; break; }
            if (tb::enabled) {
                const unsigned wdl = tb::probe_wdl(b);
                if (wdl != TB_RESULT_FAILED) {
                    const bool white = b.sideToMove() == chess::Color::WHITE;
                    result = wdl == TB_WIN ? (white ? 2 : 0) : wdl == TB_LOSS ? (white ? 0 : 2) : 1;
                    break;
                }
            }

            g_limits = Limits{};
            g_limits.nodes = nodes;
            RootResult r = search_root(b, 99, 1'000'000, 1, false, {}, 3000);
            g_limits = Limits{};
            if (r.moves.empty()) { ok = false; break; }
            const int score = r.moves[0].score;   // side to move's view
            const Move m = r.moves[0].move;
            if (ply == 0 && std::abs(score) > 1000) { ok = false; break; }   // unbalanced opening

            const bool white = b.sideToMove() == chess::Color::WHITE;
            const int white_score = white ? score : -score;
            if (std::abs(score) < TB_BOUND && !b.inCheck() && !b.isCapture(m) && m.typeOf() != Move::PROMOTION) {
                Pos p;
                encode(b, p);
                p.score = static_cast<std::int16_t>(std::clamp(white_score, -2000, 2000));
                p.stm = white ? 1 : 0;
                buf.push_back(p);
            }

            win_run = std::abs(score) >= 1000 ? win_run + 1 : 0;
            draw_run = (ply >= 80 && std::abs(score) <= 10) ? draw_run + 1 : 0;
            if (win_run >= 4) { result = white_score > 0 ? 2 : 0; break; }
            if (draw_run >= 10) { result = 1; break; }
            b.makeMove(m);
        }
        if (!ok || result < 0) continue;

        for (const Pos& p : buf) {
            f_board.write(reinterpret_cast<const char*>(p.board), 32);
            f_score.write(reinterpret_cast<const char*>(&p.score), 2);
            f_stm.put(static_cast<char>(p.stm));
            f_res.put(static_cast<char>(result));
        }
        f_board.flush(); f_score.flush(); f_stm.flush(); f_res.flush();
        total += static_cast<long long>(buf.size());
        played++;
        if (played % 50 == 0 || played == games) {
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::cout << "games " << played << "  positions " << total << "  " << static_cast<int>(total / s)
                      << " pos/s" << std::endl;
        }
    }
    return 0;
}

}  // namespace datagen
