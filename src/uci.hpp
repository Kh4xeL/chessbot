#pragma once
/**
 * UCI protocol (https://backscattering.de/chess/uci/), so standard tools can run the
 * engine: match runners (fastchess, cutechess-cli), GUIs (Arena, CuteChess, Banksia)
 * and lichess-bot. Started with no arguments or with --uci.
 *
 * Supported: uci, isready, ucinewgame, setoption, position (startpos | fen ... [moves]),
 * go (wtime btime winc binc movestogo movetime depth nodes infinite), stop, quit.
 * Options: Threads, MultiPV, Move Overhead, SyzygyPath, UCI_LimitStrength, UCI_Elo.
 *
 * Included at the end of main.cpp; the search runs on a worker thread so "stop" and
 * "isready" are answered while it thinks.
 */

#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace uci_mode {

// The search thread (info, bestmove) and the input loop (readyok, uciok) both write
// to stdout; without a lock their lines can interleave and a GUI misses "readyok".
inline std::mutex out_mutex;
inline void send(const std::string& line) {
    std::lock_guard<std::mutex> lock(out_mutex);
    std::cout << line << std::endl;
}

struct Options {
    int multipv = 1;
    long long overhead_ms = 30;
    bool limit_strength = false;
    int elo = 2900;
};

// "info depth ... score ... pv ..." for each completed iteration.
inline void print_info(const RootResult& r, int multipv, bool white_to_move) {
    (void)white_to_move;
    const long long ms = std::max(1LL, r.ms);
    for (int i = 0; i < multipv && i < static_cast<int>(r.moves.size()); i++) {
        const RootMove& m = r.moves[i];
        if (!m.exact && i > 0) break;
        std::ostringstream os;
        os << "info depth " << r.depth;
        if (multipv > 1) os << " multipv " << i + 1;
        if (std::abs(m.score) >= MATE_BOUND) {
            const int n = (MATE - std::abs(m.score) + 1) / 2;
            os << " score mate " << (m.score > 0 ? n : -n);
        } else {
            os << " score cp " << display_cp(m.score);
        }
        os << " nodes " << r.nodes << " nps " << (r.nodes * 1000 / ms) << " time " << r.ms << " pv";
        for (const auto& mv : (m.pv.empty() ? std::vector<Move>{m.move} : m.pv)) os << " " << chess::uci::moveToUci(mv);
        send(os.str());
    }
}

// One "go": tablebase filtering at the root, search, Elo-limited move choice, bestmove.
inline void think(Board board, int max_depth, long long hard_ms, Limits limits, Options opt) {
    static std::mt19937 rng(static_cast<unsigned>(std::chrono::steady_clock::now().time_since_epoch().count()));
    const int elo = opt.limit_strength ? std::clamp(opt.elo, 800, 2900) : 3000;
    Movelist legal;
    chess::movegen::legalmoves(legal, board);
    if (legal.empty()) {
        send("bestmove 0000");
        return;
    }
    tb::RootInfo tbi;
    const bool use_tb = elo >= 2200 && tb::probe_root(board, tbi);
    std::vector<Move> keep;
    if (use_tb)
        for (const auto& m : tbi.moves)
            if (m.wdl == tbi.wdl) keep.push_back(m.move);

    const int depth = std::min(max_depth, depth_cap_for_elo(elo));
    const int multipv = std::max(opt.multipv, elo >= 2900 ? 1 : 8);
    g_limits = limits;
    const bool white = board.sideToMove() == chess::Color::WHITE;
    g_on_iteration = [&](const RootResult& r) { print_info(r, opt.multipv, white); };
    RootResult r = search_root(board, depth, hard_ms, multipv, false, keep, elo);
    g_on_iteration = nullptr;
    g_limits = Limits{};

    size_t pick = 0;
    if (use_tb && tbi.wdl == TB_WIN) {
        for (size_t i = 0; i < r.moves.size(); i++)
            if (r.moves[i].move == tbi.best) pick = i;
    } else if (!r.moves.empty()) {
        pick = choose_move(r, elo, rng);
    }
    const Move best = r.moves.empty() ? legal[0] : r.moves[pick].move;
    // A partial last iteration can change the best move after the last "info"; report
    // the final line so the PV starts with the move actually played.
    if (!r.moves.empty() && pick == 0) print_info(r, 1, white);
    send("bestmove " + chess::uci::moveToUci(best));
}

inline int run() {
    Board board;
    Options opt;
    std::thread worker;
    auto wait = [&] {
        if (worker.joinable()) worker.join();
    };

    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream ss(line);
        std::string cmd;
        ss >> cmd;

        if (cmd == "uci") {
            send("id name HybridNNUE\n"
                 "id author chessbot\n"
                 "option name Threads type spin default 1 min 1 max 64\n"
                 "option name MultiPV type spin default 1 min 1 max 8\n"
                 "option name Move Overhead type spin default 30 min 0 max 5000\n"
                 "option name SyzygyPath type string default <empty>\n"
                 "option name UCI_LimitStrength type check default false\n"
                 "option name UCI_Elo type spin default 2900 min 800 max 2900\n"
                 "uciok");
        } else if (cmd == "isready") {
            send("readyok");
        } else if (cmd == "ucinewgame") {
            wait();
            tt_clear();
        } else if (cmd == "setoption") {
            // setoption name <Name with spaces> value <value>
            const auto npos = line.find("name "), vpos = line.find(" value ");
            if (npos == std::string::npos) continue;
            const std::string name = line.substr(npos + 5, vpos == std::string::npos ? std::string::npos : vpos - npos - 5);
            const std::string value = vpos == std::string::npos ? "" : line.substr(vpos + 7);
            wait();
            if (name == "Threads") g_threads = std::clamp(std::atoi(value.c_str()), 1, 64);
            else if (name == "MultiPV") opt.multipv = std::clamp(std::atoi(value.c_str()), 1, 8);
            else if (name == "Move Overhead") opt.overhead_ms = std::max(0, std::atoi(value.c_str()));
            else if (name == "SyzygyPath" && !value.empty() && value != "<empty>") tb::init(value);
            else if (name == "UCI_LimitStrength") opt.limit_strength = value == "true";
            else if (name == "UCI_Elo") opt.elo = std::atoi(value.c_str());
        } else if (cmd == "position") {
            wait();
            std::string tok;
            ss >> tok;
            if (tok == "startpos") {
                board = Board();
                ss >> tok;   // "moves" or nothing
            } else if (tok == "fen") {
                std::string fen, part;
                while (ss >> part && part != "moves") fen += (fen.empty() ? "" : " ") + part;
                board = Board(fen);
                tok = part;
            }
            if (tok == "moves") {
                for (std::string u; ss >> u;) {
                    const Move m = chess::uci::uciToMove(board, u);
                    if (m == Move(Move::NO_MOVE)) break;
                    board.makeMove(m);
                }
            }
        } else if (cmd == "go") {
            wait();
            long long wtime = -1, btime = -1, winc = 0, binc = 0, movetime = -1;
            int movestogo = 0, depth = 99;
            Limits lim;
            for (std::string tok; ss >> tok;) {
                if (tok == "wtime") ss >> wtime;
                else if (tok == "btime") ss >> btime;
                else if (tok == "winc") ss >> winc;
                else if (tok == "binc") ss >> binc;
                else if (tok == "movestogo") ss >> movestogo;
                else if (tok == "movetime") ss >> movetime;
                else if (tok == "depth") ss >> depth;
                else if (tok == "nodes") ss >> lim.nodes;
                else if (tok == "infinite") lim.infinite = true;
            }
            const bool white = board.sideToMove() == chess::Color::WHITE;
            const long long left = white ? wtime : btime, inc = white ? winc : binc;
            long long hard = 24LL * 3600 * 1000;   // no time limit unless one is given
            if (movetime >= 0) {
                hard = std::max(1LL, movetime - opt.overhead_ms);
            } else if (left >= 0) {
                // Clock: spend about 1/30 of the remaining time (or 1/movestogo) plus most of
                // the increment; stop new iterations after that, never use more than 3x it.
                const long long base = left / (movestogo > 0 ? std::min(movestogo, 50) : 30) + inc * 3 / 4;
                lim.soft_ms = std::max(1LL, base);
                hard = std::max(1LL, std::min(base * 3, left / 2) - opt.overhead_ms);
            } else if (lim.nodes || depth < 99) {
                lim.soft_ms = LLONG_MAX;   // fixed nodes / depth: no clock
            }
            worker = std::thread(think, board, depth, hard, lim, opt);
        } else if (cmd == "stop") {
            g_stop = true;
            wait();
        } else if (cmd == "quit") {
            g_stop = true;
            wait();
            break;
        }
    }
    g_stop = true;
    wait();
    return 0;
}

}  // namespace uci_mode
