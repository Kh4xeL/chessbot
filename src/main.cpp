/**
 * Hybrid Chess Engine (NNUE alpha-beta search + CNN policy at the root)
 *
 * - Leaf evaluation: an NNUE trained on Stockfish evaluations (src/nnue.hpp),
 *   updated incrementally, roughly a thousand times faster than the old CNN.
 * - Root "intuition": the original CNN's policy head (src/policy_net.hpp) orders
 *   root moves and, below full strength, makes the bot pick human-like moves.
 * - Search: iterative deepening, principal variation search, transposition
 *   table, null-move pruning, reverse futility pruning, late move reductions,
 *   killer/history move ordering, check extensions and quiescence search.
 *
 * Usage:
 *   engine <fen> [elo] [max_depth] [time_ms]   one search, then exit
 *   engine --server                            read commands from stdin:
 *       go <elo> <max_depth> <time_ms> <fen>   prints the result block, then END
 *       newgame                                clears the hash tables
 *       quit
 *   engine --selftest                          checks the CNN against PyTorch output
 *   engine --bench [ms]                        search speed on a few positions
 *
 * Output block (parsed by python/api.py):
 *   BEST_MOVE:e2e4
 *   EVAL:0.31                  (pawns, White's point of view)
 *   DEPTH_REACHED:14
 *   TUTOR_DATA:
 *   RANK:1|MOVE:e2e4|IDEA:e2e4 e7e5 g1f3 ...      (top 3 for the side to move)
 *   USER_TUTOR_DATA:
 *   USER_RANK:1|MOVE:...|IDEA:...                 (top 3 replies after BEST_MOVE)
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "../include/chess.hpp"
#include "nnue.hpp"
#include "policy_net.hpp"

using chess::Board;
using chess::Move;
using chess::Movelist;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr int INF = 32000;
constexpr int MATE = 31000;
constexpr int MATE_BOUND = MATE - 1000;   // scores beyond this are "mate in N"
constexpr int MAX_PLY = 128;

// ---------------------------------------------------------------------------
// Transposition table
// ---------------------------------------------------------------------------
enum Bound : std::uint8_t { BOUND_NONE = 0, BOUND_UPPER = 1, BOUND_LOWER = 2, BOUND_EXACT = 3 };

struct TTEntry {
    std::uint64_t key = 0;
    std::int16_t score = 0;
    std::uint16_t move = 0;
    std::int8_t depth = -1;
    std::uint8_t bound = BOUND_NONE;
};

constexpr size_t TT_SIZE = 1u << 22;   // 4M entries x 16 bytes = 64 MB
std::vector<TTEntry> TT(TT_SIZE);

inline int score_to_tt(int s, int ply) {
    if (s >= MATE_BOUND) return s + ply;
    if (s <= -MATE_BOUND) return s - ply;
    return s;
}
inline int score_from_tt(int s, int ply) {
    if (s >= MATE_BOUND) return s - ply;
    if (s <= -MATE_BOUND) return s + ply;
    return s;
}

// ---------------------------------------------------------------------------
// Search state
// ---------------------------------------------------------------------------
struct Search {
    nnue::Stack nn;
    std::uint64_t nodes = 0;
    bool stop = false;
    std::chrono::steady_clock::time_point start;
    long long time_limit_ms = 1000;

    std::uint16_t killers[MAX_PLY][2] = {};
    int history[2][64][64] = {};
    Move pv[MAX_PLY][MAX_PLY];
    int pv_len[MAX_PLY] = {};
    int lmr[64][64] = {};

    Search() {
        for (int d = 1; d < 64; d++)
            for (int m = 1; m < 64; m++)
                lmr[d][m] = static_cast<int>(0.75 + std::log(d) * std::log(m) / 2.25);
    }

    long long elapsed_ms() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start)
            .count();
    }

    void check_time() {
        if ((nodes & 2047) == 0 && elapsed_ms() >= time_limit_ms) stop = true;
    }

    void make(Board& b, const Move& m) { nn.push(b, m); b.makeMove(m); }
    void unmake(Board& b, const Move& m) { b.unmakeMove(m); nn.pop(); }

    // ------------------------------------------------------------- ordering
    static int piece_value(chess::PieceType pt) {
        static const int v[7] = {100, 320, 330, 500, 900, 0, 0};
        return v[static_cast<int>(pt)];
    }

    int score_move(const Board& b, const Move& m, std::uint16_t tt_move, int ply) const {
        if (m.move() == tt_move) return 10'000'000;
        const bool promo = m.typeOf() == Move::PROMOTION;
        if (b.isCapture(m)) {
            auto victim = m.typeOf() == Move::ENPASSANT ? chess::PieceType(chess::PieceType::PAWN)
                                                         : b.at<chess::PieceType>(m.to());
            auto attacker = b.at<chess::PieceType>(m.from());
            return 1'000'000 + piece_value(victim) * 10 - static_cast<int>(attacker) + (promo ? 5000 : 0);
        }
        if (promo) return m.promotionType() == chess::PieceType::QUEEN ? 950'000 : -100'000;
        if (ply < MAX_PLY) {
            if (killers[ply][0] == m.move()) return 900'000;
            if (killers[ply][1] == m.move()) return 800'000;
        }
        const int c = b.sideToMove() == chess::Color::WHITE ? 0 : 1;
        return history[c][m.from().index()][m.to().index()];
    }

    // Selection sort step: brings the best remaining move to position i.
    static void pick(Movelist& moves, int* scores, int i) {
        int best = i;
        for (int j = i + 1; j < moves.size(); j++)
            if (scores[j] > scores[best]) best = j;
        if (best != i) {
            std::swap(moves[i], moves[best]);
            std::swap(scores[i], scores[best]);
        }
    }

    // ------------------------------------------------------------ quiescence
    int qsearch(Board& b, int alpha, int beta, int ply) {
        nodes++;
        check_time();
        if (stop) return 0;
        if (ply >= MAX_PLY - 1) return nn.evaluate(b);

        const bool in_check = b.inCheck();
        int best = -INF;
        int stand = 0;
        if (!in_check) {
            stand = nn.evaluate(b);
            if (stand >= beta) return stand;
            if (stand > alpha) alpha = stand;
            best = stand;
        }

        Movelist moves;
        if (in_check) chess::movegen::legalmoves(moves, b);
        else chess::movegen::legalmoves<chess::movegen::MoveGenType::CAPTURE>(moves, b);
        if (in_check && moves.empty()) return -MATE + ply;

        int scores[256];
        for (int i = 0; i < moves.size(); i++) scores[i] = score_move(b, moves[i], 0, MAX_PLY);

        for (int i = 0; i < moves.size(); i++) {
            pick(moves, scores, i);
            const Move m = moves[i];
            // Delta pruning: even winning this piece can't lift us to alpha.
            if (!in_check && m.typeOf() != Move::PROMOTION) {
                auto victim = m.typeOf() == Move::ENPASSANT ? chess::PieceType(chess::PieceType::PAWN)
                                                             : b.at<chess::PieceType>(m.to());
                if (stand + piece_value(victim) + 200 < alpha) continue;
            }
            make(b, m);
            int s = -qsearch(b, -beta, -alpha, ply + 1);
            unmake(b, m);
            if (stop) return 0;
            if (s > best) {
                best = s;
                if (s > alpha) {
                    alpha = s;
                    if (alpha >= beta) break;
                }
            }
        }
        return best;
    }

    // ---------------------------------------------------------- main search
    int negamax(Board& b, int depth, int alpha, int beta, int ply, bool allow_null) {
        pv_len[ply] = ply;
        if (b.isHalfMoveDraw() || b.isInsufficientMaterial() || b.isRepetition(1)) return 0;
        if (ply >= MAX_PLY - 1) return nn.evaluate(b);

        // Mate distance pruning
        alpha = std::max(alpha, -MATE + ply);
        beta = std::min(beta, MATE - ply - 1);
        if (alpha >= beta) return alpha;

        const bool in_check = b.inCheck();
        if (in_check) depth++;
        if (depth <= 0) return qsearch(b, alpha, beta, ply);

        nodes++;
        check_time();
        if (stop) return 0;

        const bool pv_node = beta - alpha > 1;
        const std::uint64_t key = b.zobrist();
        TTEntry& tte = TT[key & (TT_SIZE - 1)];
        std::uint16_t tt_move = 0;
        if (tte.key == key) {
            tt_move = tte.move;
            if (!pv_node && tte.depth >= depth) {
                int s = score_from_tt(tte.score, ply);
                if (tte.bound == BOUND_EXACT || (tte.bound == BOUND_LOWER && s >= beta) ||
                    (tte.bound == BOUND_UPPER && s <= alpha))
                    return s;
            }
        }

        const int static_eval = in_check ? -INF : nn.evaluate(b);
        const chess::Color stm = b.sideToMove();

        if (!pv_node && !in_check) {
            // Reverse futility pruning
            if (depth <= 6 && static_eval - 80 * depth >= beta && std::abs(beta) < MATE_BOUND) return static_eval;

            // Null move pruning
            if (allow_null && depth >= 3 && static_eval >= beta && b.hasNonPawnMaterial(stm)) {
                const int R = 3 + depth / 4;
                nn.push_null();
                b.makeNullMove();
                int s = -negamax(b, depth - 1 - R, -beta, -beta + 1, ply + 1, false);
                b.unmakeNullMove();
                nn.pop();
                if (stop) return 0;
                if (s >= beta) return s >= MATE_BOUND ? beta : s;
            }
        }

        Movelist moves;
        chess::movegen::legalmoves(moves, b);
        if (moves.empty()) return in_check ? -MATE + ply : 0;

        int scores[256];
        for (int i = 0; i < moves.size(); i++) scores[i] = score_move(b, moves[i], tt_move, ply);

        const int c = stm == chess::Color::WHITE ? 0 : 1;
        int best = -INF;
        Move best_move = Move(Move::NO_MOVE);
        const int orig_alpha = alpha;
        int searched = 0;
        std::uint16_t quiets_tried[256];
        int n_quiets = 0;

        for (int i = 0; i < moves.size(); i++) {
            pick(moves, scores, i);
            const Move m = moves[i];
            const bool quiet = !b.isCapture(m) && m.typeOf() != Move::PROMOTION;

            // Late move pruning: at shallow depth skip late quiet moves outright.
            if (!pv_node && !in_check && quiet && best > -MATE_BOUND && depth <= 3 &&
                searched >= 4 + depth * depth * 2)
                continue;

            make(b, m);
            const bool gives_check = b.inCheck();
            int s;
            if (searched == 0) {
                s = -negamax(b, depth - 1, -beta, -alpha, ply + 1, true);
            } else {
                int R = 0;
                if (depth >= 3 && searched >= (pv_node ? 3 : 2) && quiet && !in_check && !gives_check) {
                    R = lmr[std::min(depth, 63)][std::min(searched, 63)];
                    if (pv_node) R--;
                    if (m.move() == killers[ply][0] || m.move() == killers[ply][1]) R--;
                    R = std::clamp(R, 0, depth - 2);
                }
                s = -negamax(b, depth - 1 - R, -alpha - 1, -alpha, ply + 1, true);
                if (s > alpha && R > 0) s = -negamax(b, depth - 1, -alpha - 1, -alpha, ply + 1, true);
                if (s > alpha && s < beta) s = -negamax(b, depth - 1, -beta, -alpha, ply + 1, true);
            }
            unmake(b, m);
            if (stop) return 0;
            searched++;
            if (quiet && n_quiets < 256) quiets_tried[n_quiets++] = m.move();

            if (s > best) {
                best = s;
                best_move = m;
                if (s > alpha) {
                    alpha = s;
                    pv[ply][ply] = m;
                    for (int k = ply + 1; k < pv_len[ply + 1]; k++) pv[ply][k] = pv[ply + 1][k];
                    pv_len[ply] = std::max(pv_len[ply + 1], ply + 1);
                    if (alpha >= beta) {
                        if (quiet) {
                            if (killers[ply][0] != m.move()) {
                                killers[ply][1] = killers[ply][0];
                                killers[ply][0] = m.move();
                            }
                            const int bonus = std::min(depth * depth, 400);
                            for (int q = 0; q < n_quiets; q++) {
                                Move qm(quiets_tried[q]);
                                int& h = history[c][qm.from().index()][qm.to().index()];
                                int delta = (quiets_tried[q] == m.move()) ? bonus : -bonus;
                                h += delta - h * std::abs(delta) / 16384;
                            }
                        }
                        break;
                    }
                }
            }
        }

        const Bound bound = best >= beta ? BOUND_LOWER : (best > orig_alpha ? BOUND_EXACT : BOUND_UPPER);
        if (tte.key != key || depth >= tte.depth || bound == BOUND_EXACT) {
            tte.key = key;
            tte.score = static_cast<std::int16_t>(score_to_tt(best, ply));
            tte.move = best_move.move();
            tte.depth = static_cast<std::int8_t>(depth);
            tte.bound = bound;
        }
        return best;
    }
};

// ---------------------------------------------------------------------------
// Root search with Multi-PV (exact scores for the best `multipv` moves)
// ---------------------------------------------------------------------------
struct RootMove {
    Move move;
    int score = -INF;     // side-to-move point of view
    bool exact = false;
    float prior = 0.0f;   // CNN policy probability
    std::vector<Move> pv;
};

struct RootResult {
    std::vector<RootMove> moves;  // sorted best first
    int depth = 0;
    std::uint64_t nodes = 0;
    long long ms = 0;
};

Search S;

std::vector<float> root_priors(const Board& board, const Movelist& moves) {
    std::vector<float> pri(moves.size(), 1.0f / std::max(1, moves.size()));
    if (!policy::NET.loaded || moves.empty()) return pri;
    auto logits = policy::NET.forward(board);
    float mx = -1e30f;
    for (int i = 0; i < moves.size(); i++) mx = std::max(mx, logits[policy::Net::move_index(moves[i])]);
    float sum = 0;
    for (int i = 0; i < moves.size(); i++) {
        pri[i] = std::exp(logits[policy::Net::move_index(moves[i])] - mx);
        sum += pri[i];
    }
    for (auto& p : pri) p /= sum;
    return pri;
}

RootResult search_root(Board board, int max_depth, long long time_ms, int multipv, bool verbose) {
    RootResult res;
    Movelist legal;
    chess::movegen::legalmoves(legal, board);
    if (legal.empty()) return res;

    auto pri = root_priors(board, legal);
    std::vector<RootMove> rms;
    for (int i = 0; i < legal.size(); i++) {
        RootMove rm;
        rm.move = legal[i];
        rm.prior = pri[i];
        rms.push_back(rm);
    }
    std::stable_sort(rms.begin(), rms.end(), [](const RootMove& a, const RootMove& b) { return a.prior > b.prior; });

    S.nodes = 0;
    S.stop = false;
    S.time_limit_ms = std::max(1LL, time_ms);
    S.start = std::chrono::steady_clock::now();
    S.nn.reset(board);
    std::memset(S.killers, 0, sizeof(S.killers));
    for (auto& side : S.history)
        for (auto& row : side)
            for (auto& h : row) h /= 2;

    const int K = std::max(1, std::min(multipv, static_cast<int>(rms.size())));
    res.moves = rms;

    for (int depth = 1; depth <= max_depth; depth++) {
        std::vector<RootMove> cur = rms;
        std::vector<int> exact_scores;  // sorted descending
        bool done = true;

        for (size_t i = 0; i < cur.size(); i++) {
            RootMove& rm = cur[i];
            int alpha = static_cast<int>(exact_scores.size()) >= K ? exact_scores[K - 1] : -INF;
            S.make(board, rm.move);
            int s;
            if (alpha == -INF) {
                s = -S.negamax(board, depth - 1, -INF, INF, 1, true);
            } else {
                s = -S.negamax(board, depth - 1, -alpha - 1, -alpha, 1, true);
                if (!S.stop && s > alpha) s = -S.negamax(board, depth - 1, -INF, -alpha, 1, true);
            }
            S.unmake(board, rm.move);
            if (S.stop) { done = false; break; }

            rm.score = s;
            rm.exact = alpha == -INF || s > alpha;
            if (rm.exact) {
                rm.pv.assign(1, rm.move);
                for (int k = 1; k < S.pv_len[1]; k++) rm.pv.push_back(S.pv[1][k]);
                exact_scores.insert(std::upper_bound(exact_scores.begin(), exact_scores.end(), s, std::greater<int>()), s);
            }
        }
        if (!done) break;

        std::stable_sort(cur.begin(), cur.end(), [](const RootMove& a, const RootMove& b) {
            if (a.exact != b.exact) return a.exact;
            return a.score > b.score;
        });
        rms = cur;
        res.moves = cur;
        res.depth = depth;
        res.nodes = S.nodes;
        res.ms = S.elapsed_ms();

        if (verbose) {
            std::cerr << "[engine] depth " << depth << " " << res.ms << " ms, " << S.nodes << " nodes ("
                      << (S.nodes * 1000 / std::max(1LL, res.ms)) << " nps), best="
                      << chess::uci::moveToUci(cur[0].move) << " score=" << cur[0].score << "\n";
        }
        if (std::abs(cur[0].score) >= MATE_BOUND && depth >= 4) break;     // forced mate found
        if (res.ms * 2 > time_ms) break;   // the next iteration would very likely not finish
    }
    res.nodes = S.nodes;
    res.ms = S.elapsed_ms();
    return res;
}

// ---------------------------------------------------------------------------
// Elo handling
// ---------------------------------------------------------------------------
// Full strength at >= 2900. Below that the search depth is capped and the move
// is sampled among near-best candidates, weighted by how good the search thinks
// each one is and by the CNN's "human intuition" for it.
int depth_cap_for_elo(int elo) { return elo >= 2900 ? 99 : std::clamp((elo - 500) / 150, 1, 20); }

size_t choose_move(const RootResult& r, int elo, std::mt19937& rng) {
    if (elo >= 2900 || r.moves.size() < 2) return 0;
    const double temp = std::max(8.0, (2900.0 - elo) / 7.0);   // centipawns
    const int best = r.moves[0].score;
    if (best >= MATE_BOUND && elo >= 1400) return 0;          // never miss a found mate above beginner level
    std::vector<double> w(r.moves.size(), 0.0);
    for (size_t i = 0; i < r.moves.size(); i++) {
        const RootMove& m = r.moves[i];
        if (!m.exact) continue;
        const double diff = static_cast<double>(m.score - best);
        if (diff < -3.0 * temp) continue;
        w[i] = std::exp(diff / temp) * std::sqrt(std::max(1e-4f, m.prior));
    }
    std::discrete_distribution<size_t> dist(w.begin(), w.end());
    return dist(rng);
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------
std::string pv_string(const std::vector<Move>& pv, size_t max_len = 8) {
    std::string s;
    for (size_t i = 0; i < pv.size() && i < max_len; i++) {
        if (i) s += " ";
        s += chess::uci::moveToUci(pv[i]);
    }
    return s;
}

void run_go(const std::string& fen, int elo, int max_depth, long long total_ms, std::ostream& out) {
    static std::mt19937 rng(static_cast<unsigned>(std::chrono::steady_clock::now().time_since_epoch().count()));
    Board board(fen);
    Movelist legal;
    chess::movegen::legalmoves(legal, board);
    if (legal.empty()) {
        out << "BEST_MOVE:0000\nEVAL:0\nDEPTH_REACHED:0\nTUTOR_DATA:\nUSER_TUTOR_DATA:\nEND" << std::endl;
        return;
    }

    // Small slice for the "your best replies" analysis after the bot's move.
    const long long supp_ms = std::min(500LL, total_ms / 8);
    const long long main_ms = std::max(1LL, total_ms - supp_ms);
    const int depth = std::min(max_depth, depth_cap_for_elo(elo));
    const int multipv = elo >= 2900 ? 3 : 8;

    RootResult r = search_root(board, depth, main_ms, multipv, true);
    const size_t pick = choose_move(r, elo, rng);
    const RootMove& chosen = r.moves[pick];

    const bool white = board.sideToMove() == chess::Color::WHITE;
    auto to_white_pawns = [white](int s) {
        if (s >= MATE_BOUND) s = 10000;
        if (s <= -MATE_BOUND) s = -10000;
        return (white ? s : -s) / 100.0;
    };

    out << "BEST_MOVE:" << chess::uci::moveToUci(chosen.move) << "\n";
    // If time ran out before depth 1 finished there are no scores; report the static eval.
    const int eval_stm = r.depth > 0 ? chosen.score : S.nn.evaluate(board);
    out << "EVAL:" << to_white_pawns(eval_stm) << "\n";
    out << "DEPTH_REACHED:" << r.depth << "\n";
    out << "TUTOR_DATA:\n";
    for (size_t i = 0; i < r.moves.size() && i < 3; i++) {
        const RootMove& m = r.moves[i];
        out << "RANK:" << i + 1 << "|MOVE:" << chess::uci::moveToUci(m.move)
            << "|IDEA:" << pv_string(m.pv.empty() ? std::vector<Move>{m.move} : m.pv) << "\n";
    }

    // Top replies for the human, on the position after the bot's move.
    Board after = board;
    after.makeMove(chosen.move);
    RootResult u = search_root(after, std::max(1, r.depth - 1), supp_ms, 3, false);
    out << "USER_TUTOR_DATA:\n";
    for (size_t i = 0; i < u.moves.size() && i < 3; i++) {
        const RootMove& m = u.moves[i];
        out << "USER_RANK:" << i + 1 << "|MOVE:" << chess::uci::moveToUci(m.move)
            << "|IDEA:" << pv_string(m.pv.empty() ? std::vector<Move>{m.move} : m.pv) << "\n";
    }
    out << "END" << std::endl;

    std::cerr << "[engine] summary: depth=" << r.depth << " nodes=" << r.nodes << " ms=" << r.ms
              << " nps=" << (r.nodes * 1000 / std::max(1LL, r.ms)) << " supp_depth=" << u.depth
              << " picked_rank=" << pick + 1 << "\n";
}

// ---------------------------------------------------------------------------
// Setup / modes
// ---------------------------------------------------------------------------
std::string find_asset(const std::string& name, const char* argv0) {
    std::vector<std::string> candidates = {"assets/" + name, "../assets/" + name, "../../assets/" + name};
    std::string exe(argv0 ? argv0 : "");
    auto slash = exe.find_last_of("/\\");
    if (slash != std::string::npos) {
        std::string dir = exe.substr(0, slash + 1);
        candidates.push_back(dir + "../assets/" + name);
        candidates.push_back(dir + "assets/" + name);
    }
    for (auto& c : candidates)
        if (std::ifstream(c, std::ios::binary)) return c;
    return "assets/" + name;
}

bool load_networks(const char* argv0) {
    const std::string nn_path = find_asset("nnue.bin", argv0);
    if (!nnue::NET.load(nn_path)) {
        std::cerr << "[engine] could not load " << nn_path << "\n";
        std::cout << "BEST_MOVE:ERROR_LOADING_MODEL" << std::endl;
        return false;
    }
    const std::string pol_path = find_asset("policy_cnn.bin", argv0);
    if (!policy::NET.load(pol_path)) std::cerr << "[engine] policy net not found (" << pol_path << "), continuing without it\n";
    return true;
}

int selftest(const char* argv0) {
    std::ifstream f(find_asset("policy_cnn_test.txt", argv0));
    std::string fen, ref;
    int fails = 0;
    while (std::getline(f, fen) && std::getline(f, ref)) {
        Board b(fen);
        auto t0 = std::chrono::steady_clock::now();
        auto logits = policy::NET.forward(b);
        auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::istringstream ss(ref);
        std::string tok;
        while (ss >> tok) {
            int idx = std::stoi(tok.substr(0, tok.find(':')));
            float want = std::stof(tok.substr(tok.find(':') + 1));
            if (std::abs(logits[idx] - want) > 1e-3f) {
                fails++;
                std::cout << "MISMATCH idx " << idx << " got " << logits[idx] << " want " << want << "\n";
            }
        }
        nnue::Stack st;
        st.reset(b);
        std::cout << "policy ok-check (" << ms << " ms), nnue eval " << st.evaluate(b) << " cp (stm)  " << fen << "\n";
    }
    std::cout << (fails ? "SELFTEST FAILED" : "SELFTEST PASSED") << "\n";
    return fails ? 1 : 0;
}

int bench(long long ms) {
    const char* fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
    };
    std::uint64_t total_nodes = 0;
    long long total_ms = 0;
    for (auto fen : fens) {
        std::fill(TT.begin(), TT.end(), TTEntry{});
        RootResult r = search_root(Board(fen), 99, ms, 1, false);
        total_nodes += r.nodes;
        total_ms += r.ms;
        std::cout << "depth " << r.depth << "  best " << chess::uci::moveToUci(r.moves[0].move) << "  score "
                  << r.moves[0].score << "  " << r.nodes << " nodes  " << r.ms << " ms   " << fen << "\n";
    }
    std::cout << "TOTAL " << total_nodes << " nodes, " << (total_nodes * 1000 / std::max(1LL, total_ms)) << " nps\n";
    return 0;
}

int main(int argc, char* argv[]) {
    std::ios::sync_with_stdio(false);
    if (argc < 2) {
        std::cerr << "usage: engine <fen> [elo] [depth] [time_ms] | --server | --selftest | --bench [ms]\n";
        return 1;
    }
    if (!load_networks(argv[0])) return 2;

    std::string mode = argv[1];
    if (mode == "--selftest") return selftest(argv[0]);
    if (mode == "--bench") return bench(argc >= 3 ? std::stoll(argv[2]) : 2000);

    if (mode == "--server") {
        std::string line;
        std::cout << "READY" << std::endl;
        while (std::getline(std::cin, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::istringstream ss(line);
            std::string cmd;
            ss >> cmd;
            if (cmd == "quit") break;
            if (cmd == "newgame") {
                std::fill(TT.begin(), TT.end(), TTEntry{});
                std::cout << "OK" << std::endl;
                continue;
            }
            if (cmd == "go") {
                int elo = 3000, depth = 99;
                long long ms = 3000;
                ss >> elo >> depth >> ms;
                std::string fen;
                std::getline(ss, fen);
                fen.erase(0, fen.find_first_not_of(' '));
                try {
                    run_go(fen, elo, depth, ms, std::cout);
                } catch (const std::exception& e) {
                    std::cout << "ERROR:" << e.what() << "\nEND" << std::endl;
                }
                continue;
            }
            std::cout << "ERROR:unknown command\nEND" << std::endl;
        }
        return 0;
    }

    // One-shot mode (same arguments as the old engine).
    const std::string fen = argv[1];
    const int elo = argc >= 3 ? static_cast<int>(std::stof(argv[2])) : 3000;
    const int depth = argc >= 4 ? std::stoi(argv[3]) : 99;
    const long long ms = argc >= 5 ? std::stoll(argv[4]) : 3000;
    run_go(fen, elo, depth, ms, std::cout);
    return 0;
}
