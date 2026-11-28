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
 *   engine                                     UCI protocol (chess GUIs, fastchess, cutechess); also --uci
 *   engine <fen> [elo] [max_depth] [time_ms]   one search, then exit
 *   engine --server                            read commands from stdin:
 *       go <elo> <max_depth> <time_ms> <fen> [moves m1 m2 ...]   prints the result block, then END
 *       humanprobs <elo> <fen> [moves ...]     each legal move's probability for a player of that rating:
 *                                              HUMAN_PROBS:<elo>, then MOVE:e2e4|P:0.41 ... (most likely first), END
 *       humangame <elo,elo,...> <w|b|all> <fen> moves m1 m2 ...
 *                                              per move of that side: PLY:n|SIDE:w|MOVE:..|LEGAL:n|P:p1,p2,..|RANK:r1,r2,..
 *                                              (played move's probability and rank at each rating), then END
 *       newgame                                clears the hash tables
 *       quit
 *   engine --selftest                          checks the CNN against PyTorch output
 *   engine --test                              all engine self-checks (NNUE, SEE, mates, ...)
 *   engine --bench [ms | dN]                   search speed on a few positions (dN: fixed depth N,
 *                                              the same tree every run, for comparing builds)
 *   --threads N                                (any mode) Lazy SMP search threads, default 1
 *   --syzygy DIR                               (any mode) Syzygy tablebases; default $SYZYGY_PATH or
 *                                              tools/syzygy, "none" to disable
 *
 * Output block (parsed by python/api.py):
 *   BEST_MOVE:e2e4
 *   EVAL:0.31                  (pawns, White's point of view)
 *   MATE:3                     (only when a mate is found: mate in N, + = White mates)
 *   TABLEBASE:1                (only in tablebase positions: 1 / 0 / -1 = White wins / draw / loses)
 *   DEPTH_REACHED:14
 *   TUTOR_DATA:
 *   RANK:1|MOVE:e2e4|IDEA:e2e4 e7e5 g1f3 ...|EVAL:0.31[|MATE:n]   (top 3 for the side to move;
 *                                                  EVAL = line's final evaluation, White's view)
 *   USER_TUTOR_DATA:
 *   USER_RANK:1|MOVE:...|IDEA:...                 (top 3 replies after BEST_MOVE)
 */

#include <algorithm>
#include <atomic>
#include <climits>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "../include/chess.hpp"
#include "nnue.hpp"
#include "policy_net.hpp"
#include "tablebase.hpp"

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
constexpr int TB_WIN_SCORE = 28000;        // tablebase win (minus ply), below mate scores
constexpr int TB_BOUND = TB_WIN_SCORE - 1000;

// ---------------------------------------------------------------------------
// Search parameters. Each one is also a UCI spin option (same name), so SPSA
// tuners can set them; defaults are the tuned values.
// ---------------------------------------------------------------------------
namespace tune {
struct Param {
    const char* name;
    int* value;
    int def, min, max;
};
inline std::vector<Param>& all() {
    static std::vector<Param> params;
    return params;
}
struct Reg {
    Reg(const char* name, int* v, int mn, int mx) { all().push_back({name, v, *v, mn, mx}); }
};
}  // namespace tune
#define TUNE(name, def, mn, mx) \
    int name = def;             \
    tune::Reg name##_reg(#name, &name, mn, mx);

TUNE(rfp_depth, 4, 3, 10)
TUNE(rfp_margin, 96, 30, 160)
TUNE(nmp_base, 2, 1, 5)
TUNE(nmp_div, 3, 2, 8)
TUNE(pc_depth, 3, 3, 8)
TUNE(pc_margin, 189, 80, 400)
TUNE(pc_improving, 57, 0, 150)
TUNE(se_depth, 5, 5, 12)
TUNE(se_mult, 2, 1, 5)
TUNE(lmp_base, 1, 1, 6)
TUNE(lmp_imp_base, 4, 2, 10)
TUNE(see_depth, 7, 3, 10)
TUNE(see_quiet, 50, 20, 150)
TUNE(see_noisy, 96, 40, 200)
TUNE(lmr_base, 79, 0, 150)     // reduction = base/100 + ln(d) ln(m) / (div/100)
TUNE(lmr_div, 201, 150, 400)
TUNE(qs_delta, 204, 50, 400)
TUNE(asp_delta, 21, 8, 60)
TUNE(hist_max_bonus, 405, 100, 1200)
TUNE(counter_bonus, 815, 0, 2000)   // in thousands, ordering score of the counter-move

int LMR[64][64] = {};
inline void init_lmr() {
    for (int d = 1; d < 64; d++)
        for (int m = 1; m < 64; m++)
            LMR[d][m] = static_cast<int>(lmr_base / 100.0 + std::log(d) * std::log(m) / (lmr_div / 100.0));
}

// Score shown to users: mates as +-100 pawns, tablebase wins as +-90, else unchanged.
inline int display_cp(int s) {
    if (s >= MATE_BOUND) return 10000;
    if (s <= -MATE_BOUND) return -10000;
    if (s >= TB_BOUND) return 9000;
    if (s <= -TB_BOUND) return -9000;
    return s;
}

// ---------------------------------------------------------------------------
// Transposition table
// ---------------------------------------------------------------------------
enum Bound : std::uint8_t { BOUND_NONE = 0, BOUND_UPPER = 1, BOUND_LOWER = 2, BOUND_EXACT = 3 };

// Shared by all search threads without locks: each slot stores the data word and
// key ^ data, so a slot torn by two threads writing at once simply fails the key check.
struct TTData {
    int score = 0;
    std::uint16_t move = 0;
    int depth = -1;
    Bound bound = BOUND_NONE;
};

struct TTSlot {
    std::atomic<std::uint64_t> key_xor{0};
    std::atomic<std::uint64_t> data{0};
};

// Buckets of 4 slots (one 64-byte cache line). Each entry carries the generation of the
// search that wrote it, so entries left over from earlier moves are replaced first.
constexpr int TT_WAYS = 4;
constexpr size_t TT_BUCKETS = 1u << 20;   // 1M buckets x 4 slots x 16 bytes = 64 MB
struct alignas(64) TTBucket {
    TTSlot slot[TT_WAYS];
};
std::vector<TTBucket> TT(TT_BUCKETS);
std::uint8_t tt_gen = 0;

inline void tt_new_search() { tt_gen = static_cast<std::uint8_t>(tt_gen + 1); }

inline std::uint64_t tt_pack(int score, std::uint16_t move, int depth, Bound bound) {
    return static_cast<std::uint64_t>(static_cast<std::uint16_t>(static_cast<std::int16_t>(score))) |
           (static_cast<std::uint64_t>(move) << 16) |
           (static_cast<std::uint64_t>(static_cast<std::uint8_t>(static_cast<std::int8_t>(depth))) << 32) |
           (static_cast<std::uint64_t>(bound) << 40) | (static_cast<std::uint64_t>(tt_gen) << 48);
}

inline TTBucket& tt_bucket(std::uint64_t key) { return TT[key & (TT_BUCKETS - 1)]; }
inline int tt_depth_of(std::uint64_t d) { return static_cast<std::int8_t>((d >> 32) & 0xFF); }
inline std::uint8_t tt_gen_of(std::uint64_t d) { return static_cast<std::uint8_t>((d >> 48) & 0xFF); }

inline bool tt_probe(std::uint64_t key, TTData& out) {
    const TTBucket& bk = tt_bucket(key);
    for (const TTSlot& slot : bk.slot) {
        const std::uint64_t d = slot.data.load(std::memory_order_relaxed);
        if ((slot.key_xor.load(std::memory_order_relaxed) ^ d) != key || d == 0) continue;
        out.score = static_cast<std::int16_t>(d & 0xFFFF);
        out.move = static_cast<std::uint16_t>((d >> 16) & 0xFFFF);
        out.depth = tt_depth_of(d);
        out.bound = static_cast<Bound>((d >> 40) & 0xFF);
        return true;
    }
    return false;
}

// Pick the slot to write: the one already holding this key, else the least valuable
// (empty first, then by depth minus 8 per generation of age). Returns null when the
// entry for this key should be kept (deeper, same search, and the new one isn't exact).
inline TTSlot* tt_victim(std::uint64_t key, int depth, Bound bound, std::uint16_t& move) {
    TTBucket& bk = tt_bucket(key);
    TTSlot* victim = nullptr;
    int worst = 1 << 30;
    for (TTSlot& slot : bk.slot) {
        const std::uint64_t d = slot.data.load(std::memory_order_relaxed);
        const std::uint64_t kx = slot.key_xor.load(std::memory_order_relaxed);
        if (d == 0 && kx == 0) {
            if (worst > -1000) { victim = &slot; worst = -1000; }
            continue;
        }
        if ((kx ^ d) == key) {
            if (move == 0) move = static_cast<std::uint16_t>((d >> 16) & 0xFFFF);   // keep the old best move
            if (tt_gen_of(d) == tt_gen && depth < tt_depth_of(d) && bound != BOUND_EXACT) return nullptr;
            return &slot;
        }
        const int age = static_cast<std::uint8_t>(tt_gen - tt_gen_of(d));
        const int value = tt_depth_of(d) - 8 * age;
        if (value < worst) { victim = &slot; worst = value; }
    }
    return victim;
}

inline void tt_write(TTSlot* slot, std::uint64_t key, std::uint64_t d) {
    slot->data.store(d, std::memory_order_relaxed);
    slot->key_xor.store(key ^ d, std::memory_order_relaxed);
}

inline void tt_store(std::uint64_t key, int score, std::uint16_t move, int depth, Bound bound) {
    if (TTSlot* slot = tt_victim(key, depth, bound, move)) tt_write(slot, key, tt_pack(score, move, depth, bound));
}

// Quiescence results (depth 0) never push out results of real searches from this search.
inline void tt_store_q(std::uint64_t key, int score, std::uint16_t move, Bound bound) {
    TTSlot* slot = tt_victim(key, 0, bound, move);
    if (!slot) return;
    const std::uint64_t old = slot->data.load(std::memory_order_relaxed);
    if (old != 0 && tt_depth_of(old) > 0 && tt_gen_of(old) == tt_gen) return;
    tt_write(slot, key, tt_pack(score, move, 0, bound));
}

void tt_clear() {
    for (auto& bk : TT)
        for (auto& slot : bk.slot) {
            slot.key_xor.store(0, std::memory_order_relaxed);
            slot.data.store(0, std::memory_order_relaxed);
        }
}

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
// Set by the main thread when time is up (or by UCI "stop"); every search thread polls it.
std::atomic<bool> g_stop{false};

struct RootResult;

// Extra limits for the next search, used by UCI. soft_ms: don't start a new iteration
// after this (default: 60% of the hard limit); nodes: stop the main thread after this
// many nodes; infinite: keep searching until stopped; dynamic: soft_ms comes from a game
// clock, so scale it each iteration by how settled the search is (see iterate()).
struct Limits {
    long long soft_ms = -1;
    std::uint64_t nodes = 0;
    bool infinite = false;
    bool dynamic = false;
};
Limits g_limits;
// Called by the main thread after every completed iteration (UCI "info" lines).
std::function<void(const RootResult&)> g_on_iteration;

struct Search {
    nnue::Stack nn;
    std::uint64_t nodes = 0;
    std::uint64_t tb_hits = 0;
    std::atomic<bool>& stop = g_stop;
    bool is_main = true;      // only the main thread checks the clock
    std::chrono::steady_clock::time_point start;
    long long time_limit_ms = 1000;
    std::uint64_t node_limit = 0;   // 0 = none

    std::uint16_t killers[MAX_PLY][2] = {};
    int history[2][64][64] = {};
    // Continuation history: how a quiet move (piece, to) did right after a given
    // previous move (piece, to), one and two plies back. Pieces are 0..11.
    std::vector<std::int16_t> cont = std::vector<std::int16_t>(12 * 64 * 12 * 64, 0);
    std::uint16_t counter[12][64] = {};   // quiet move that refuted the previous move
    struct Frame {
        int piece = -1;   // piece moved at this ply (-1: none / null move)
        int to = 0;
        int eval = -INF;  // static eval at this ply (-INF when in check)
        std::uint16_t excluded = 0;   // move skipped by a singular-extension search
    };
    Frame ss[MAX_PLY + 4];
    Move pv[MAX_PLY][MAX_PLY];
    int pv_len[MAX_PLY] = {};

    long long elapsed_ms() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start)
            .count();
    }

    void check_time() {
        if (!is_main) return;
        if (node_limit && nodes >= node_limit) stop = true;   // exact, so fixed-node games are reproducible
        else if ((nodes & 2047) == 0 && elapsed_ms() >= time_limit_ms) stop = true;
    }

    void make(Board& b, const Move& m) { nn.push(b, m); b.makeMove(m); nn.finish(b); }
    void unmake(Board& b, const Move& m) { b.unmakeMove(m); nn.pop(); }

    // ------------------------------------------------------------- ordering
    static int piece_value(chess::PieceType pt) {
        static const int v[7] = {100, 320, 330, 500, 900, 0, 0};
        return v[static_cast<int>(pt)];
    }

    // Static exchange evaluation: does the capture sequence on m's target square,
    // with both sides always recapturing with their cheapest piece, gain at least
    // `threshold` centipawns? (Pins are ignored, as in most engines.)
    static bool see_ge(const Board& b, const Move& m, int threshold) {
        if (m.typeOf() != Move::NORMAL) return 0 >= threshold;
        using chess::PieceType;
        const chess::Square to = m.to();
        int swap = piece_value(b.at<PieceType>(to)) - threshold;
        if (swap < 0) return false;
        swap = piece_value(b.at<PieceType>(m.from())) - swap;
        if (swap <= 0) return true;

        const std::uint64_t pawns = b.pieces(PieceType::PAWN).getBits();
        const std::uint64_t knights = b.pieces(PieceType::KNIGHT).getBits();
        const std::uint64_t bishops = b.pieces(PieceType::BISHOP).getBits();
        const std::uint64_t rooks = b.pieces(PieceType::ROOK).getBits();
        const std::uint64_t queens = b.pieces(PieceType::QUEEN).getBits();
        const std::uint64_t kings = b.pieces(PieceType::KING).getBits();
        const std::uint64_t diag = bishops | queens, ortho = rooks | queens;

        std::uint64_t occ = b.occ().getBits() ^ (1ULL << m.from().index()) ^ (1ULL << to.index());
        std::uint64_t attackers =
            (chess::attacks::pawn(chess::Color::BLACK, to).getBits() & pawns & b.us(chess::Color::WHITE).getBits()) |
            (chess::attacks::pawn(chess::Color::WHITE, to).getBits() & pawns & b.us(chess::Color::BLACK).getBits()) |
            (chess::attacks::knight(to).getBits() & knights) |
            (chess::attacks::bishop(to, occ).getBits() & diag) |
            (chess::attacks::rook(to, occ).getBits() & ortho) |
            (chess::attacks::king(to).getBits() & kings);

        chess::Color stm = b.sideToMove();
        int res = 1;
        while (true) {
            stm = ~stm;
            attackers &= occ;
            const std::uint64_t mine = attackers & b.us(stm).getBits();
            if (!mine) break;
            res ^= 1;
            std::uint64_t bb;
            if ((bb = mine & pawns)) {
                if ((swap = 100 - swap) < res) break;
                occ ^= bb & (~bb + 1);
                attackers |= chess::attacks::bishop(to, occ).getBits() & diag;
            } else if ((bb = mine & knights)) {
                if ((swap = 320 - swap) < res) break;
                occ ^= bb & (~bb + 1);
            } else if ((bb = mine & bishops)) {
                if ((swap = 330 - swap) < res) break;
                occ ^= bb & (~bb + 1);
                attackers |= chess::attacks::bishop(to, occ).getBits() & diag;
            } else if ((bb = mine & rooks)) {
                if ((swap = 500 - swap) < res) break;
                occ ^= bb & (~bb + 1);
                attackers |= chess::attacks::rook(to, occ).getBits() & ortho;
            } else if ((bb = mine & queens)) {
                if ((swap = 900 - swap) < res) break;
                occ ^= bb & (~bb + 1);
                attackers |= (chess::attacks::bishop(to, occ).getBits() & diag) |
                             (chess::attacks::rook(to, occ).getBits() & ortho);
            } else {
                // King "captures": only legal if the other side has no attackers left.
                return (attackers & ~b.us(stm).getBits()) ? (res ^ 1) : res;
            }
        }
        return res;
    }

    static int moved_piece(const Board& b, const Move& m) {
        return (b.sideToMove() == chess::Color::WHITE ? 0 : 6) + static_cast<int>(b.at<chess::PieceType>(m.from()));
    }
    static int dest(const Move& m) {
        const int from = m.from().index(), to = m.to().index();
        return m.typeOf() == Move::CASTLING ? (from & 56) + (to > from ? 6 : 2) : to;
    }
    std::int16_t* cont_entry(int ply, int back, int piece, int to) {
        if (ply < back) return nullptr;
        const Frame& f = ss[ply - back];
        if (f.piece < 0) return nullptr;
        return &cont[((static_cast<size_t>(f.piece) * 64 + f.to) * 12 + piece) * 64 + to];
    }
    int quiet_score(const Board& b, const Move& m, int ply) {
        const int c = b.sideToMove() == chess::Color::WHITE ? 0 : 1;
        const int piece = moved_piece(b, m), to = dest(m);
        int sc = history[c][m.from().index()][m.to().index()];
        if (auto* e = cont_entry(ply, 1, piece, to)) sc += *e;
        if (auto* e = cont_entry(ply, 2, piece, to)) sc += *e;
        return sc;
    }
    static void gravity(int& h, int delta) { h += delta - h * std::abs(delta) / 16384; }
    static void gravity16(std::int16_t& h, int delta) {
        int v = h;
        v += delta - v * std::abs(delta) / 16384;
        h = static_cast<std::int16_t>(std::clamp(v, -16384, 16384));
    }

    int score_move(Board& b, const Move& m, std::uint16_t tt_move, int ply) {
        if (m.move() == tt_move) return 10'000'000;
        const bool promo = m.typeOf() == Move::PROMOTION;
        if (b.isCapture(m)) {
            auto victim = m.typeOf() == Move::ENPASSANT ? chess::PieceType(chess::PieceType::PAWN)
                                                         : b.at<chess::PieceType>(m.to());
            auto attacker = b.at<chess::PieceType>(m.from());
            const int mvv_lva = piece_value(victim) * 10 - static_cast<int>(attacker) + (promo ? 5000 : 0);
            // Captures that lose material (by SEE) go after the killers and good quiets.
            return (see_ge(b, m, 0) ? 1'000'000 : -500'000) + mvv_lva;
        }
        if (promo) return m.promotionType() == chess::PieceType::QUEEN ? 950'000 : -100'000;
        if (ply < MAX_PLY) {
            if (killers[ply][0] == m.move()) return 900'000;
            if (killers[ply][1] == m.move()) return 800'000;
            if (ply > 0 && ss[ply - 1].piece >= 0 && counter[ss[ply - 1].piece][ss[ply - 1].to] == m.move())
                return counter_bonus * 1000;
            return quiet_score(b, m, ply);
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

        // Hash table: any stored result (every search depth counts here) can end this node.
        const std::uint64_t key = b.zobrist();
        TTData tte;
        std::uint16_t tt_move = 0;
        if (tt_probe(key, tte)) {
            const int s = score_from_tt(tte.score, ply);
            if (tte.bound == BOUND_EXACT || (tte.bound == BOUND_LOWER && s >= beta) ||
                (tte.bound == BOUND_UPPER && s <= alpha))
                return s;
            tt_move = tte.move;
        }
        const int orig_alpha = alpha;

        const bool in_check = b.inCheck();
        int best = -INF;
        int stand = 0;
        if (!in_check) {
            stand = nn.evaluate(b);
            if (stand >= beta) {
                tt_store_q(key, score_to_tt(stand, ply), 0, BOUND_LOWER);
                return stand;
            }
            if (stand > alpha) alpha = stand;
            best = stand;
        }

        Movelist moves;
        if (in_check) chess::movegen::legalmoves(moves, b);
        else chess::movegen::legalmoves<chess::movegen::MoveGenType::CAPTURE>(moves, b);
        if (in_check && moves.empty()) return -MATE + ply;

        int scores[256];
        for (int i = 0; i < moves.size(); i++) scores[i] = score_move(b, moves[i], tt_move, MAX_PLY);
        Move best_move = Move(Move::NO_MOVE);

        for (int i = 0; i < moves.size(); i++) {
            pick(moves, scores, i);
            const Move m = moves[i];
            // Delta pruning: even winning this piece can't lift us to alpha.
            if (!in_check && m.typeOf() != Move::PROMOTION) {
                auto victim = m.typeOf() == Move::ENPASSANT ? chess::PieceType(chess::PieceType::PAWN)
                                                             : b.at<chess::PieceType>(m.to());
                if (stand + piece_value(victim) + qs_delta < alpha) continue;
                // Losing captures can't improve a quiet position.
                if (!see_ge(b, m, 0)) continue;
            }
            make(b, m);
            int s = -qsearch(b, -beta, -alpha, ply + 1);
            unmake(b, m);
            if (stop) return 0;
            if (s > best) {
                best = s;
                if (s > alpha) {
                    alpha = s;
                    best_move = m;
                    if (alpha >= beta) break;
                }
            }
        }
        const Bound bound = best >= beta ? BOUND_LOWER : (best > orig_alpha ? BOUND_EXACT : BOUND_UPPER);
        tt_store_q(key, score_to_tt(best, ply), best_move.move(), bound);
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
        const std::uint16_t excluded = ss[ply].excluded;
        TTData tte;
        std::uint16_t tt_move = 0;
        const bool tt_hit = tt_probe(key, tte);
        if (tt_hit) {
            tt_move = tte.move;
            if (!pv_node && !excluded && tte.depth >= depth) {
                int s = score_from_tt(tte.score, ply);
                if (tte.bound == BOUND_EXACT || (tte.bound == BOUND_LOWER && s >= beta) ||
                    (tte.bound == BOUND_UPPER && s <= alpha))
                    return s;
            }
        }

        // Endgame tablebase (after captures / pawn moves): the exact result for the side to move.
        if (tb::enabled) {
            const unsigned wdl = tb::probe_wdl(b);
            if (wdl != TB_RESULT_FAILED) {
                tb_hits++;
                const int v = wdl == TB_WIN ? TB_WIN_SCORE - ply : wdl == TB_LOSS ? -TB_WIN_SCORE + ply : 0;
                const Bound bd = wdl == TB_WIN ? BOUND_LOWER : wdl == TB_LOSS ? BOUND_UPPER : BOUND_EXACT;
                if (bd == BOUND_EXACT || (bd == BOUND_LOWER && v >= beta) || (bd == BOUND_UPPER && v <= alpha)) {
                    tt_store(key, v, 0, std::min(depth + 6, 120), bd);
                    return v;
                }
            }
        }

        const int static_eval = in_check ? -INF : nn.evaluate(b);
        const chess::Color stm = b.sideToMove();
        ss[ply].eval = static_eval;
        // "Improving": our static eval is better than two plies ago (our previous turn).
        // When it isn't, the position is drifting the wrong way, so prune and reduce more.
        const bool improving = !in_check && ply >= 2 && ss[ply - 2].eval != -INF && static_eval > ss[ply - 2].eval;

        if (!pv_node && !in_check) {
            // Reverse futility pruning
            if (depth <= rfp_depth && static_eval - rfp_margin * (depth - improving) >= beta && std::abs(beta) < MATE_BOUND)
                return static_eval;

            // Null move pruning. Not with less than a rook or two minors: in such endgames
            // zugzwang is common and "passing" would wrongly look safe (e.g. Morphy's Ra6!).
            const int npm = 3 * (b.pieces(chess::PieceType::KNIGHT, stm).count() + b.pieces(chess::PieceType::BISHOP, stm).count()) +
                            5 * b.pieces(chess::PieceType::ROOK, stm).count() + 9 * b.pieces(chess::PieceType::QUEEN, stm).count();
            if (allow_null && !excluded && depth >= 3 && static_eval >= beta && npm >= 5) {
                const int R = nmp_base + depth / nmp_div;
                ss[ply].piece = -1;
                nn.push_null();
                b.makeNullMove();
                int s = -negamax(b, depth - 1 - R, -beta, -beta + 1, ply + 1, false);
                b.unmakeNullMove();
                nn.pop();
                if (stop) return 0;
                if (s >= beta) return s >= MATE_BOUND ? beta : s;
            }

            // ProbCut: a capture that beats beta by a clear margin in a shallow search
            // almost surely beats beta in the full one, so cut here.
            const int pc_beta = beta + pc_margin - pc_improving * improving;
            if (!excluded && depth >= pc_depth && std::abs(beta) < MATE_BOUND &&
                !(tt_hit && tte.depth >= depth - 3 && score_from_tt(tte.score, ply) < pc_beta)) {
                Movelist caps;
                chess::movegen::legalmoves<chess::movegen::MoveGenType::CAPTURE>(caps, b);
                int cs[256];
                for (int i = 0; i < caps.size(); i++) cs[i] = score_move(b, caps[i], tt_move, MAX_PLY);
                for (int i = 0; i < caps.size(); i++) {
                    pick(caps, cs, i);
                    const Move m = caps[i];
                    if (!see_ge(b, m, pc_beta - static_eval)) continue;
                    ss[ply].piece = moved_piece(b, m);
                    ss[ply].to = dest(m);
                    make(b, m);
                    int s = -qsearch(b, -pc_beta, -pc_beta + 1, ply + 1);
                    if (s >= pc_beta) s = -negamax(b, depth - 4, -pc_beta, -pc_beta + 1, ply + 1, true);
                    unmake(b, m);
                    if (stop) return 0;
                    if (s >= pc_beta) {
                        tt_store(key, score_to_tt(s, ply), m.move(), depth - 3, BOUND_LOWER);
                        return s;
                    }
                }
            }
        }

        Movelist moves;
        chess::movegen::legalmoves(moves, b);
        if (moves.empty()) return in_check ? -MATE + ply : 0;
        if (excluded && moves.size() == 1) return alpha;   // the only move is excluded: nothing else to try

        int scores[256];
        for (int i = 0; i < moves.size(); i++) scores[i] = score_move(b, moves[i], tt_move, ply);

        // Singular extension: if the hash move is much better than every alternative
        // (searched at reduced depth with it excluded), search it one ply deeper. If even
        // the alternatives beat beta, this node fails high without searching further.
        int singular_ext = 0;
        if (ply > 0 && depth >= se_depth && tt_hit && tt_move && !excluded && tte.depth >= depth - 3 &&
            tte.bound != BOUND_UPPER && std::abs(tte.score) < TB_BOUND) {
            const int tt_score = score_from_tt(tte.score, ply);
            const int sbeta = tt_score - se_mult * depth;
            ss[ply].excluded = tt_move;
            const int v = negamax(b, (depth - 1) / 2, sbeta - 1, sbeta, ply, false);
            ss[ply].excluded = 0;
            if (stop) return 0;
            if (v < sbeta) singular_ext = 1;
            else if (sbeta >= beta) return sbeta;
        }

        const int c = stm == chess::Color::WHITE ? 0 : 1;
        int best = -INF;
        Move best_move = Move(Move::NO_MOVE);
        const int orig_alpha = alpha;
        int searched = 0;
        Move quiets_tried[256];
        int n_quiets = 0;

        for (int i = 0; i < moves.size(); i++) {
            pick(moves, scores, i);
            const Move m = moves[i];
            if (m.move() == excluded) continue;
            const bool quiet = !b.isCapture(m) && m.typeOf() != Move::PROMOTION;
            const int ext = (m.move() == tt_move) ? singular_ext : 0;

            // Late move pruning: at shallow depth skip late quiet moves outright.
            if (!pv_node && !in_check && quiet && best > -MATE_BOUND && depth <= 3 &&
                searched >= (improving ? lmp_imp_base + depth * depth * 2 : lmp_base + depth * depth))
                continue;

            // SEE pruning: at shallow depth skip moves that lose too much material outright.
            if (!pv_node && !in_check && searched > 0 && best > -MATE_BOUND && depth <= see_depth &&
                !see_ge(b, m, quiet ? -see_quiet * depth : -see_noisy * depth))
                continue;

            ss[ply].piece = moved_piece(b, m);
            ss[ply].to = dest(m);
            make(b, m);
            const bool gives_check = b.inCheck();
            int s;
            if (searched == 0) {
                s = -negamax(b, depth - 1 + ext, -beta, -alpha, ply + 1, true);
            } else {
                int R = 0;
                if (depth >= 3 && searched >= (pv_node ? 3 : 2) && quiet && !in_check && !gives_check) {
                    R = LMR[std::min(depth, 63)][std::min(searched, 63)];
                    if (pv_node) R--;
                    if (!improving) R++;
                    if (m.move() == killers[ply][0] || m.move() == killers[ply][1]) R--;
                    R = std::clamp(R, 0, depth - 2);
                }
                s = -negamax(b, depth - 1 - R, -alpha - 1, -alpha, ply + 1, true);
                if (s > alpha && R > 0) s = -negamax(b, depth - 1 + ext, -alpha - 1, -alpha, ply + 1, true);
                if (s > alpha && s < beta) s = -negamax(b, depth - 1 + ext, -beta, -alpha, ply + 1, true);
            }
            unmake(b, m);
            if (stop) return 0;
            searched++;
            if (quiet && n_quiets < 256) quiets_tried[n_quiets++] = m;

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
                            if (ply > 0 && ss[ply - 1].piece >= 0) counter[ss[ply - 1].piece][ss[ply - 1].to] = m.move();
                            // Reward the cutoff move, penalise the quiets tried before it, in the
                            // butterfly table and both continuation tables.
                            const int bonus = std::min(depth * depth, hist_max_bonus);
                            for (int q = 0; q < n_quiets; q++) {
                                const Move qm = quiets_tried[q];
                                const int delta = (qm == m) ? bonus : -bonus;
                                gravity(history[c][qm.from().index()][qm.to().index()], delta);
                                const int pc = moved_piece(b, qm), to = dest(qm);
                                if (auto* e = cont_entry(ply, 1, pc, to)) gravity16(*e, delta);
                                if (auto* e = cont_entry(ply, 2, pc, to)) gravity16(*e, delta);
                            }
                        }
                        break;
                    }
                }
            }
        }

        if (best == -INF) return alpha;   // every move skipped (excluded / pruned)
        const Bound bound = best >= beta ? BOUND_LOWER : (best > orig_alpha ? BOUND_EXACT : BOUND_UPPER);
        if (!excluded) tt_store(key, score_to_tt(best, ply), best_move.move(), depth, bound);
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
    std::uint64_t nodes = 0;  // spent on this move in the last iteration (main thread)
    std::vector<Move> pv;
};

// Time management: how much of the clock-based soft limit to use after an iteration.
// More when the best move just changed, when the best move took a small share of the
// nodes (other moves were close), or when the score dropped; less when all is settled.
double soft_scale(int stable_iters, double best_share, int score_drop) {
    static const double stability[] = {2.5, 1.2, 0.9, 0.8, 0.75};
    const double s = stability[std::clamp(stable_iters, 0, 4)];
    const double share = (1.62 - best_share) * 1.4;
    const double drop = 1.0 + std::clamp(score_drop, 0, 100) / 200.0;
    return 0.8 * s * share * drop;   // 0.8: about the same average time as a fixed soft limit
}

struct RootResult {
    std::vector<RootMove> moves;  // sorted best first
    int depth = 0;
    std::uint64_t nodes = 0;
    long long ms = 0;
};

Search S;                                    // main search thread
std::vector<std::unique_ptr<Search>> HELPERS;  // extra Lazy SMP threads
int g_threads = 1;

// Policy probabilities for the root moves; `elo` is the rating whose play to imitate
// (both players are assumed to be about that strength).
std::vector<float> root_priors(const Board& board, const Movelist& moves, int elo) {
    std::vector<float> pri(moves.size(), 1.0f / std::max(1, moves.size()));
    if (!policy::NET.loaded || moves.empty()) return pri;
    const int r = std::clamp(elo, 600, 2800);
    auto logits = policy::NET.forward(board, r, r);
    const bool white = board.sideToMove() == chess::Color::WHITE;
    float mx = -1e30f;
    for (int i = 0; i < moves.size(); i++) mx = std::max(mx, logits[policy::NET.move_index(moves[i], white)]);
    float sum = 0;
    for (int i = 0; i < moves.size(); i++) {
        pri[i] = std::exp(logits[policy::NET.move_index(moves[i], white)] - mx);
        sum += pri[i];
    }
    for (auto& p : pri) p /= sum;
    return pri;
}

// Iterative deepening on one thread. Helper threads run the same loop (single PV,
// staggered start depth) purely to fill the shared transposition table.
RootResult iterate(Search& T, Board board, std::vector<RootMove> rms, int max_depth, long long time_ms, int multipv,
                   int first_depth, bool verbose) {
    RootResult res;
    T.nodes = 0;
    T.time_limit_ms = std::max(1LL, time_ms);
    T.node_limit = T.is_main ? g_limits.nodes : 0;
    const long long soft_ms = g_limits.infinite ? LLONG_MAX : (g_limits.soft_ms >= 0 ? g_limits.soft_ms : time_ms * 6 / 10);
    T.start = std::chrono::steady_clock::now();
    T.nn.reset(board);
    std::memset(T.killers, 0, sizeof(T.killers));
    for (auto& side : T.history)
        for (auto& row : side)
            for (auto& h : row) h /= 2;
    for (auto& h : T.cont) h /= 2;

    const int K = std::max(1, std::min(multipv, static_cast<int>(rms.size())));
    res.moves = rms;
    const bool dynamic = T.is_main && g_limits.dynamic && !g_limits.infinite;
    int stable_iters = 0, prev_score = 0;
    Move prev_best = Move(Move::NO_MOVE);

    for (int depth = first_depth; depth <= max_depth; depth++) {
        std::vector<RootMove> cur = rms;
        std::vector<int> exact_scores;  // sorted descending
        bool done = true;
        size_t completed = 0;

        for (size_t i = 0; i < cur.size(); i++) {
            RootMove& rm = cur[i];
            int alpha = static_cast<int>(exact_scores.size()) >= K ? exact_scores[K - 1] : -INF;
            T.ss[0].eval = -INF;
            T.ss[0].piece = Search::moved_piece(board, rm.move);
            T.ss[0].to = Search::dest(rm.move);
            T.make(board, rm.move);
            const std::uint64_t nodes_before = T.nodes;
            int s;
            if (alpha == -INF && depth >= 5 && rm.exact && std::abs(rm.score) < MATE_BOUND) {
                // Aspiration window around last iteration's score; widen on a fail.
                int delta = asp_delta, lo = rm.score - delta, hi = rm.score + delta;
                while (true) {
                    s = -T.negamax(board, depth - 1, -hi, -lo, 1, true);
                    if (T.stop) break;
                    if (s > lo && s < hi) break;
                    delta *= 2;
                    if (delta > 400) { lo = -INF; hi = INF; continue; }
                    if (s <= lo) lo = std::max(-INF, s - delta);
                    else hi = std::min(INF, s + delta);
                }
            } else if (alpha == -INF) {
                s = -T.negamax(board, depth - 1, -INF, INF, 1, true);
            } else {
                s = -T.negamax(board, depth - 1, -alpha - 1, -alpha, 1, true);
                if (!T.stop && s > alpha) s = -T.negamax(board, depth - 1, -INF, -alpha, 1, true);
            }
            T.unmake(board, rm.move);
            if (T.stop) { done = false; break; }
            completed++;
            rm.nodes = T.nodes - nodes_before;

            rm.score = s;
            rm.exact = alpha == -INF || s > alpha;
            if (rm.exact) {
                rm.pv.assign(1, rm.move);
                for (int k = 1; k < T.pv_len[1]; k++) rm.pv.push_back(T.pv[1][k]);
                exact_scores.insert(std::upper_bound(exact_scores.begin(), exact_scores.end(), s, std::greater<int>()), s);
            }
        }
        auto by_score = [](const RootMove& a, const RootMove& b) {
            if (a.exact != b.exact) return a.exact;
            return a.score > b.score;
        };
        if (!done) {
            // Out of time mid-iteration. Once the previous top moves have been re-searched
            // at the new depth their scores are trustworthy, so keep that partial result
            // (it often catches a better move found late in the iteration).
            if (T.is_main && completed >= static_cast<size_t>(K)) {
                std::stable_sort(cur.begin(), cur.begin() + completed, by_score);
                res.moves = cur;
            }
            break;
        }

        std::stable_sort(cur.begin(), cur.end(), by_score);
        rms = cur;
        res.moves = cur;
        res.depth = depth;
        res.ms = T.elapsed_ms();

        if (verbose) {
            std::cerr << "[engine] depth " << depth << " " << res.ms << " ms, best="
                      << chess::uci::moveToUci(cur[0].move) << " score=" << cur[0].score << "\n";
        }
        if (!T.is_main) continue;
        res.nodes = T.nodes;
        if (g_on_iteration) g_on_iteration(res);
        if (g_limits.infinite) continue;
        if (std::abs(cur[0].score) >= MATE_BOUND && depth >= 4) break;     // forced mate found
        // Don't start an iteration that probably can't re-search the best move in time.
        double scale = 1.0;
        if (dynamic) {
            stable_iters = cur[0].move == prev_best ? stable_iters + 1 : 0;
            std::uint64_t total = 0;
            for (const auto& rm : cur) total += rm.nodes;
            const double share = total ? static_cast<double>(cur[0].nodes) / total : 1.0;
            const bool mate = std::abs(cur[0].score) >= MATE_BOUND || std::abs(prev_score) >= MATE_BOUND;
            if (depth >= 6) scale = soft_scale(stable_iters, share, mate ? 0 : prev_score - cur[0].score);
            prev_best = cur[0].move;
            prev_score = cur[0].score;
        }
        if (res.ms > soft_ms * scale) break;
    }
    res.nodes = T.nodes;
    res.ms = T.elapsed_ms();
    return res;
}

// `only`, when non-empty, restricts the root moves (used to keep tablebase-preserving moves).
RootResult search_root(Board board, int max_depth, long long time_ms, int multipv, bool verbose,
                       const std::vector<Move>& only = {}, int elo = 2800) {
    RootResult res;
    tt_new_search();
    Movelist all, legal;
    chess::movegen::legalmoves(all, board);
    for (const auto& m : all)
        if (only.empty() || std::find(only.begin(), only.end(), m) != only.end()) legal.add(m);
    if (legal.empty()) return res;

    auto pri = root_priors(board, legal, elo);
    std::vector<RootMove> rms;
    for (int i = 0; i < legal.size(); i++) {
        RootMove rm;
        rm.move = legal[i];
        rm.prior = pri[i];
        rms.push_back(rm);
    }
    std::stable_sort(rms.begin(), rms.end(), [](const RootMove& a, const RootMove& b) { return a.prior > b.prior; });

    while (static_cast<int>(HELPERS.size()) < g_threads - 1) {
        HELPERS.push_back(std::make_unique<Search>());
        HELPERS.back()->is_main = false;
    }

    g_stop = false;
    std::vector<std::thread> threads;
    std::vector<std::uint64_t> helper_nodes(g_threads, 0);
    for (int t = 1; t < g_threads; t++) {
        threads.emplace_back([&, t] {
            Search& H = *HELPERS[t - 1];
            helper_nodes[t] = iterate(H, board, rms, max_depth, time_ms, 1, 1 + (t & 1), false).nodes;
        });
    }
    res = iterate(S, board, rms, max_depth, time_ms, multipv, 1, verbose);
    g_stop = true;   // main is done: release the helpers
    for (auto& th : threads) th.join();
    for (auto n : helper_nodes) res.nodes += n;
    res.ms = S.elapsed_ms();
    if (verbose) {
        std::cerr << "[engine] " << g_threads << " thread(s), " << res.nodes << " nodes, "
                  << (res.nodes * 1000 / std::max(1LL, res.ms)) << " nps\n";
    }
    return res;
}

// ---------------------------------------------------------------------------
// Elo handling
// ---------------------------------------------------------------------------
// Full strength at >= 2900. Below that the search depth is capped and the move
// is sampled among near-best candidates, weighted by how good the search thinks
// each one is and by the CNN's "human intuition" for it.
int depth_cap_for_elo(int elo) {
    if (elo >= 2900) return 99;
    return std::clamp(elo < 1800 ? (elo - 1100) / 100 : (elo - 600) / 160, 1, 20);   // 1400: 3, 1800: 7, 2600: 12
}

size_t choose_move(const RootResult& r, int elo, std::mt19937& rng) {
    if (elo >= 2900 || r.moves.size() < 2) return 0;
    // Sampling temperature in centipawns, interpolated between points calibrated with
    // scripts/calibrate_elo.py against Stockfish's UCI_Elo (1400 and up; below that extrapolated).
    static const double pts[][2] = {{600, 700}, {800, 460}, {1400, 355}, {1800, 172}, {2200, 83}, {2600, 55}, {2900, 21}};
    double temp = 700;
    for (size_t i = 1; i < std::size(pts); i++) {
        if (elo <= pts[i][0]) {
            const double t = std::clamp((elo - pts[i - 1][0]) / (pts[i][0] - pts[i - 1][0]), 0.0, 1.0);
            temp = pts[i - 1][1] + t * (pts[i][1] - pts[i - 1][1]);
            break;
        }
    }
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

// ---------------------------------------------------------------------------
// Human move probabilities (rating-aware policy, no search)
// ---------------------------------------------------------------------------
// The board after `moves`; throws on an illegal or unparsable move.
Board board_after(const std::string& fen, const std::vector<std::string>& moves, std::vector<Move>* played = nullptr) {
    Board board(fen);
    for (const auto& u : moves) {
        const Move m = chess::uci::uciToMove(board, u);
        Movelist legal;
        chess::movegen::legalmoves(legal, board);
        if (m == Move(Move::NO_MOVE) || std::find(legal.begin(), legal.end(), m) == legal.end())
            throw std::runtime_error("illegal move " + u);
        if (played) played->push_back(m);
        board.makeMove(m);
    }
    return board;
}

// humanprobs: every legal move with its probability for a player of `elo`, most likely first.
void run_humanprobs(const std::string& fen, const std::vector<std::string>& moves, int elo, std::ostream& out) {
    const Board board = board_after(fen, moves);
    Movelist legal;
    chess::movegen::legalmoves(legal, board);
    const auto pri = root_priors(board, legal, elo);
    std::vector<int> idx(legal.size());
    for (int i = 0; i < legal.size(); i++) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return pri[a] > pri[b]; });
    out << "HUMAN_PROBS:" << elo << "\n";
    for (int i : idx) out << "MOVE:" << chess::uci::moveToUci(legal[i]) << "|P:" << pri[i] << "\n";
    out << "END" << std::endl;
}

// humangame: for each move of a game (optionally one side's only), the probability and rank
// of the move actually played, for players of each rating in `elos` (comma-separated).
void run_humangame(const std::string& fen, const std::vector<std::string>& moves, const std::string& elos_csv,
                   const std::string& side, std::ostream& out) {
    std::vector<int> elos;
    std::istringstream es(elos_csv);
    for (std::string e; std::getline(es, e, ',');)
        if (!e.empty()) elos.push_back(std::stoi(e));
    if (elos.empty()) throw std::runtime_error("no ratings given");
    if (side != "w" && side != "b" && side != "all") throw std::runtime_error("side must be w, b or all");

    std::vector<Move> played;
    board_after(fen, moves, &played);   // validate the whole game first
    Board board(fen);
    out << "ELOS:" << elos_csv << "\n";
    for (size_t ply = 0; ply < played.size(); ply++) {
        const bool white = board.sideToMove() == chess::Color::WHITE;
        if (side == "all" || (side == "w") == white) {
            Movelist legal;
            chess::movegen::legalmoves(legal, board);
            const int k = static_cast<int>(std::find(legal.begin(), legal.end(), played[ply]) - legal.begin());
            std::string ps, rs;
            for (size_t e = 0; e < elos.size(); e++) {
                const auto pri = root_priors(board, legal, elos[e]);
                const int rank = 1 + static_cast<int>(std::count_if(pri.begin(), pri.end(),
                                                                     [&](float p) { return p > pri[k]; }));
                std::ostringstream p;
                p << pri[k];
                ps += (e ? "," : "") + p.str();
                rs += (e ? "," : "") + std::to_string(rank);
            }
            out << "PLY:" << ply + 1 << "|SIDE:" << (white ? 'w' : 'b') << "|MOVE:" << moves[ply]
                << "|LEGAL:" << legal.size() << "|P:" << ps << "|RANK:" << rs << "\n";
        }
        board.makeMove(played[ply]);
    }
    out << "END" << std::endl;
}

void run_go(const std::string& fen, const std::vector<std::string>& moves, int elo, int max_depth, long long total_ms,
            std::ostream& out) {
    static std::mt19937 rng(static_cast<unsigned>(std::chrono::steady_clock::now().time_since_epoch().count()));
    Board board(fen);
    // Replaying the game's moves gives the board its history, so the search sees repetitions.
    for (const auto& u : moves) {
        Move m = chess::uci::uciToMove(board, u);
        if (m == Move(Move::NO_MOVE)) break;
        board.makeMove(m);
    }
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

    // Tablebase position: only search moves that keep the theoretical result. Below
    // ~2200 the bot plays endgames by its own (imperfect) judgement, like a human would.
    tb::RootInfo tbi;
    const bool use_tb = elo >= 2200 && tb::probe_root(board, tbi);
    std::vector<Move> keep;
    if (use_tb)
        for (const auto& m : tbi.moves)
            if (m.wdl == tbi.wdl) keep.push_back(m.move);

    RootResult r = search_root(board, depth, main_ms, multipv, true, keep, elo);
    const bool tb_win = use_tb && tbi.wdl == TB_WIN;
    if (tb_win) {
        // Winning: the tablebase's move converts fastest (search alone can shuffle), so it
        // goes first, for the bot and for the tutor's top line.
        auto it = std::find_if(r.moves.begin(), r.moves.end(), [&](const RootMove& m) { return m.move == tbi.best; });
        if (it != r.moves.end()) {
            std::rotate(r.moves.begin(), it, it + 1);
            r.moves[0].score = TB_WIN_SCORE - 1;
            r.moves[0].exact = true;
        }
    }
    const size_t pick = tb_win ? 0 : choose_move(r, elo, rng);
    const RootMove& chosen = r.moves[pick];

    const bool white = board.sideToMove() == chess::Color::WHITE;
    auto to_white_pawns = [white](int s) {
        s = display_cp(s);
        return (white ? s : -s) / 100.0;
    };

    out << "BEST_MOVE:" << chess::uci::moveToUci(chosen.move) << "\n";
    // If time ran out before depth 1 finished there are no scores; report the static eval.
    int eval_stm = r.depth > 0 ? chosen.score : S.nn.evaluate(board);
    if (use_tb) eval_stm = tbi.wdl == TB_WIN ? TB_WIN_SCORE : tbi.wdl == TB_LOSS ? -TB_WIN_SCORE : 0;
    out << "EVAL:" << to_white_pawns(eval_stm) << "\n";
    if (std::abs(eval_stm) >= MATE_BOUND) {
        // Mate in N full moves, positive when White is the one mating.
        const int n = (MATE - std::abs(eval_stm) + 1) / 2;
        out << "MATE:" << (((eval_stm > 0) == white) ? n : -n) << "\n";
    }
    if (use_tb) {
        // Theoretical result from the tablebase, White's point of view: 1 win, 0 draw, -1 loss.
        const int stm_res = (tbi.wdl == TB_WIN) ? 1 : (tbi.wdl == TB_LOSS) ? -1 : 0;
        out << "TABLEBASE:" << (white ? stm_res : -stm_res) << "\n";
    }
    out << "DEPTH_REACHED:" << r.depth << "\n";

    // "|EVAL:<pawns, White's view>[|MATE:<n>]" for a line's final evaluation; only exact scores.
    auto line_eval = [](const RootMove& m, bool white_to_move) {
        if (!m.exact || m.score <= -INF) return std::string();
        std::ostringstream os;
        const int s = display_cp(m.score);
        os << "|EVAL:" << (white_to_move ? s : -s) / 100.0;
        if (std::abs(m.score) >= MATE_BOUND) {
            const int n = (MATE - std::abs(m.score) + 1) / 2;
            os << "|MATE:" << (((m.score > 0) == white_to_move) ? n : -n);
        }
        return os.str();
    };

    out << "TUTOR_DATA:\n";
    for (size_t i = 0; i < r.moves.size() && i < 3; i++) {
        const RootMove& m = r.moves[i];
        out << "RANK:" << i + 1 << "|MOVE:" << chess::uci::moveToUci(m.move)
            << "|IDEA:" << pv_string(m.pv.empty() ? std::vector<Move>{m.move} : m.pv) << line_eval(m, white) << "\n";
    }

    // Top replies for the human, on the position after the bot's move.
    Board after = board;
    after.makeMove(chosen.move);
    RootResult u = search_root(after, std::max(1, r.depth - 1), supp_ms, 3, false, {}, elo);
    out << "USER_TUTOR_DATA:\n";
    for (size_t i = 0; i < u.moves.size() && i < 3; i++) {
        const RootMove& m = u.moves[i];
        out << "USER_RANK:" << i + 1 << "|MOVE:" << chess::uci::moveToUci(m.move)
            << "|IDEA:" << pv_string(m.pv.empty() ? std::vector<Move>{m.move} : m.pv) << line_eval(m, !white)
            << "\n";
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

std::string g_syzygy;   // --syzygy <dir>; "none" disables

void init_tablebases(const char* argv0) {
    std::string path = g_syzygy;
    if (path.empty()) {
        if (const char* env = std::getenv("SYZYGY_PATH")) path = env;
    }
    if (path.empty()) {
        // tools/syzygy next to assets/ (same search as the networks)
        std::string probe = find_asset("../tools/syzygy/KQvK.rtbw", argv0);
        for (const char* c : {"tools/syzygy", "../tools/syzygy", "../../tools/syzygy"})
            if (std::ifstream(std::string(c) + "/KQvK.rtbw", std::ios::binary)) { path = c; break; }
        if (path.empty() && std::ifstream(probe, std::ios::binary)) path = probe.substr(0, probe.size() - 10);
    }
    if (path == "none") return;
    if (tb::init(path)) std::cerr << "[engine] Syzygy tablebases: up to " << tb::largest() << " pieces (" << path << ")\n";
}

bool load_networks(const char* argv0) {
    const std::string nn_path = find_asset("nnue.bin", argv0);
    if (!nnue::NET.load(nn_path)) {
        std::cerr << "[engine] could not load " << nn_path << "\n";
        std::cout << "BEST_MOVE:ERROR_LOADING_MODEL" << std::endl;
        return false;
    }
    // Prefer the rating-aware policy; fall back to the original CNN's.
    const std::string elo_path = find_asset("policy_elo.bin", argv0);
    if (!policy::NET.load(elo_path)) {
        const std::string pol_path = find_asset("policy_cnn.bin", argv0);
        if (!policy::NET.load(pol_path))
            std::cerr << "[engine] policy net not found (" << pol_path << "), continuing without it\n";
    }
    return true;
}

int selftest(const char* argv0) {
    // Reference logits written by the exporter: "fen[|mover_elo|opp_elo]" then "idx:logit ...".
    std::ifstream f(find_asset(policy::NET.version == 2 ? "policy_elo_test.txt" : "policy_cnn_test.txt", argv0));
    std::string line, ref;
    int fails = 0, cases = 0;
    while (std::getline(f, line) && std::getline(f, ref)) {
        cases++;
        std::string fen = line;
        int me = 2000, opp = 2000;
        if (line.find('|') != std::string::npos) {
            fen = line.substr(0, line.find('|'));
            std::sscanf(line.c_str() + line.find('|') + 1, "%d|%d", &me, &opp);
        }
        Board b(fen);
        auto t0 = std::chrono::steady_clock::now();
        auto logits = policy::NET.forward(b, me, opp);
        auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::istringstream ss(ref);
        std::string tok;
        while (ss >> tok) {
            int idx = std::stoi(tok.substr(0, tok.find(':')));
            float want = std::stof(tok.substr(tok.find(':') + 1));
            if (std::abs(logits[idx] - want) > 1e-3f * std::max(1.0f, std::abs(want))) {
                fails++;
                std::cout << "MISMATCH idx " << idx << " got " << logits[idx] << " want " << want << "\n";
            }
        }
        nnue::Stack st;
        st.reset(b);
        std::cout << "policy ok-check (" << ms << " ms), nnue eval " << st.evaluate(b) << " cp (stm)  " << fen << "\n";
    }
    if (!cases) { std::cout << "no policy reference file\n"; fails++; }
    std::cout << "policy v" << policy::NET.version << ": " << (fails ? "SELFTEST FAILED" : "SELFTEST PASSED") << "\n";
    return fails ? 1 : 0;
}

int bench(long long ms, int depth = 99) {
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
        tt_clear();
        RootResult r = search_root(Board(fen), depth, ms, 1, false);
        total_nodes += r.nodes;
        total_ms += r.ms;
        std::cout << "depth " << r.depth << "  best " << chess::uci::moveToUci(r.moves[0].move) << "  score "
                  << r.moves[0].score << "  " << r.nodes << " nodes  " << r.ms << " ms   " << fen << "\n";
    }
    std::cout << "TOTAL " << total_nodes << " nodes, " << (total_nodes * 1000 / std::max(1LL, total_ms)) << " nps\n";
    return 0;
}

#include "tests.hpp"
#include "uci.hpp"
#include "datagen.hpp"

int main(int argc, char* argv_in[]) {
    init_lmr();
    std::ios::sync_with_stdio(false);
    // "--threads N" may appear anywhere; strip it before the positional arguments are read.
    std::vector<char*> args;
    for (int i = 0; i < argc; i++) {
        if (std::string(argv_in[i]) == "--threads" && i + 1 < argc) {
            g_threads = std::clamp(std::atoi(argv_in[++i]), 1, 256);
            continue;
        }
        if (std::string(argv_in[i]) == "--syzygy" && i + 1 < argc) {
            g_syzygy = argv_in[++i];
            continue;
        }
        args.push_back(argv_in[i]);
    }
    argc = static_cast<int>(args.size());
    char** argv = args.data();

    if (!load_networks(argv[0])) return 2;
    init_tablebases(argv[0]);
    // No arguments (how chess GUIs and match tools start an engine) or --uci: UCI protocol.
    if (argc < 2 || std::string(argv[1]) == "--uci") return uci_mode::run();

    std::string mode = argv[1];
    if (mode == "--selftest") return selftest(argv[0]);
    if (mode == "--test") return tests::run(argv[0]);
    if (mode == "--eval") {   // static NNUE eval of each FEN argument (side to move's view)
        for (int i = 2; i < argc; i++) {
            Board b(argv[i]);
            S.nn.reset(b);
            std::cout << S.nn.evaluate(b) << " cp  " << argv[i] << "\n";
        }
        return 0;
    }
    if (mode == "--bench") {
        const std::string a = argc >= 3 ? argv[2] : "2000";
        return a[0] == 'd' ? bench(24LL * 3600 * 1000, std::stoi(a.substr(1))) : bench(std::stoll(a));
    }
    if (mode == "--datagen") {   // --datagen <out_dir> [games] [nodes] [seed]
        if (argc < 3) { std::cerr << "usage: engine --datagen <out_dir> [games] [nodes] [seed]\n"; return 1; }
        return datagen::run(argv[2], argc >= 4 ? std::stoll(argv[3]) : 1'000'000,
                            argc >= 5 ? std::stoull(argv[4]) : 5000,
                            argc >= 6 ? static_cast<unsigned>(std::stoul(argv[5])) : std::random_device{}());
    }

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
                tt_clear();
                std::cout << "OK" << std::endl;
                continue;
            }
            // Rest of the line: "<fen> [moves m1 m2 ...]"
            auto read_position = [&ss](std::string& fen, std::vector<std::string>& moves) {
                std::string rest;
                std::getline(ss, rest);
                fen = rest;
                const auto mpos = rest.find(" moves");
                if (mpos != std::string::npos) {
                    fen = rest.substr(0, mpos);
                    std::istringstream ms_in(rest.substr(mpos + 6));
                    for (std::string u; ms_in >> u;) moves.push_back(u);
                }
                fen.erase(0, fen.find_first_not_of(' '));
            };
            if (cmd == "go" || cmd == "humanprobs" || cmd == "humangame") {
                std::string fen;
                std::vector<std::string> moves;
                try {
                    if (cmd == "go") {
                        int elo = 3000, depth = 99;
                        long long ms = 3000;
                        ss >> elo >> depth >> ms;
                        read_position(fen, moves);
                        run_go(fen, moves, elo, depth, ms, std::cout);
                    } else if (cmd == "humanprobs") {
                        int elo = 1500;
                        ss >> elo;
                        read_position(fen, moves);
                        run_humanprobs(fen, moves, elo, std::cout);
                    } else {
                        std::string elos, side;
                        ss >> elos >> side;
                        read_position(fen, moves);
                        run_humangame(fen, moves, elos, side, std::cout);
                    }
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
    run_go(fen, {}, elo, depth, ms, std::cout);
    return 0;
}
