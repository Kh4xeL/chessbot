/**
 * Hybrid AlphaZero C++ Chess Engine
 * * This engine combines the deep positional evaluation of a PyTorch Convolutional Neural Network
 * (trained on millions of positions) with the raw tactical calculation power of a traditional
 * Alpha-Beta Minimax search tree.
 */

#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <atomic>
#include <algorithm>
#include "../include/chess.hpp"

// --- Global Controls ---
std::atomic<bool> out_of_time(false);
auto start_time = std::chrono::high_resolution_clock::now();
int time_limit_ms = 10000;

// --- Transposition Table (TT) ---
// Acts as the engine's memory. It caches previously calculated board states
// to prevent redundant calculations and extract the Principal Variation (PV).
const int EXACT = 0;
const int LOWERBOUND = 1;
const int UPPERBOUND = 2;
const int TT_SIZE = 1048576;

struct TTEntry {
    uint64_t hash;
    float score;
    int depth;
    int flag;
    chess::Move best_move;
};
std::vector<TTEntry> TT(TT_SIZE);

// --- NN Evaluation Cache ---
// The CNN forward pass is the single most expensive operation in the search
// (~90% of CPU time). Many leaf positions are reached repeatedly via different
// move orders (transpositions), so memoising the value head by zobrist hash
// typically eliminates 30-70% of NN calls. We cache only the value (the policy
// is only consumed at the root, not at leaves).
struct NNEntry {
    uint64_t hash;
    float value;
};
const int NN_CACHE_SIZE = 524288; // ~4 MB, power of two
std::vector<NNEntry> NN_CACHE(NN_CACHE_SIZE);

// --- Node counter ---
// Counts minimax invocations across the current search. Used purely for the
// stderr diagnostic line; the actual time-bounding is the per-node chrono
// check in minimax itself.
uint64_t node_count = 0;

// --- PyTorch Neural Network Configuration ---
torch::jit::script::Module module;
torch::Tensor elo_tensor;
torch::Tensor global_board_tensor;
float* global_board_data;
std::vector<torch::jit::IValue> global_inputs;

/**
 * Converts a traditional chess board into a 13-Channel AlphaZero Tensor format.
 * Channels 0-5: White pieces (Pawn, Knight, Bishop, Rook, Queen, King)
 * Channels 6-11: Black pieces
 * Channel 12: Turn indicator (1.0 for White, 0.0 for Black)
 */
void update_global_tensor(const chess::Board& board) {
    std::fill(global_board_data, global_board_data + (13 * 8 * 8), 0.0f);

    for (int i = 0; i < 64; i++) {
        chess::Square sq = chess::Square(i);
        chess::Piece piece = board.at(sq);
        if (piece != chess::Piece::NONE) {
            int val = 0;
            if (piece.type() == chess::PieceType::PAWN) val = 1;
            else if (piece.type() == chess::PieceType::KNIGHT) val = 2;
            else if (piece.type() == chess::PieceType::BISHOP) val = 3;
            else if (piece.type() == chess::PieceType::ROOK) val = 4;
            else if (piece.type() == chess::PieceType::QUEEN) val = 5;
            else if (piece.type() == chess::PieceType::KING) val = 6;

            if (piece.color() == chess::Color::BLACK) val += 6;

            int rank = i / 8;
            int file = i % 8;
            int channel = val - 1;
            global_board_data[(channel * 64) + ((7 - rank) * 8) + file] = 1.0f;
        }
    }

    float turn_val = (board.sideToMove() == chess::Color::WHITE) ? 1.0f : 0.0f;
    for (int i = 0; i < 64; i++) {
        global_board_data[(12 * 64) + i] = turn_val;
    }
}

/**
 * Queries the CNN to get the Policy (4096 logits representing move intuition)
 * and the Value (-1 to 1 representing the board evaluation).
 */
void get_nn_outputs(const chess::Board& board, std::vector<float>& policy_logits, float& value) {
    update_global_tensor(board);
    auto outputs = module.forward(global_inputs).toTuple();
    at::Tensor policy_tensor = outputs->elements()[0].toTensor();
    value = outputs->elements()[1].toTensor().item<float>();

    // Copy the 4096 logits out of the tensor
    if (!policy_logits.empty()) {
        policy_logits.assign(policy_tensor.data_ptr<float>(), policy_tensor.data_ptr<float>() + 4096);
    }
}

// Counts pieces to determine if we are in the Midgame or Endgame.
// Used later for dynamic Policy Pruning.
int count_active_pieces(const chess::Board& board) {
    int count = 0;
    chess::PieceType types[] = { chess::PieceType::PAWN, chess::PieceType::KNIGHT, chess::PieceType::BISHOP, chess::PieceType::ROOK, chess::PieceType::QUEEN };
    for (int i = 0; i < 5; i++) {
        count += board.pieces(types[i], chess::Color::WHITE).count();
        count += board.pieces(types[i], chess::Color::BLACK).count();
    }
    return count;
}

// Classical Material Evaluation (Fallback heuristic)
float evaluate_material(const chess::Board& board) {
    float score = 0;
    chess::PieceType types[] = { chess::PieceType::PAWN, chess::PieceType::KNIGHT, chess::PieceType::BISHOP, chess::PieceType::ROOK, chess::PieceType::QUEEN };
    int values[] = {1, 3, 3, 5, 9};

    for (int i = 0; i < 5; i++) {
        score += board.pieces(types[i], chess::Color::WHITE).count() * values[i];
        score -= board.pieces(types[i], chess::Color::BLACK).count() * values[i];
    }
    return score;
}

// Classical Tactical Move Ordering (Speeds up Alpha-Beta Pruning).
//
// Pure tactical scoring: captures (MVV-LVA), promotions, and checking moves.
// Killer/history heuristics were tried but added per-node overhead without
// gaining effective depth at the 2-3 ply search horizon this engine reaches,
// so they were reverted in favour of keeping score_move as cheap as possible.
int score_move(const chess::Board& board, const chess::Move& move) {
    int score = 0;
    if (board.isCapture(move)) {
        chess::Piece victim = board.at(move.to());
        chess::Piece attacker = board.at(move.from());
        if (victim != chess::Piece::NONE && attacker != chess::Piece::NONE) {
            score += 100 + (10 * static_cast<int>(victim.type())) - static_cast<int>(attacker.type());
        } else {
            score += 100;
        }
    }
    // NOTE: the chess library's `promotionType()` returns garbage for
    // non-promotion moves (the comment in chess.hpp says "should only be used
    // if typeOf() returns PROMOTION"). The original code used
    // `promotionType() != PieceType::NONE` which is ALWAYS true — it gave
    // every quiet move a fake +90 bonus. That was a harmless no-op in the
    // original (all quiets get the same bonus, ordering preserved) but it
    // made `is_forcing_move` mis-classify every move as forcing, which broke
    // root pruning. Use the type discriminator instead.
    if (move.typeOf() == chess::Move::PROMOTION) score += 90;

    // Prioritise checking moves: they are the only way to deliver mate.
    // board.givesCheck(move) uses bitboard math (no board copy / makeMove),
    // which is roughly an order of magnitude faster than the naive approach.
    if (board.givesCheck(move) != chess::CheckType::NO_CHECK) score += 80;
    return score;
}

// Returns true if `move` is a "forcing" move that must never be pruned at the
// root, regardless of NN policy ranking: captures, promotions, and checks.
// These are exactly the moves that can deliver (or escape) mate.
bool is_forcing_move(const chess::Board& board, const chess::Move& move) {
    if (board.isCapture(move)) return true;
    // See score_move: promotionType() is garbage outside of promotion moves;
    // must check the type discriminator.
    if (move.typeOf() == chess::Move::PROMOTION) return true;
    return board.givesCheck(move) != chess::CheckType::NO_CHECK;
}

// In-place move ordering. Uses Move::setScore + Movelist's contiguous storage so
// no heap allocation occurs per node. Sorting Movelist directly is safe because
// chess::Movelist::iterator is just a Move*.
void order_moves(const chess::Board& board, chess::Movelist& moves) {
    for (int i = 0; i < moves.size(); i++) {
        moves[i].setScore(static_cast<std::int16_t>(score_move(board, moves[i])));
    }
    std::sort(moves.begin(), moves.end(), [](const chess::Move& a, const chess::Move& b) {
        return a.score() > b.score();
    });
}

/**
 * Quiescence Search: Evaluates noisy positions (like active captures) deeply
 * to ensure the engine doesn't suffer from the Horizon Effect mid-trade.
 *
 * `qply` caps recursion to prevent quiescence explosions in capture-heavy
 * positions (multiple recaptures around the same square).
 */
const int QSEARCH_MAX_PLY = 8;
float quiescence(chess::Board& board, float alpha, float beta, bool maximizing_player, float base_nn_score, int qply = 0) {
    // Bail if a parent minimax call already set out_of_time. Quiescence itself
    // doesn't check the wall clock — the precise per-node check in minimax
    // bounds the budget; quiescence inherits via this flag.
    if (out_of_time) return 0.0f;

    // 1 Pawn = 1.0. We multiply the NN score (from -1 to 1) by 4.0.
    // This allows the AI to value positional dominance up to 4 pawns worth of advantage.
    float stand_pat = evaluate_material(board) + (base_nn_score * 4.0f);

    if (qply >= QSEARCH_MAX_PLY) return stand_pat;

    if (maximizing_player) {
        if (stand_pat >= beta) return beta;
        if (alpha < stand_pat) alpha = stand_pat;

        chess::Movelist moves;
        chess::movegen::legalmoves<chess::movegen::MoveGenType::CAPTURE>(moves, board);
        order_moves(board, moves);

        for (const auto& move : moves) {
            board.makeMove(move);
            float score = quiescence(board, alpha, beta, false, base_nn_score, qply + 1);
            board.unmakeMove(move);
            if (score >= beta) return beta;
            if (score > alpha) alpha = score;
        }
        return alpha;
    } else {
        if (stand_pat <= alpha) return alpha;
        if (beta > stand_pat) beta = stand_pat;

        chess::Movelist moves;
        chess::movegen::legalmoves<chess::movegen::MoveGenType::CAPTURE>(moves, board);
        order_moves(board, moves);

        for (const auto& move : moves) {
            board.makeMove(move);
            float score = quiescence(board, alpha, beta, true, base_nn_score, qply + 1);
            board.unmakeMove(move);
            if (score <= alpha) return alpha;
            if (score < beta) beta = score;
        }
        return beta;
    }
}

/**
 * Core Alpha-Beta Minimax Search.
 * Includes Null Move Pruning, Transposition Tables, and Leaf Node CNN querying.
 */
float minimax(chess::Board& board, int depth, float alpha, float beta, bool maximizing_player, bool allow_null = true) {
    // Per-node wall-clock check. chrono::now() is cheap enough on this engine
    // (~12k nodes/sec, so ~1 ms/sec overhead) that we get precise time
    // bounding without sacrificing depth. The throttled 4096-stride version
    // only fired a handful of times across an entire search at our depth,
    // which let main overshoot the budget by seconds without realising it.
    node_count++;
    auto current_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count();
    if (elapsed >= time_limit_ms) {
        out_of_time = true;
        return 0.0f;
    }

    // --- Cheap draw detection (no movegen required) ---
    if (board.isHalfMoveDraw() || board.isInsufficientMaterial() || board.isRepetition()) {
        return 0.0f;
    }

    uint64_t board_hash = board.zobrist();
    int tt_index = board_hash % TT_SIZE;

    // Check Transposition Table Memory
    if (TT[tt_index].hash == board_hash && TT[tt_index].depth >= depth) {
        const TTEntry& entry = TT[tt_index];
        if (entry.flag == EXACT) return entry.score;
        else if (entry.flag == LOWERBOUND && entry.score > alpha) alpha = entry.score;
        else if (entry.flag == UPPERBOUND && entry.score < beta) beta = entry.score;
        if (alpha >= beta) return entry.score;
    }

    // --- LEAF NODE: The Brain stays ON ---
    // Instead of raw material, use the CNN's deep positional evaluation at the bottom of the tree.
    if (depth == 0) {
        // Detect terminal positions at the leaf (mate / stalemate). anylegalmoves
        // is cheaper than full legalmoves because it bails on the first legal move.
        if (!chess::movegen::anylegalmoves(board)) {
            if (board.inCheck()) {
                return board.sideToMove() == chess::Color::WHITE ? -10000.0f + depth : 10000.0f - depth;
            }
            return 0.0f; // stalemate
        }
        // NN value cache lookup. The CNN forward pass dominates runtime; this
        // memoisation typically eliminates a large fraction of leaf NN calls.
        int nn_idx = static_cast<int>(board_hash % NN_CACHE_SIZE);
        float nn_score;
        if (NN_CACHE[nn_idx].hash == board_hash) {
            nn_score = NN_CACHE[nn_idx].value;
        } else {
            std::vector<float> empty_policy;
            get_nn_outputs(board, empty_policy, nn_score);
            NN_CACHE[nn_idx] = {board_hash, nn_score};
        }
        return quiescence(board, alpha, beta, maximizing_player, nn_score);
    }

    // Null Move Pruning (Threat Detection)
    if (depth >= 3 && !board.inCheck() && allow_null) {
        board.makeNullMove();
        float null_score = minimax(board, depth - 3, alpha, beta, !maximizing_player, false);
        board.unmakeNullMove();
        if (maximizing_player && null_score >= beta) return beta;
        if (!maximizing_player && null_score <= alpha) return alpha;
    }

    // Single move generation: detect terminal (mate/stalemate) directly from the
    // resulting movelist instead of calling isGameOver() which would re-run
    // anylegalmoves internally and double the most expensive non-NN op.
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (moves.empty()) {
        if (board.inCheck()) {
            return board.sideToMove() == chess::Color::WHITE ? -10000.0f + depth : 10000.0f - depth;
        }
        return 0.0f; // stalemate
    }
    order_moves(board, moves); // Fast classical sort deep in the tree

    float original_alpha = alpha;
    float original_beta = beta;
    chess::Move best_move_found = chess::Move::NULL_MOVE;

    if (maximizing_player) {
        float max_eval = -10000.0f;
        for (const auto& move : moves) {
            board.makeMove(move);
            float eval_score = minimax(board, depth - 1, alpha, beta, false, true);
            board.unmakeMove(move);

            if (out_of_time) return 0.0f; // Bail without poisoning the TT.

            // Remember the best move found for the PV (Principal Variation)
            if (eval_score > max_eval) {
                max_eval = eval_score;
                best_move_found = move;
            }

            alpha = std::max(alpha, eval_score);
            if (beta <= alpha) break;
        }

        // Save to TT (depth-preferred replacement: keep deeper entries for the
        // same position to avoid re-searching during iterative deepening).
        int flag = EXACT;
        if (max_eval <= original_alpha) flag = UPPERBOUND;
        else if (max_eval >= beta) flag = LOWERBOUND;
        if (TT[tt_index].hash != board_hash || TT[tt_index].depth <= depth) {
            TT[tt_index] = {board_hash, max_eval, depth, flag, best_move_found};
        }

        return max_eval;
    } else {
        float min_eval = 10000.0f;
        for (const auto& move : moves) {
            board.makeMove(move);
            float eval_score = minimax(board, depth - 1, alpha, beta, true, true);
            board.unmakeMove(move);

            if (out_of_time) return 0.0f; // Bail without poisoning the TT.

            if (eval_score < min_eval) {
                min_eval = eval_score;
                best_move_found = move;
            }

            beta = std::min(beta, eval_score);
            if (beta <= alpha) break;
        }

        // Save to TT (depth-preferred replacement).
        int flag = EXACT;
        if (min_eval >= original_beta) flag = LOWERBOUND;
        else if (min_eval <= alpha) flag = UPPERBOUND;
        if (TT[tt_index].hash != board_hash || TT[tt_index].depth <= depth) {
            TT[tt_index] = {board_hash, min_eval, depth, flag, best_move_found};
        }

        return min_eval;
    }
}

/**
 * Reconstructs the engine's expected line of play (Idea/PV) by reading
 * the memory chain stored in the Transposition Table.
 */
std::string get_principal_variation(chess::Board board, int depth) {
    std::string pv = "";
    for (int i = 0; i < depth; i++) {
        uint64_t hash = board.zobrist();
        TTEntry entry = TT[hash % TT_SIZE];

        if (entry.hash == hash && entry.best_move != chess::Move::NULL_MOVE) {
            pv += chess::uci::moveToUci(entry.best_move) + " ";
            board.makeMove(entry.best_move); // Play it on dummy board to look deeper
        } else {
            break;
        }
    }
    return pv;
}

int main(int argc, char* argv[]) {
    if (argc < 2) return 1;
    std::string fen = argv[1];
    float target_elo = (argc >= 3) ? std::stof(argv[2]) : 2500.0f;
    int SEARCH_DEPTH = (argc >= 4) ? std::stoi(argv[3]) : 4;
    int total_time_ms = (argc >= 5) ? std::stoi(argv[4]) : 10000;

    // Reserve a small slice of the time budget for the supplementary user-reply
    // analysis (a TT-warm shallow search on the position resulting from the
    // bot's chosen move). The budget is intentionally small — the supp search
    // is just for ranking the user's top replies for Level-2 tutor hints, not
    // for finding the absolute best move. Capped at 500 ms or 1/16 of the
    // total budget, whichever is smaller, so the main search keeps almost
    // all the time for going deeper.
    int supp_time_ms = std::min(500, total_time_ms / 16);
    time_limit_ms = total_time_ms - supp_time_ms;

    at::set_num_threads(1); // Force CPU computation
    torch::NoGradGuard no_grad; // Disable gradient tracking for speed

    try {
        module = torch::jit::load("assets/traced_alphazero.pt");
    } catch (const c10::Error& e) {
        std::cerr << "BEST_MOVE:ERROR_LOADING_MODEL\n";
        return -1;
    }

    elo_tensor = torch::tensor({{target_elo / 3000.0f}}, torch::kFloat32);
    // SHAPE: 1 Batch, 13 Channels, 8x8 Board
    global_board_tensor = torch::zeros({1, 13, 8, 8}, torch::kFloat32);
    global_board_data = global_board_tensor.data_ptr<float>();
    global_inputs.push_back(global_board_tensor);
    global_inputs.push_back(elo_tensor);

    chess::Board board(fen);
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);

    // --- THE ROOT POLICY TRICK ---
    // Ask the AI to evaluate all legal moves with its raw intuition *before* search starts.
    std::vector<float> root_policy(4096, 0.0f);
    float root_value = 0.0f;
    get_nn_outputs(board, root_policy, root_value);

    std::vector<std::pair<float, chess::Move>> policy_scored_moves;
    for (const auto& move : moves) {
        int src = move.from().index();
        int dst = move.to().index();
        float intuition_score = root_policy[src * 64 + dst];
        policy_scored_moves.push_back({intuition_score, move});
    }

    // Sort moves so the most intuitive ones are searched first
    std::sort(policy_scored_moves.begin(), policy_scored_moves.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });

    // --- DYNAMIC POLICY PRUNING (Beam Search logic) ---
    // By permanently deleting unintuitive moves at the root, the engine calculates exponentially deeper.
    //
    // SAFETY NETS (critical for tactical/mate awareness):
    //   1. If we are in check, do NOT prune. The set of legal moves is already
    //      tiny, and the NN policy can easily mis-rank the only saving move.
    //   2. Forcing moves (captures, promotions, checks) are ALWAYS kept,
    //      regardless of policy rank, because they are the only moves that can
    //      deliver mate or refute an opponent's tactic. Pruning these out is
    //      what previously caused the engine to "miss" obvious checkmates when
    //      the enemy king was fully exposed but the mating move was quiet
    //      (e.g. Qa8#) and ranked low by the NN policy.
    moves.clear();

    int active_pieces = count_active_pieces(board);
    int moves_to_keep;

    if (board.inCheck()) {
        // Keep everything: we cannot afford to drop the only legal escape.
        moves_to_keep = (int)policy_scored_moves.size();
    } else if (active_pieces > 14) {
        // Complex Midgame: Wider safety net (Top 50%, min 8 moves) to avoid missing tactical defenses.
        moves_to_keep = std::max(8, (int)(policy_scored_moves.size() * 0.50));
    } else {
        // Endgame: Far fewer tactical variations. Prune aggressively (Top 25%, min 3 moves).
        // This accelerates depth calculation to easily solve endgames.
        moves_to_keep = std::max(3, (int)(policy_scored_moves.size() * 0.25));
    }

    moves_to_keep = std::min(moves_to_keep, (int)policy_scored_moves.size()); // Prevent overflow

    std::vector<bool> kept(policy_scored_moves.size(), false);
    int added = 0;
    for (size_t i = 0; i < policy_scored_moves.size(); i++) {
        if (is_forcing_move(board, policy_scored_moves[i].second)) {
            moves.add(policy_scored_moves[i].second);
            kept[i] = true;
            added++;
        }
    }
    for (size_t i = 0; i < policy_scored_moves.size() && added < moves_to_keep; i++) {
        if (kept[i]) continue;
        moves.add(policy_scored_moves[i].second);
        added++;
    }

    // --- ITERATIVE DEEPENING SEARCH ---
    chess::Move best_move_overall = moves[0];
    out_of_time = false;
    node_count = 0;
    start_time = std::chrono::high_resolution_clock::now();
    bool is_white = board.sideToMove() == chess::Color::WHITE;
    int depth_reached = 0;      // Last ID iteration that fully completed.
    uint64_t main_nodes = 0;    // Nodes searched by the main search alone.

    for (int current_depth = 1; current_depth <= SEARCH_DEPTH; current_depth++) {
        std::vector<std::pair<float, chess::Move>> current_scores;
        bool depth_completed = true;

        for (const auto& move : moves) {
            board.makeMove(move);
            float move_val = minimax(board, current_depth - 1, -10000.0f, 10000.0f, !is_white, true);
            board.unmakeMove(move);

            if (out_of_time) {
                depth_completed = false;
                break; // Time limit hit, abort this depth branch
            }
            current_scores.push_back({move_val, move});
        }

        // CRITICAL: If time ran out mid-depth, discard the partial calculations.
        if (!depth_completed) break;

        // If the depth fully finished, update the official move order
        std::sort(current_scores.begin(), current_scores.end(), [is_white](const auto& a, const auto& b) {
            return is_white ? (a.first > b.first) : (a.first < b.first);
        });

        moves.clear();
        for (const auto& scored_move : current_scores) {
            moves.add(scored_move.second);
        }
        best_move_overall = moves[0];
        depth_reached = current_depth;

        // Per-iteration timing to stderr for visibility (api.py captures stderr
        // but only parses stdout, so this is harmless to the protocol).
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
        std::cerr << "[engine] depth " << current_depth
                  << " complete: " << elapsed << " ms, "
                  << node_count << " nodes, best="
                  << chess::uci::moveToUci(best_move_overall) << "\n";

        // --- Mate short-circuit ---
        // Once a forced mate is on the board there's no point spending more
        // time refining the search; the result will not change. Score >= 9000
        // (or <= -9000) corresponds to mate within the search horizon.
        if (!current_scores.empty()) {
            float top = current_scores[0].first;
            if (top >= 9000.0f || top <= -9000.0f) break;
        }
    }
    main_nodes = node_count;

    // --- TUTOR MODE OUTPUT ---
    // Export the finalized calculation data formatted specifically for the Python API to parse.
    std::cout << "BEST_MOVE:" << chess::uci::moveToUci(best_move_overall) << "\n";
    // Achieved search depth (last fully completed ID iteration). This is the
    // real measure of search strength; the IDEA/PV chain length below can be
    // shorter due to TT collisions or quiescence leaves, so don't confuse
    // PV-string length with search depth.
    std::cout << "DEPTH_REACHED:" << depth_reached << "\n";
    std::cout << "TUTOR_DATA:\n";

    // Print the top 3 best moves with their predicted lines
    int display_count = std::min(3, (int)moves.size());
    for (int i = 0; i < display_count; i++) {
        chess::Move m = moves[i];
        chess::Board dummy_board = board;
        dummy_board.makeMove(m);

        // Append the memory-extracted Principal Variation
        std::string idea = chess::uci::moveToUci(m) + " " + get_principal_variation(dummy_board, 4);

        std::cout << "RANK:" << i+1
                  << "|MOVE:" << chess::uci::moveToUci(m)
                  << "|IDEA:" << idea << "\n";
    }

    // --- USER REPLY ANALYSIS (TT-warm) ---
    // After the bot has chosen `best_move_overall`, run a quick iterative-
    // deepening search on the resulting position to surface the human's top
    // replies. The transposition table is already populated for many of these
    // positions from the deep main search, so this typically completes far
    // under the reserved supplementary budget.
    chess::Board user_board = board;
    user_board.makeMove(best_move_overall);
    chess::Movelist user_moves;
    chess::movegen::legalmoves(user_moves, user_board);

    std::vector<std::pair<float, chess::Move>> user_scored;
    int supp_depth_reached = 0;
    auto supp_start_wall = std::chrono::high_resolution_clock::now();
    if (!user_moves.empty()) {
        out_of_time = false;
        start_time = std::chrono::high_resolution_clock::now();
        supp_start_wall = start_time;
        time_limit_ms = supp_time_ms;
        node_count = 0;
        bool user_is_white = (user_board.sideToMove() == chess::Color::WHITE);

        // Iterative deepening so even if we run out of supp budget mid-depth
        // we keep the most recent fully-completed ranking.
        for (int d = 1; d <= 4 && !out_of_time; d++) {
            std::vector<std::pair<float, chess::Move>> sc;
            sc.reserve(user_moves.size());
            bool ok = true;
            for (const auto& m : user_moves) {
                user_board.makeMove(m);
                float v = minimax(user_board, d - 1, -10000.0f, 10000.0f, !user_is_white, true);
                user_board.unmakeMove(m);
                if (out_of_time) { ok = false; break; }
                sc.push_back({v, m});
            }
            if (!ok) break;
            std::sort(sc.begin(), sc.end(), [user_is_white](const auto& a, const auto& b) {
                return user_is_white ? (a.first > b.first) : (a.first < b.first);
            });
            user_scored = std::move(sc);
            supp_depth_reached = d;
        }
    }
    auto supp_end_wall = std::chrono::high_resolution_clock::now();
    auto supp_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(supp_end_wall - supp_start_wall).count();
    std::cerr << "[engine] summary: main depth=" << depth_reached
              << " main_nodes=" << main_nodes
              << " supp depth=" << supp_depth_reached
              << " supp_ms=" << supp_elapsed << "\n";

    std::cout << "USER_TUTOR_DATA:\n";
    int u_display = std::min(3, (int)user_scored.size());
    for (int i = 0; i < u_display; i++) {
        chess::Move m = user_scored[i].second;
        chess::Board dummy_user = user_board;
        dummy_user.makeMove(m);
        std::string idea = chess::uci::moveToUci(m) + " " + get_principal_variation(dummy_user, 4);
        std::cout << "USER_RANK:" << i+1
                  << "|MOVE:" << chess::uci::moveToUci(m)
                  << "|IDEA:" << idea << "\n";
    }
    return 0;
}