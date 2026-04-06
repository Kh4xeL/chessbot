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

// Classical Tactical Move Ordering (Speeds up Alpha-Beta Pruning)
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
    if (move.promotionType() != chess::PieceType::NONE) score += 90;
    return score;
}

void order_moves(const chess::Board& board, chess::Movelist& moves) {
    std::vector<std::pair<int, chess::Move>> scored_moves;
    for (int i = 0; i < moves.size(); i++) {
        scored_moves.push_back({score_move(board, moves[i]), moves[i]});
    }
    std::sort(scored_moves.begin(), scored_moves.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });
    moves.clear();
    for (const auto& sm : scored_moves) moves.add(sm.second);
}

/**
 * Quiescence Search: Evaluates noisy positions (like active captures) deeply
 * to ensure the engine doesn't suffer from the Horizon Effect mid-trade.
 */
float quiescence(chess::Board& board, float alpha, float beta, bool maximizing_player, float base_nn_score) {
    // 1 Pawn = 1.0. We multiply the NN score (from -1 to 1) by 4.0.
    // This allows the AI to value positional dominance up to 4 pawns worth of advantage.
    float stand_pat = evaluate_material(board) + (base_nn_score * 4.0f);

    if (maximizing_player) {
        if (stand_pat >= beta) return beta;
        if (alpha < stand_pat) alpha = stand_pat;

        chess::Movelist moves;
        chess::movegen::legalmoves<chess::movegen::MoveGenType::CAPTURE>(moves, board);
        order_moves(board, moves);

        for (const auto& move : moves) {
            board.makeMove(move);
            float score = quiescence(board, alpha, beta, false, base_nn_score);
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
            float score = quiescence(board, alpha, beta, true, base_nn_score);
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
    auto current_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count();

    if (elapsed >= time_limit_ms) {
        out_of_time = true;
        return 0.0f;
    }

    uint64_t board_hash = board.zobrist();
    int tt_index = board_hash % TT_SIZE;

    // Check Transposition Table Memory
    if (TT[tt_index].hash == board_hash && TT[tt_index].depth >= depth) {
        TTEntry entry = TT[tt_index];
        if (entry.flag == EXACT) return entry.score;
        else if (entry.flag == LOWERBOUND && entry.score > alpha) alpha = entry.score;
        else if (entry.flag == UPPERBOUND && entry.score < beta) beta = entry.score;
        if (alpha >= beta) return entry.score;
    }

    auto result = board.isGameOver();
    if (result.second != chess::GameResult::NONE) {
        if (result.second == chess::GameResult::LOSE) {
            if (board.sideToMove() == chess::Color::WHITE) return -10000.0f + depth;
            else return 10000.0f - depth;
        }
        return 0.0f;
    }

    // --- LEAF NODE: The Brain stays ON ---
    // Instead of raw material, use the CNN's deep positional evaluation at the bottom of the tree
    if (depth == 0) {
        float nn_score = 0.0f;
        std::vector<float> empty_policy;
        get_nn_outputs(board, empty_policy, nn_score);
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

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
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

            // Remember the best move found for the PV (Principal Variation)
            if (eval_score > max_eval) {
                max_eval = eval_score;
                best_move_found = move;
            }

            alpha = std::max(alpha, eval_score);
            if (beta <= alpha) break;
        }

        // Save to TT
        int flag = EXACT;
        if (max_eval <= original_alpha) flag = UPPERBOUND;
        else if (max_eval >= beta) flag = LOWERBOUND;
        TT[tt_index] = {board_hash, max_eval, depth, flag, best_move_found};

        return max_eval;
    } else {
        float min_eval = 10000.0f;
        for (const auto& move : moves) {
            board.makeMove(move);
            float eval_score = minimax(board, depth - 1, alpha, beta, true, true);
            board.unmakeMove(move);

            if (eval_score < min_eval) {
                min_eval = eval_score;
                best_move_found = move;
            }

            beta = std::min(beta, eval_score);
            if (beta <= alpha) break;
        }

        // Save to TT
        int flag = EXACT;
        if (min_eval >= original_beta) flag = LOWERBOUND;
        else if (min_eval <= alpha) flag = UPPERBOUND;
        TT[tt_index] = {board_hash, min_eval, depth, flag, best_move_found};

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
    time_limit_ms = (argc >= 5) ? std::stoi(argv[4]) : 10000;

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
    moves.clear();

    int active_pieces = count_active_pieces(board);
    int moves_to_keep;

    if (active_pieces > 14) {
        // Complex Midgame: Wider safety net (Top 50%, min 8 moves) to avoid missing tactical defenses.
        moves_to_keep = std::max(8, (int)(policy_scored_moves.size() * 0.50));
    } else {
        // Endgame: Far fewer tactical variations. Prune aggressively (Top 25%, min 3 moves).
        // This accelerates depth calculation to easily solve endgames.
        moves_to_keep = std::max(3, (int)(policy_scored_moves.size() * 0.25));
    }

    moves_to_keep = std::min(moves_to_keep, (int)policy_scored_moves.size()); // Prevent overflow

    for (int i = 0; i < moves_to_keep; i++) {
        moves.add(policy_scored_moves[i].second);
    }

    // --- ITERATIVE DEEPENING SEARCH ---
    chess::Move best_move_overall = moves[0];
    out_of_time = false;
    start_time = std::chrono::high_resolution_clock::now();
    bool is_white = board.sideToMove() == chess::Color::WHITE;

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
    }

    // --- TUTOR MODE OUTPUT ---
    // Export the finalized calculation data formatted specifically for the Python API to parse.
    std::cout << "BEST_MOVE:" << chess::uci::moveToUci(best_move_overall) << "\n";
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
    return 0;
}