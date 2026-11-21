"""

Usage:
    python semantic_analysis.py

Dependencies:
    pip install chess stockfish matplotlib
"""

import chess
import chess.pgn
import chess.engine
import json
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
STOCKFISH_PATH    = "/opt/homebrew/bin/stockfish"
ANALYSIS_TIME     = 0.05
EVAL_CAP          = 1000

BLUNDER_THRESHOLD    = 200
MISTAKE_THRESHOLD    = 100
INACCURACY_THRESHOLD = 50

# Material values — Shannon (1950), standard centipawn weights
PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 300,
    chess.BISHOP: 300,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   0,
}

PIECE_NAMES = {
    chess.PAWN:   "Pawn",
    chess.KNIGHT: "Knight",
    chess.BISHOP: "Bishop",
    chess.ROOK:   "Rook",
    chess.QUEEN:  "Queen",
    chess.KING:   "King",
}

COLOR_MAP = {
    "Blunder":    "#e74c3c",
    "Mistake":    "#e67e22",
    "Inaccuracy": "#f1c40f",
    "Good":       "#2ecc71",
}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def classify_move(delta_cp):
    if delta_cp >= BLUNDER_THRESHOLD:    return "Blunder"
    if delta_cp >= MISTAKE_THRESHOLD:    return "Mistake"
    if delta_cp >= INACCURACY_THRESHOLD: return "Inaccuracy"
    return "Good"

def get_game_phase(move_number, total_moves):
    if move_number <= 10:                    return "Opening"
    if move_number <= total_moves * 0.65:    return "Middlegame"
    return "Endgame"

def material_count(board, color):
    return sum(PIECE_VALUES[pt] * len(board.pieces(pt, color))
               for pt in PIECE_VALUES)

def sq(square):
    return chess.square_name(square)


# ─────────────────────────────────────────────
# RULE-BASED SEMANTIC ENGINE
# ─────────────────────────────────────────────
def rule_based_explanation(board_before, move, color, best_move=None, delta_cp=0):
    """
    Attempt to explain a bad move using a priority-ordered set of chess rules.
    Returns a plain-English string. Never returns None — always falls back
    to a generic description so no move is left unexplained.

    Rules checked (priority order):
      1.  Missed checkmate
      2.  Delivered checkmate (own move was actually good — flag mislabel)
      3.  Walked into checkmate
      4.  Check delivered
      5.  Unfavorable capture (loses material)
      6.  Hanging piece — moved away from defence
      7.  Piece left en prise after move
      8.  Fork created against the player
      9.  Pin created against the player
      10. Discovered attack against the player
      11. Back-rank weakness exploited
      12. King safety — castling rights lost unnecessarily
      13. Generic material drop fallback
      14. Positional fallback
    """
    board_after = board_before.copy()
    board_after.push(move)
    mover    = "White" if color == chess.WHITE else "Black"
    opponent = not color
    move_san = board_before.san(move)

    # ── Rule 1: Missed checkmate ─────────────────────────────────────
    # Check if the best move (from engine) delivers mate while this move doesn't
    if best_move and best_move != move:
        test = board_before.copy()
        test.push(best_move)
        if test.is_checkmate():
            return (f"{mover} missed a checkmate with "
                    f"{board_before.san(best_move)}. "
                    f"Instead {move_san} was played, letting the opponent escape.")

    # ── Rule 2: Move delivers checkmate (shouldn't be flagged bad) ───
    if board_after.is_checkmate():
        return f"{mover} delivers checkmate with {move_san}."

    # ── Rule 3: Walked into checkmate ────────────────────────────────
    if board_after.is_check():
        # Count legal moves for the side in check
        if board_after.is_checkmate():
            return f"{mover} plays {move_san}, but this results in checkmate."

    # ── Rule 4: Delivered check ──────────────────────────────────────
    if board_after.is_check():
        return f"{mover} plays {move_san}, putting the opponent's King in check."

    # ── Rule 5: Unfavorable capture ──────────────────────────────────
    if board_before.is_capture(move):
        captured_pt = board_before.piece_type_at(move.to_square)
        moving_pt   = board_before.piece_type_at(move.from_square)
        if captured_pt and moving_pt:
            cap_val  = PIECE_VALUES[captured_pt]
            move_val = PIECE_VALUES[moving_pt]
            cap_name  = PIECE_NAMES[captured_pt]
            move_name = PIECE_NAMES[moving_pt]
            # Is the landing square defended by opponent?
            if board_after.is_attacked_by(opponent, move.to_square):
                if move_val > cap_val:
                    loss = move_val - cap_val
                    return (f"{mover} captures a {cap_name} ({cap_val}cp) "
                            f"with a {move_name} ({move_val}cp), "
                            f"but the {move_name} is immediately recaptured — "
                            f"losing {loss}cp of material.")
            else:
                return (f"{mover} captures the opponent's {cap_name} "
                        f"on {sq(move.to_square)} with {move_san}.")

    # ── Rule 6: Moving piece was defending another piece ─────────────
    from_sq = move.from_square
    to_sq   = move.to_square
    # Find pieces that were defended by the moving piece before the move
    moving_pt = board_before.piece_type_at(from_sq)
    if moving_pt:
        previously_defended = []
        for sq_idx in chess.SQUARES:
            piece = board_before.piece_at(sq_idx)
            if (piece and piece.color == color and sq_idx != from_sq
                    and from_sq in board_before.attackers(color, sq_idx)):
                # Was this square still defended after the move?
                if not board_after.is_attacked_by(color, sq_idx):
                    pt = piece.piece_type
                    if pt >= chess.ROOK:  # Only flag if valuable piece left undefended
                        previously_defended.append((sq_idx, pt))

        if previously_defended:
            sq_idx, pt = previously_defended[0]
            return (f"{mover} moves {PIECE_NAMES[moving_pt]} away from "
                    f"{sq(from_sq)}, leaving the {PIECE_NAMES[pt]} on "
                    f"{sq(sq_idx)} undefended and vulnerable to capture.")

    # ── Rule 7: Moved piece is now en prise ──────────────────────────
    landed_pt = board_after.piece_type_at(to_sq)
    if landed_pt and landed_pt != chess.KING:
        landed_val = PIECE_VALUES[landed_pt]
        if (board_after.is_attacked_by(opponent, to_sq)
                and not board_after.is_attacked_by(color, to_sq)
                and landed_val >= PIECE_VALUES[chess.KNIGHT]):
            return (f"{mover} moves {PIECE_NAMES[landed_pt]} to {sq(to_sq)}, "
                    f"leaving it completely undefended — "
                    f"the opponent can capture it for free.")

    # ── Rule 8: Opponent fork after this move ────────────────────────
    # Check if opponent now has a move that attacks 2+ valuable pieces
    for opp_move in board_after.legal_moves:
        test = board_after.copy()
        test.push(opp_move)
        attacked_valuable = []
        for sq_idx in chess.SQUARES:
            piece = test.piece_at(sq_idx)
            if (piece and piece.color == color
                    and piece.piece_type >= chess.KNIGHT
                    and test.is_attacked_by(opponent, sq_idx)):
                attacked_valuable.append(PIECE_NAMES[piece.piece_type])
        if len(attacked_valuable) >= 2:
            fork_move_san = board_after.san(opp_move)
            targets = " and ".join(set(attacked_valuable[:2]))
            return (f"After {move_san}, the opponent can play {fork_move_san} "
                    f"forking {mover}'s {targets} simultaneously.")

    # ── Rule 9: Pin created against the player ───────────────────────
    for opp_move in board_after.legal_moves:
        test = board_after.copy()
        opp_pt = board_after.piece_type_at(opp_move.from_square)
        if opp_pt in (chess.BISHOP, chess.ROOK, chess.QUEEN):
            test.push(opp_move)
            # Look for pinned pieces (pieces that can't move without exposing king)
            for sq_idx in chess.SQUARES:
                piece = test.piece_at(sq_idx)
                if piece and piece.color == color and piece.piece_type != chess.KING:
                    # Try moving the piece — if it exposes king, it's pinned
                    for friendly_move in list(test.legal_moves):
                        if friendly_move.from_square == sq_idx:
                            break
                    else:
                        if (test.is_attacked_by(opponent, sq_idx)
                                and piece.piece_type >= chess.ROOK):
                            pin_san = board_after.san(opp_move)
                            return (f"After {move_san}, the opponent can pin "
                                    f"{mover}'s {PIECE_NAMES[piece.piece_type]} "
                                    f"on {sq(sq_idx)} with {pin_san}.")

    # ── Rule 10: Back-rank threat ─────────────────────────────────────
    back_rank = chess.BB_RANK_1 if color == chess.WHITE else chess.BB_RANK_8
    king_sq   = board_after.king(color)
    if king_sq and chess.BB_SQUARES[king_sq] & back_rank:
        # Count escape squares for the king
        king_moves = [m for m in board_after.legal_moves
                      if m.from_square == king_sq]
        if len(king_moves) == 0:
            rooks_queens = (board_after.pieces(chess.ROOK,   opponent)
                          | board_after.pieces(chess.QUEEN,  opponent))
            if rooks_queens:
                return (f"{mover}'s King is trapped on the back rank after "
                        f"{move_san} — vulnerable to a back-rank checkmate.")

    # ── Rule 11: King safety — moved toward center in opening ────────
    if moving_pt == chess.KING and get_game_phase(
            board_before.fullmove_number, 40) != "Endgame":
        return (f"{mover} moves the King with {move_san} during the middlegame, "
                f"exposing it to attacks and losing castling safety.")

    # ── Rule 12: Generic material drop ───────────────────────────────
    mat_before = material_count(board_before, color)
    mat_after  = material_count(board_after,  color)
    lost = mat_before - mat_after
    if lost >= 300:
        return (f"{mover} loses approximately {lost}cp of material "
                f"as a result of {move_san}.")

    # ── Rule 13: Positional fallback ─────────────────────────────────
    drop_pawns = round(delta_cp / 100, 1)
    return (f"{mover} plays {move_san}, causing a positional deterioration "
            f"of {drop_pawns} pawns. The evaluation drops significantly "
            f"without an obvious material reason — likely a positional error.")


# ─────────────────────────────────────────────
# MAIN ANALYSIS PIPELINE
# ─────────────────────────────────────────────
def analyze_pgn(pgn_string):
    pgn_io = io.StringIO(pgn_string)
    game   = chess.pgn.read_game(pgn_io)
    if game is None:
        raise ValueError("Could not parse PGN.")

    board       = game.board()
    move_nodes  = list(game.mainline())
    total_moves = len(move_nodes)
    results     = []

    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:

        info          = engine.analyse(board, chess.engine.Limit(time=ANALYSIS_TIME))
        prev_score_cp = clamp(
            info["score"].white().score(mate_score=10000), -EVAL_CAP, EVAL_CAP)

        for i, node in enumerate(move_nodes):
            move        = node.move
            move_san    = board.san(move)
            color       = board.turn
            color_name  = "White" if color == chess.WHITE else "Black"
            move_number = board.fullmove_number

            board_before = board.copy()
            board.push(move)

            info          = engine.analyse(board, chess.engine.Limit(time=ANALYSIS_TIME))
            curr_score_cp = clamp(
                info["score"].white().score(mate_score=10000), -EVAL_CAP, EVAL_CAP)

            delta = (prev_score_cp - curr_score_cp
                     if color == chess.WHITE
                     else curr_score_cp - prev_score_cp)

            classification = classify_move(delta)
            phase          = get_game_phase(move_number, total_moves // 2)

            # ── Semantic layer ──────────────────────────────────
            explanation = None
            if classification != "Good":
                # Get engine's best move for this position (for missed-mate detection)
                best_info = engine.analyse(
                    board_before,
                    chess.engine.Limit(time=ANALYSIS_TIME),
                    multipv=1
                )
                best_move = (best_info.get("pv", [None])[0]
                             if isinstance(best_info, dict)
                             else None)

                explanation = rule_based_explanation(
                    board_before, move, color, best_move, delta
                )

            results.append({
                "move_index"    : i + 1,
                "move_number"   : move_number,
                "color"         : color_name,
                "san"           : move_san,
                "eval_cp"       : curr_score_cp,
                "eval_pawns"    : round(curr_score_cp / 100, 2),
                "delta_cp"      : round(delta, 1),
                "classification": classification,
                "phase"         : phase,
                "explanation"   : explanation,
            })

            prev_score_cp = curr_score_cp

    # ── Summary ─────────────────────────────────────────────────────
    counts      = {"Blunder": 0, "Mistake": 0, "Inaccuracy": 0, "Good": 0}
    white_stats = {k: 0 for k in counts}
    black_stats = {k: 0 for k in counts}

    for m in results:
        counts[m["classification"]] += 1
        if m["color"] == "White":
            white_stats[m["classification"]] += 1
        else:
            black_stats[m["classification"]] += 1

    return {
        "moves"  : results,
        "summary": {
            "total_moves"   : total_moves,
            "overall"       : counts,
            "white"         : white_stats,
            "black"         : black_stats,
            "final_eval_cp" : results[-1]["eval_cp"] if results else 0,
        }
    }


# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────
def plot_evaluation_curve(analysis, output_path):
    moves = analysis["moves"]
    xs    = [m["move_index"] for m in moves]
    ys    = [m["eval_pawns"] for m in moves]

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#1e1e2e")
    ax.axhspan(0,  10, alpha=0.04, color="white")
    ax.axhspan(-10, 0, alpha=0.04, color="black")
    ax.axhline(0, color="#555577", linewidth=1.0, linestyle="--")

    for i in range(1, len(moves)):
        if moves[i]["phase"] != moves[i-1]["phase"]:
            ax.axvline(moves[i]["move_index"], color="#8888aa",
                       linewidth=1, linestyle=":")
            ax.text(moves[i]["move_index"] + 0.3, 8.5,
                    moves[i]["phase"], color="#aaaacc", fontsize=8)

    ax.plot(xs, ys, color="#7777ff", linewidth=1.5, zorder=2)

    for m in moves:
        if m["classification"] != "Good":
            ax.scatter(m["move_index"], m["eval_pawns"],
                       color=COLOR_MAP[m["classification"]], s=60, zorder=3)
            if m["explanation"]:
                short = (m["explanation"][:45] + "…"
                         if len(m["explanation"]) > 45
                         else m["explanation"])
                ax.annotate(short,
                            xy=(m["move_index"], m["eval_pawns"]),
                            xytext=(5, 10), textcoords="offset points",
                            fontsize=5, color="#dddddd",
                            arrowprops=dict(arrowstyle="-",
                                            color="#555577", lw=0.5))

    ax.set_xlim(0, len(moves) + 1)
    ax.set_ylim(-11, 11)
    ax.set_xlabel("Move index", color="#cccccc")
    ax.set_ylabel("Evaluation (pawns)", color="#cccccc")
    ax.set_title("Post-Game Semantic Analysis — Evaluation Curve ",
                 color="white", fontsize=13)
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")

    legend_items = [mpatches.Patch(color=v, label=k)
                    for k, v in COLOR_MAP.items()]
    ax.legend(handles=legend_items, loc="upper right",
              facecolor="#2a2a3e", edgecolor="#555577", labelcolor="white")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Plot saved → {output_path}")


def print_summary(analysis):
    s = analysis["summary"]
    print("\n═══════════════════════════════════════════════")
    print("  POST-GAME SEMANTIC ANALYSIS — SUMMARY")
    print("═══════════════════════════════════════════════")
    print(f"  Total half-moves : {s['total_moves']}")
    print(f"  Final evaluation : {s['final_eval_cp']/100:+.2f} pawns")
    print()
    print(f"  {'Category':<14} {'Total':>6} {'White':>8} {'Black':>8}")
    print(f"  {'─'*40}")
    for cat in ["Blunder", "Mistake", "Inaccuracy", "Good"]:
        print(f"  {cat:<14} {s['overall'][cat]:>6} "
              f"{s['white'][cat]:>8} {s['black'][cat]:>8}")
    print()
    print("  Flagged moves with explanations:")
    print(f"  {'─'*60}")
    for m in analysis["moves"]:
        if m["classification"] != "Good" and m["explanation"]:
            print(f"  Move {m['move_index']:>3} ({m['color']:>5}) "
                  f"{m['san']:<8} {m['classification']:<12}  "
                  f"{m['explanation']}")
    print("═══════════════════════════════════════════════\n")


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────
SAMPLE_PGN = """
[Event "Demo Game"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. b4 Bxb4 5. c3 Ba5
6. d4 exd4 7. O-O d3 8. Qb3 Qf6 9. e5 Qg6 10. Re1 Nge7
11. Ba3 b5 12. Qxb5 Rb8 13. Qa4 Bb6 14. Nbd2 Bb7 15. Ne4 Qf5
16. Bxd3 Qh5 17. Nf6+ gxf6 18. exf6 Rg8 19. Rad1 Qxf3
20. Rxe7+ Nxe7 21. Qxd7+ Kxd7 22. Bf5+ Ke8 23. Bd7+ Kf8 24. Bxe7# 1-0
"""

if __name__ == "__main__":
    print("[*] Running semantic analysis (rule-based)...")
    analysis = analyze_pgn(SAMPLE_PGN)

    with open("semantic_analysis_output.json", "w") as f:
        json.dump(analysis, f, indent=2)
    print("[✓] JSON saved → semantic_analysis_output.json")

    print_summary(analysis)
    plot_evaluation_curve(analysis, "eval_curve.png")