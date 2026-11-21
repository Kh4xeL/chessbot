"""

Usage:
    python semantic_analysis.py

Dependencies:
    pip install chess stockfish matplotlib
"""

import atexit
import contextlib
import threading
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
def _find_stockfish():
    """STOCKFISH_PATH env var, then PATH, then tools/ (Windows download), then Homebrew."""
    import glob, os, shutil
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [os.environ.get("STOCKFISH_PATH"), shutil.which("stockfish")]
    candidates += sorted(glob.glob(os.path.join(root, "tools", "**", "stockfish*.exe"), recursive=True))
    candidates += ["/opt/homebrew/bin/stockfish", "/usr/local/bin/stockfish", "/usr/games/stockfish"]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return "stockfish"


STOCKFISH_PATH    = _find_stockfish()

_SF = None
_SF_LOCK = threading.Lock()


@contextlib.contextmanager
def shared_stockfish():
    """One long-lived Stockfish shared by all requests; starting a process per request
    cost more than the 0.05 s analyses themselves. Restarted if it dies."""
    global _SF
    with _SF_LOCK:
        if _SF is None:
            _SF = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        try:
            yield _SF
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError):
            try:
                _SF.quit()
            except Exception:
                pass
            _SF = None
            raise


@atexit.register
def _close_stockfish():
    if _SF is not None:
        try:
            _SF.quit()
        except Exception:
            pass
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
    """
    board_after = board_before.copy()
    board_after.push(move)
    mover    = "White" if color == chess.WHITE else "Black"
    opponent = not color
    move_san = board_before.san(move)

    # ── Rule 1: Missed checkmate ─────────────────────────────────────
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

    # ── Rule 3: Allowed Checkmate (Blundered Mate in 1) ──────────────
    for opp_move in board_after.legal_moves:
        test = board_after.copy()
        test.push(opp_move)
        if test.is_checkmate():
            return (f"After {move_san}, the opponent can play {board_after.san(opp_move)} "
                    f"and deliver checkmate.")

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
    moving_pt = board_before.piece_type_at(from_sq)
    if moving_pt:
        previously_defended = []
        for sq_idx in chess.SQUARES:
            piece = board_before.piece_at(sq_idx)
            if (piece and piece.color == color and sq_idx != from_sq
                    and from_sq in board_before.attackers(color, sq_idx)):
                if not board_after.is_attacked_by(color, sq_idx):
                    if board_after.is_attacked_by(opponent, sq_idx):
                        pt = piece.piece_type
                        previously_defended.append((sq_idx, pt))

        if previously_defended:
            previously_defended.sort(key=lambda x: PIECE_VALUES[x[1]], reverse=True)
            sq_idx, pt = previously_defended[0]
            return (f"{mover} moves {PIECE_NAMES[moving_pt]} away from "
                    f"{sq(from_sq)}, leaving the {PIECE_NAMES[pt]} on "
                    f"{sq(sq_idx)} undefended and vulnerable to capture.")

    # ── Rule 7: Moved piece is now en prise ──────────────────────────
    landed_pt = board_after.piece_type_at(to_sq)
    if landed_pt and landed_pt != chess.KING:
        landed_val = PIECE_VALUES[landed_pt]
        attackers = list(board_after.attackers(opponent, to_sq))
        if attackers:
            min_attacker_val = min(PIECE_VALUES[board_after.piece_type_at(a)] for a in attackers)
            is_defended = board_after.is_attacked_by(color, to_sq)
            
            if not is_defended:
                return (f"{mover} moves {PIECE_NAMES[landed_pt]} to {sq(to_sq)}, "
                        f"leaving it completely undefended — "
                        f"the opponent can capture it for free.")
            elif min_attacker_val < landed_val:
                best_attacker_pt = min([board_after.piece_type_at(a) for a in attackers], key=lambda x: PIECE_VALUES[x])
                return (f"{mover} moves {PIECE_NAMES[landed_pt]} to {sq(to_sq)}, "
                        f"where it can be captured by a lesser piece "
                        f"({PIECE_NAMES[best_attacker_pt]}).")

    # ── Rule 8: Opponent fork after this move ────────────────────────
    for opp_move in board_after.legal_moves:
        test = board_after.copy()
        test.push(opp_move)
        attacker_sq = opp_move.to_square
        attacker_pt = test.piece_type_at(attacker_sq)
        if not attacker_pt: continue
        
        attacked_valuable = []
        for sq_idx in chess.SQUARES:
            piece = test.piece_at(sq_idx)
            if piece and piece.color == color and piece.piece_type >= chess.KNIGHT:
                if attacker_sq in test.attackers(opponent, sq_idx):
                    is_defended = test.is_attacked_by(color, sq_idx)
                    favorable = PIECE_VALUES[attacker_pt] < PIECE_VALUES[piece.piece_type]
                    if piece.piece_type == chess.KING or not is_defended or favorable:
                        attacked_valuable.append(PIECE_NAMES[piece.piece_type])
        
        if len(attacked_valuable) >= 2:
            fork_move_san = board_after.san(opp_move)
            targets = list(set(attacked_valuable))
            if len(targets) == 1:
                target_str = f"two {targets[0]}s"
            else:
                target_str = " and ".join(targets[:2])
            return (f"After {move_san}, the opponent can play {fork_move_san} "
                    f"forking {mover}'s {target_str} simultaneously.")

    # ── Rule 9: Pin created against the player ───────────────────────
    for opp_move in board_after.legal_moves:
        test = board_after.copy()
        opp_pt = board_after.piece_type_at(opp_move.from_square)
        if opp_pt in (chess.BISHOP, chess.ROOK, chess.QUEEN):
            test.push(opp_move)
            for sq_idx in chess.SQUARES:
                piece = test.piece_at(sq_idx)
                if piece and piece.color == color and piece.piece_type != chess.KING:
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

    # ── Rule 10: King safety — moved toward center in opening ────────
    if moving_pt == chess.KING and not board_before.is_castling(move) and get_game_phase(
            board_before.fullmove_number, 40) != "Endgame":
        return (f"{mover} moves the King with {move_san} during the middlegame, "
                f"exposing it to attacks and losing castling safety.")

    # ── Rule 11: Generic material drop ───────────────────────────────
    mat_before = material_count(board_before, color)
    mat_after  = material_count(board_after,  color)
    lost = mat_before - mat_after
    if lost >= 300:
        return (f"{mover} loses approximately {lost}cp of material "
                f"as a result of {move_san}.")

    # ── Rule 12: Positional fallback ─────────────────────────────────
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

    with shared_stockfish() as engine:

        # Analyze the starting position to get the very first prev_score_cp
        start_info = engine.analyse(board, chess.engine.Limit(time=ANALYSIS_TIME))
        prev_score_cp = clamp(start_info["score"].white().score(mate_score=10000), -EVAL_CAP, EVAL_CAP)

        for i, node in enumerate(move_nodes):
            move        = node.move
            move_san    = board.san(move)
            color       = board.turn
            color_name  = "White" if color == chess.WHITE else "Black"
            move_number = board.fullmove_number

            board_before = board.copy()
            board.push(move)

            # Evaluate board AFTER the move
            info_after = engine.analyse(board, chess.engine.Limit(time=ANALYSIS_TIME))
            curr_score_cp = clamp(
                info_after["score"].white().score(mate_score=10000), -EVAL_CAP, EVAL_CAP)

            # Evaluate board BEFORE the move with multipv=2 to find Perfect/Brilliant and missed mates
            info_before = engine.analyse(
                board_before,
                chess.engine.Limit(time=ANALYSIS_TIME),
                multipv=2
            )
            if isinstance(info_before, dict):
                info_before = [info_before]
                
            best_move = info_before[0].get("pv", [None])[0] if "pv" in info_before[0] else None
            best_move_uci = best_move.uci() if best_move else None
            
            # Recalculate prev_score_cp precisely from the multipv analysis to ensure accurate delta
            prev_score_cp = clamp(
                info_before[0]["score"].white().score(mate_score=10000), -EVAL_CAP, EVAL_CAP)

            delta = (prev_score_cp - curr_score_cp
                     if color == chess.WHITE
                     else curr_score_cp - prev_score_cp)

            classification = classify_move(delta)
            
            # ── Advanced Move Classifications ────────────────────
            pv_sequence = []
            if classification == "Good":
                # Check for Perfect (huge gap between best and 2nd best)
                is_perfect = False
                perfect_explanation = None
                if len(info_before) > 1 and best_move == move:
                    score1 = info_before[0]["score"].white().score(mate_score=10000)
                    score2 = info_before[1]["score"].white().score(mate_score=10000)
                    
                    # Convert to side-to-move perspective
                    stm_score1 = score1 if color == chess.WHITE else -score1
                    stm_score2 = score2 if color == chess.WHITE else -score2
                    gap = stm_score1 - stm_score2
                    
                    if gap > 250 and not board_before.is_capture(move):
                        is_perfect = True
                        if stm_score2 < 0 and stm_score1 > 0:
                            perfect_explanation = "The only winning move!"
                        elif stm_score2 < -100 and stm_score1 >= -50:
                            perfect_explanation = "The only way to save the situation."
                        else:
                            perfect_explanation = "A brilliantly precise move, far better than alternatives."

                # Check for Brilliant (sacrificed material but eval is good)
                is_brilliant = False
                mat_before = material_count(board_before, color)
                mat_after  = material_count(board, color)
                if mat_before - mat_after >= 300: # Sacrificed a minor piece or more
                    is_brilliant = True

                if is_brilliant:
                    classification = "Brilliant"
                elif is_perfect:
                    classification = "Perfect"
                    
                if classification in ["Brilliant", "Perfect"]:
                    if "pv" in info_before[0]:
                        pv_sequence = [m.uci() for m in info_before[0]["pv"][:4]]

            phase = get_game_phase(move_number, total_moves // 2)

            # ── Semantic layer ──────────────────────────────────
            explanation = None
            threat_move_uci = None
            if board.is_checkmate():
                classification = "Perfect"
                explanation = "Checkmate!"
            elif classification == "Perfect" and perfect_explanation:
                explanation = perfect_explanation
            elif classification not in ["Good", "Brilliant", "Perfect"]:
                explanation = rule_based_explanation(
                    board_before, move, color, best_move, delta
                )
                if classification in ["Blunder", "Mistake"]:
                    threat_move = info_after.get("pv", [None])[0]
                    threat_move_uci = threat_move.uci() if threat_move else None

            results.append({
                "move_index"    : i + 1,
                "move_number"   : move_number,
                "color"         : color_name,
                "san"           : move_san,
                "from_sq"       : chess.square_name(move.from_square),
                "to_sq"         : chess.square_name(move.to_square),
                "fen_after"     : board.fen(),
                "eval_cp"       : curr_score_cp,
                "eval_pawns"    : round(curr_score_cp / 100, 2),
                "delta_cp"      : round(delta, 1),
                "classification": classification,
                "phase"         : phase,
                "explanation"   : explanation,
                "best_move"     : best_move_uci,
                "threat_move"   : threat_move_uci,
                "pv_sequence"   : pv_sequence,
                "is_checkmate"  : board.is_checkmate()
            })

            prev_score_cp = curr_score_cp

    # ── Summary ─────────────────────────────────────────────────────
    counts      = {"Brilliant": 0, "Perfect": 0, "Blunder": 0, "Mistake": 0, "Inaccuracy": 0, "Good": 0}
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