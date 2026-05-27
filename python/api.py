"""
AI Chess Tutor - FastAPI Backend
This script serves the web UI and acts as a middleman between the frontend and the C++ AlphaZero engine.
It handles opening book lookups, formats engine data, and generates human-readable tactical annotations.
"""

import os

# Add LibTorch shared libraries to runtime path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
LIBTORCH_LIB = os.path.join(PROJECT_ROOT, "libtorch", "lib")

# Prepend to LD_LIBRARY_PATH so the engine subprocess can find .so files
os.environ["LD_LIBRARY_PATH"] = LIBTORCH_LIB + ":" + os.environ.get("LD_LIBRARY_PATH", "")


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import subprocess
import chess
import chess.polyglot
import time

app = FastAPI(title="Hybrid AlphaZero Engine API")

# Allow the frontend to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Request Models ---
class MoveRequest(BaseModel):
    fen: str
    elo: int = 2500
    depth: int = 4
    time_limit: int = 10000 # Time limit in milliseconds

# --- Path Resolution ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
HTML_PATH = os.path.join(PROJECT_ROOT, "web", "index.html")
BOOK_PATH = os.path.join(PROJECT_ROOT, "assets", "book.bin")
ENGINE_PATH = os.path.join(PROJECT_ROOT, "build", "engine")

@app.get("/")
def serve_ui():
    """Serves the frontend interactive chessboard UI."""
    if os.path.exists(HTML_PATH):
        return FileResponse(HTML_PATH)
    raise HTTPException(status_code=404, detail="Frontend HTML not found in web/ folder.")

@app.post("/get_move")
def calculate_move(request: MoveRequest):
    """
    Main route to calculate the bot's move or generate Tutor hints.
    It checks the opening book first, and if no book move is found, falls back to the C++ engine.
    """
    board = chess.Board(request.fen)

    # 1. Opening Book Lookup (Zero calculation cost)
    try:
        if os.path.exists(BOOK_PATH):
            with chess.polyglot.open_reader(BOOK_PATH) as reader:
                start_time = time.time()
                book_move = reader.choice(board).move
                end_time = time.time()
                return {
                    "move": book_move.uci(),
                    "engine": "PolyGlot Book",
                    "depth": 0,
                    "time": round(end_time - start_time, 3)
                }
    except IndexError:
        pass # No book move found for this position, proceed to engine

    if not os.path.exists(ENGINE_PATH):
        raise HTTPException(status_code=500, detail="C++ Engine executable not found!")

    try:
        # 2. Call the C++ Engine Subprocess
        start_time = time.time()
        result = subprocess.run(
            [ENGINE_PATH, request.fen, str(request.elo), str(request.depth), str(request.time_limit)],
            capture_output=True,
            text=True,
            check=True
        )
        end_time = time.time()
        calculation_time = round(end_time - start_time, 2)

        best_move = None
        top_moves = []

        # 3. Parse the C++ stdout stream
        # The engine returns a specific format: BEST_MOVE:e2e4 and RANK:1|MOVE:e2e4|IDEA:e2e4 c7c5...
        for line in result.stdout.split('\n'):
            line = line.strip()
            if line.startswith("BEST_MOVE:"):
                best_move = line.split(":")[1].strip()
            elif line.startswith("RANK:"):
                parts = line.split("|")
                if len(parts) == 3:
                    rank = parts[0].split(":")[1]
                    move = parts[1].split(":")[1]
                    idea = parts[2].split(":")[1]

                    top_moves.append({
                        "rank": int(rank),
                        "move": move,
                        "idea": idea
                    })

        # --- 4. TACTICAL MOTIF ANNOTATOR (Python Logic Layer) ---
        # Instead of calculating forks/skewers in C++, Python analyzes the moves
        # returned by C++ to generate human-readable explanations for the UI.
        piece_names = {1: "Pawn", 2: "Knight", 3: "Bishop", 4: "Rook", 5: "Queen", 6: "King"}

        for tutor in top_moves:
            try:
                m = chess.Move.from_uci(tutor["move"])
                tags = []

                # Motif A: Capture & Trade Detection
                if board.is_capture(m):
                    victim = board.piece_at(m.to_square)
                    attacker = board.piece_at(m.from_square)

                    if victim and attacker:
                        # Simulate the move to see if the opponent can recapture immediately
                        board.push(m)
                        is_recaptured = board.is_attacked_by(board.turn, m.to_square) # Board turn is now opponent's
                        board.pop()

                        v_type = victim.piece_type
                        a_type = attacker.piece_type
                        v_name = piece_names.get(v_type, "piece")
                        a_name = piece_names.get(a_type, "piece")

                        if is_recaptured:
                            if v_type == a_type:
                                tags.append(f"🔄 Trades {v_name}s")
                            elif v_type < a_type:
                                tags.append(f"⚠️ Sacrifices {a_name} for {v_name}")
                            else:
                                tags.append(f"💥 Wins {v_name} for {a_name}")
                        else:
                            tags.append(f"💥 Captures {v_name}")
                    elif board.is_en_passant(m):
                        tags.append("💥 En Passant Capture")

                # Motif B: Check Detection
                board.push(m) # Temporarily apply the move
                if board.is_checkmate():
                    tags.append("🏆 Checkmate!")
                elif board.is_check():
                    tags.append("🎯 Delivers Check")

                # Motif C: Fork / Double Attack Detection (Basic Heuristic)
                # Count how many valuable pieces the moved piece is currently attacking
                attacked_squares = board.attacks(m.to_square)
                valuable_targets = 0
                for sq in attacked_squares:
                    target_piece = board.piece_at(sq)
                    if target_piece and target_piece.color == board.turn and target_piece.piece_type > 1:
                        valuable_targets += 1

                if valuable_targets >= 2:
                    tags.append("🔱 Creates a Fork/Double Attack!")

                board.pop() # Revert simulation

                # Save the human-readable tags back to the dictionary
                if tags:
                    tutor["tactics"] = " | ".join(tags)
                else:
                    tutor["tactics"] = "🛡️ Solid positional move"

            except ValueError:
                tutor["tactics"] = "Positional move"

        if best_move:
            return {
                "move": best_move,
                "engine": "Hybrid AlphaZero C++",
                "depth": request.depth,
                "time": calculation_time,
                "top_3": top_moves
            }

        raise HTTPException(status_code=500, detail="Engine failed to return a move format.")
    
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"C++ Engine crashed.\nSTDERR: {e.stderr}\nSTDOUT: {e.stdout}\nEngine path: {ENGINE_PATH}\nExists: {os.path.exists(ENGINE_PATH)}")


class ThreatRequest(BaseModel):
    fen: str
    elo: int = 2500

@app.post("/get_threats")
def check_threats(request: ThreatRequest):
    """
    Level 1 Assistance: Analyzes the board for immediate tactical threats.
    Works by playing a 'Null Move' (passing the turn) to see what the opponent would do
    if given two moves in a row.
    """
    board = chess.Board(request.fen)

    if board.is_check():
        return {"threat": "You are in check! You must defend your King!", "hint": None, "is_threat": True}

    # 1. Quick hint generation (Shallow search on real board)
    hint_move_uci = None
    try:
        hint_res = subprocess.run([ENGINE_PATH, request.fen, str(request.elo), "3", "1000"], capture_output=True, text=True)
        for line in hint_res.stdout.split('\n'):
            if line.startswith("BEST_MOVE:"):
                hint_move_uci = line.split(":")[1].strip()
    except:
        pass

    # 2. Pass the turn to the opponent to expose their threats
    try:
        board.push(chess.Move.null())
    except AssertionError:
        return {"threat": "Could not analyze threats.", "hint": None, "is_threat": False}

    null_fen = board.fen()

    try:
        # Give the engine 1.5 seconds to find the opponent's master plan
        threat_res = subprocess.run([ENGINE_PATH, null_fen, str(request.elo), "4", "1500"], capture_output=True, text=True)

        threat_move_uci = None
        for line in threat_res.stdout.split('\n'):
            if line.startswith("BEST_MOVE:"):
                threat_move_uci = line.split(":")[1].strip()
                break

        if threat_move_uci and threat_move_uci != "0000":
            m = chess.Move.from_uci(threat_move_uci)
            piece_names = {1: "Pawn", 2: "Knight", 3: "Bishop", 4: "Rook", 5: "Queen", 6: "King"}

            is_threat = False
            threat_msg = ""

            # --- STEP A: Immediate Capture (What dies right now?) ---
            immediate_victim = board.piece_at(m.to_square) if board.is_capture(m) else None
            victim_name = piece_names.get(immediate_victim.piece_type, "piece") if immediate_victim else None
            is_defended = board.is_attacked_by(board.turn, m.to_square) if immediate_victim else True

            # --- STEP B: Secondary Threats (What dies next?) ---
            # Simulate the opponent's threat move to see what new pieces they attack
            board.push(m)
            secondary_targets = []

            attacked_squares = board.attacks(m.to_square)
            for sq in attacked_squares:
                target = board.piece_at(sq)
                # If they are aiming at our valuable piece (Knight or better)
                if target and target.color != board.turn and target.piece_type >= chess.KNIGHT:
                    # ONLY warn if our piece is currently undefended
                    if not board.is_attacked_by(target.color, sq):
                        secondary_targets.append(piece_names.get(target.piece_type, "piece"))

            board.pop() # Revert the simulated threat move

            # --- STEP C: Build the Smart Warning Message ---
            if immediate_victim and not is_defended:
                threat_msg = f"⚠️ WATCH OUT! Your {victim_name} on {chess.square_name(m.to_square)} is completely undefended! "
                is_threat = True
            elif immediate_victim and immediate_victim.piece_type >= chess.KNIGHT:
                threat_msg = f"⚠️ WATCH OUT! Opponent threatens to capture your {victim_name} with {threat_move_uci}! "
                is_threat = True

            # Append secondary sequence threats
            if len(secondary_targets) > 0:
                sec_str = " and ".join(list(set(secondary_targets))) # Remove duplicates
                if is_threat:
                    threat_msg += f"<br>If they play this, they will <strong>ALSO</strong> threaten your undefended {sec_str}!"
                else:
                    threat_msg = f"⚠️ DANGER! If opponent plays {threat_move_uci}, it will immediately attack your undefended {sec_str}!"
                    is_threat = True

            # --- STEP D: Immediate Forks ---
            board.push(chess.Move.null())
            board.push(m)
            valuable_targets_fork = sum(1 for sq in board.attacks(m.to_square) if board.piece_at(sq) and board.piece_at(sq).color != board.turn and board.piece_at(sq).piece_type >= chess.KNIGHT)
            board.pop()
            board.pop()

            if valuable_targets_fork >= 2 and not is_threat:
                threat_msg = f"🔱 DANGER! Opponent threatens {threat_move_uci}, creating a brutal fork/double attack!"
                is_threat = True

            if not is_threat:
                threat_msg = "Position seems stable. No immediate tactical threats detected."

            return {"threat": threat_msg, "hint": hint_move_uci, "is_threat": is_threat}

        return {"threat": "Position seems stable.", "hint": hint_move_uci, "is_threat": False}

    except subprocess.CalledProcessError:
        return {"threat": "Could not analyze threats.", "hint": None, "is_threat": False}