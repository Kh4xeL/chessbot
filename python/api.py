"""
AI Chess Tutor - FastAPI Backend
This script serves the web UI and acts as a middleman between the frontend and the C++ AlphaZero engine.
It handles opening book lookups, formats engine data, and generates human-readable tactical annotations.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import atexit
import queue
import subprocess
import threading
import chess
import chess.polyglot
import os
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
    elo: int = 2500         # >= 2900 means full strength; lower Elo caps depth and adds human-like choice
    depth: int = 99         # maximum search depth (the time limit usually stops it first)
    time_limit: int = 3000  # Time limit in milliseconds
    start_fen: Optional[str] = None  # game start position; with `moves` it lets the engine see repetitions
    moves: List[str] = []            # UCI moves played from start_fen up to `fen`
    use_book: bool = True            # tutor hints turn this off so they always get engine analysis


# The opening book is master-level; below this Elo the bot plays its own (weaker) moves.
BOOK_MIN_ELO = 1800

# --- Path Resolution ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
HTML_PATH = os.path.join(PROJECT_ROOT, "web", "index.html")
WEB_DIR = os.path.join(PROJECT_ROOT, "web")
BOOK_PATH = os.path.join(PROJECT_ROOT, "assets", "book.bin")

# Serve static files from the web directory
app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")


def _find_engine():
    """The engine can be built by CLion (cmake-build-release) or by hand (build/)."""
    names = ["engine.exe", "engine"] if os.name == "nt" else ["engine"]
    for d in ["cmake-build-release", "build", "cmake-build-relwithdebinfo", "cmake-build-debug"]:
        for n in names:
            p = os.path.join(PROJECT_ROOT, d, n)
            if os.path.isfile(p):
                return p
    return os.path.join(PROJECT_ROOT, "build", names[0])


ENGINE_PATH = _find_engine()
FULL_STRENGTH = 3000


class EngineProcess:
    """Keeps one engine running in --server mode. Starting a process per request
    cost more than the search itself, and the hash table now survives between
    moves. Requests are serialised with a lock (the engine is single-threaded)."""

    def __init__(self, path):
        self.path = path
        self.proc = None
        self.lock = threading.Lock()

    def _start(self):
        self.proc = subprocess.Popen(
            [self.path, "--server"], cwd=PROJECT_ROOT, text=True, bufsize=1,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None)
        line = self.proc.stdout.readline().strip()
        if line != "READY":
            self.proc.kill()
            self.proc = None
            raise RuntimeError(f"engine failed to start: {line}")

    def _go_locked(self, fen, elo, depth, time_ms):
        if self.proc is None or self.proc.poll() is not None:
            self._start()
        self.proc.stdin.write(f"go {int(elo)} {int(depth)} {int(time_ms)} {fen}\n")
        self.proc.stdin.flush()
        lines = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine exited unexpectedly")
            line = line.rstrip("\r\n")
            if line == "END":
                return "\n".join(lines)
            lines.append(line)

    def go(self, fen, elo, depth, time_ms):
        """Returns the engine's output block (BEST_MOVE:..., RANK:..., ...)."""
        with self.lock:
            try:
                return self._go_locked(fen, elo, depth, time_ms)
            except (RuntimeError, OSError):
                self.close()
                return self._go_locked(fen, elo, depth, time_ms)  # one retry with a fresh process

    def newgame(self):
        """Clears the engine's hash table between games."""
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                return
            self.proc.stdin.write("newgame\n")
            self.proc.stdin.flush()
            self.proc.stdout.readline()  # "OK"

    def close(self):
        if self.proc is not None:
            try:
                self.proc.kill()
            except OSError:
                pass
            self.proc = None


class EnginePool:
    """A few engine processes so a tutor hint or threat check doesn't wait behind the
    bot's 3 s move. Each request takes a free process; processes start on first use."""

    def __init__(self, path, size):
        self.engines = [EngineProcess(path) for _ in range(size)]
        self.free = queue.Queue()
        for e in self.engines:
            self.free.put(e)

    def go(self, fen, elo, depth, time_ms):
        e = self.free.get()
        try:
            return e.go(fen, elo, depth, time_ms)
        finally:
            self.free.put(e)

    def newgame(self):
        for e in self.engines:
            e.newgame()

    def close(self):
        for e in self.engines:
            e.close()


ENGINE = EnginePool(ENGINE_PATH, size=int(os.environ.get("ENGINE_PROCESSES", "3")))
atexit.register(ENGINE.close)


def position_string(fen, start_fen=None, moves=None):
    """FEN alone, or start FEN + moves so the engine knows the game history (repetitions)."""
    if start_fen and moves:
        return f"{start_fen} moves {' '.join(moves)}"
    return fen


def engine_best_move(fen, elo, depth, time_ms):
    for line in ENGINE.go(fen, elo, depth, time_ms).split("\n"):
        if line.startswith("BEST_MOVE:"):
            return line.split(":", 1)[1].strip()
    return None

PIECE_NAMES = {1: "Pawn", 2: "Knight", 3: "Bishop", 4: "Rook", 5: "Queen", 6: "King"}


def annotate_tutor_moves(board: chess.Board, tutor_moves: list) -> None:
    """Mutates each entry in `tutor_moves` to add a `tactics` field with
    emoji tags describing what the move does on `board` (capture quality,
    check/mate, fork). Used to render UI hints for both the bot's top moves
    and the user's top replies. The board is treated as read-only (we push
    + pop within each entry)."""
    for tutor in tutor_moves:
        try:
            m = chess.Move.from_uci(tutor["move"])
            tags = []

            # A mating move needs no other label (its "sacrifice" can't be recaptured).
            board.push(m)
            mates = board.is_checkmate()
            board.pop()
            if mates:
                tutor["tactics"] = "🏆 Checkmate!"
                continue

            # Motif A: Capture & Trade Detection
            if board.is_capture(m):
                victim = board.piece_at(m.to_square)
                attacker = board.piece_at(m.from_square)

                if victim and attacker:
                    board.push(m)
                    # Only a legal recapture counts (a pinned piece or the king walking
                    # into a defended square can't take back).
                    is_recaptured = any(r.to_square == m.to_square for r in board.legal_moves)
                    board.pop()

                    v_type = victim.piece_type
                    a_type = attacker.piece_type
                    v_name = PIECE_NAMES.get(v_type, "piece")
                    a_name = PIECE_NAMES.get(a_type, "piece")

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

            # Motif B: Check / Mate
            board.push(m)
            if board.is_check():
                tags.append("🎯 Delivers Check")

            # Motif C: Fork / Double Attack
            attacked_squares = board.attacks(m.to_square)
            valuable_targets = 0
            for sq in attacked_squares:
                target_piece = board.piece_at(sq)
                if target_piece and target_piece.color == board.turn and target_piece.piece_type > 1:
                    valuable_targets += 1
            if valuable_targets >= 2:
                tags.append("🔱 Creates a Fork/Double Attack!")

            board.pop()

            tutor["tactics"] = " | ".join(tags) if tags else "🛡️ Solid positional move"
        except ValueError:
            tutor["tactics"] = "Positional move"


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

    # 1. Opening Book Lookup (Zero calculation cost). Skipped for tutor hints (they need the
    # engine's top moves) and for weak bot settings (book moves are master-level).
    try:
        if request.use_book and request.elo >= BOOK_MIN_ELO and os.path.exists(BOOK_PATH):
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
        # 2. Ask the persistent C++ engine
        start_time = time.time()
        position = position_string(request.fen, request.start_fen, request.moves)
        engine_output = ENGINE.go(position, request.elo, request.depth, request.time_limit)
        end_time = time.time()
        calculation_time = round(end_time - start_time, 2)

        best_move = None
        eval_score = 0.0
        mate_in = None
        depth_reached = 0
        top_moves = []
        user_top_moves = []

        # 3. Parse the C++ stdout stream
        # The engine returns: BEST_MOVE:e2e4, DEPTH_REACHED:N (last fully
        # completed iterative-deepening iteration), RANK:1|MOVE:e2e4|IDEA:...,
        # and USER_RANK:1|MOVE:c7c5|IDEA:... (top user replies on the position
        # AFTER the bot plays best_move; pre-computed in the same engine call
        # using a TT-warm supplementary search).
        for line in engine_output.split('\n'):
            line = line.strip()
            if line.startswith("BEST_MOVE:"):
                best_move = line.split(":")[1].strip()
            elif line.startswith("EVAL:"):
                try:
                    eval_score = float(line.split(":")[1].strip())
                except ValueError:
                    eval_score = 0.0
            elif line.startswith("MATE:"):
                try:
                    mate_in = int(line.split(":")[1].strip())
                except ValueError:
                    mate_in = None
            elif line.startswith("DEPTH_REACHED:"):
                try:
                    depth_reached = int(line.split(":")[1].strip())
                except ValueError:
                    depth_reached = 0
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
            elif line.startswith("USER_RANK:"):
                parts = line.split("|")
                if len(parts) == 3:
                    rank = parts[0].split(":")[1]
                    move = parts[1].split(":")[1]
                    idea = parts[2].split(":")[1]
                    user_top_moves.append({
                        "rank": int(rank),
                        "move": move,
                        "idea": idea
                    })

        # --- 4. TACTICAL MOTIF ANNOTATOR (Python Logic Layer) ---
        # Annotates each tutor entry with emoji tags (capture/check/fork/etc.)
        # for the position it is being suggested from. Used for both the bot's
        # top_3 (annotated against the request FEN) and the user's top_3 replies
        # (annotated against the position AFTER the bot plays best_move).
        annotate_tutor_moves(board, top_moves)

        # Build the post-bot-move position and annotate the user's top replies
        # against it. The supplementary search ran on this exact position, so
        # the moves listed here are legal in it.
        user_top_3 = []
        if best_move and user_top_moves:
            try:
                user_board = board.copy()
                user_board.push(chess.Move.from_uci(best_move))
                annotate_tutor_moves(user_board, user_top_moves)
                user_top_3 = user_top_moves
            except (ValueError, AssertionError):
                user_top_3 = []

        if best_move:
            return {
                "move": best_move,
                "engine": "Hybrid NNUE C++",
                "depth": request.depth,
                "depth_reached": depth_reached,
                "eval_pawns": eval_score,
                "mate": mate_in,  # mate in N moves (+ White mates, - Black mates), else None
                "time": calculation_time,
                "top_3": top_moves,
                "user_top_3": user_top_3,
            }

        raise HTTPException(status_code=500, detail="Engine failed to return a move format.")

    except (RuntimeError, OSError) as e:
        raise HTTPException(status_code=500, detail=f"C++ Engine error: {e}")


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

    # 1. Quick hint generation (full-strength search on the real board)
    hint_move_uci = None
    try:
        hint_move_uci = engine_best_move(request.fen, FULL_STRENGTH, 99, 600)
    except (RuntimeError, OSError):
        pass

    # 2. Pass the turn to the opponent to expose their threats
    try:
        board.push(chess.Move.null())
    except AssertionError:
        return {"threat": "Could not analyze threats.", "hint": None, "is_threat": False}

    null_fen = board.fen()

    try:
        # Let the engine find the opponent's most dangerous move if they could move twice
        threat_move_uci = engine_best_move(null_fen, FULL_STRENGTH, 99, 800)

        if threat_move_uci and threat_move_uci != "0000":
            m = chess.Move.from_uci(threat_move_uci)
            piece_names = {1: "Pawn", 2: "Knight", 3: "Bishop", 4: "Rook", 5: "Queen", 6: "King"}

            is_threat = False
            threat_msg = ""

            # --- STEP 0: Mate threat (checked first: it outranks any material threat) ---
            threat_san = board.san(m)
            board.push(m)
            mates = board.is_checkmate()
            board.pop()
            if mates:
                return {"threat": f"☠️ MATE THREAT! If you ignore it, opponent plays {threat_san} and it's checkmate!",
                        "hint": hint_move_uci, "is_threat": True, "threat_move": threat_move_uci}

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

            return {"threat": threat_msg, "hint": hint_move_uci, "is_threat": is_threat, "threat_move": threat_move_uci}

        return {"threat": "Position seems stable.", "hint": hint_move_uci, "is_threat": False, "threat_move": None}

    except (RuntimeError, OSError):
        return {"threat": "Could not analyze threats.", "hint": None, "is_threat": False, "threat_move": None}


@app.post("/new_game")
def new_game():
    """Clears the engine's hash table so a new game starts from a clean slate."""
    try:
        ENGINE.newgame()
    except (RuntimeError, OSError):
        pass
    return {"ok": True}


from .semantic_analysis import analyze_pgn as _analyze_pgn
from pydantic import BaseModel
from typing import List, Optional

class AnalyzeRequest(BaseModel):
    pgn: str

@app.post("/analyze")
def analyze_game(request: AnalyzeRequest):
    try:
        pgn_data = request.pgn.strip()
        # If it looks like a bare FEN string, wrap it in a PGN structure
        if pgn_data.count('/') >= 7 and " " in pgn_data and not pgn_data.startswith("["):
            pgn_data = f'[FEN "{pgn_data}"]\n\n'
            
        return _analyze_pgn(pgn_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

class ExploreRequest(BaseModel):
    fen: str
    move: str

@app.post("/explore_move")
def explore_move(request: ExploreRequest):
    import chess.engine
    from .semantic_analysis import shared_stockfish, clamp, EVAL_CAP, classify_move
    
    try:
        board = chess.Board(request.fen)
        color = board.turn
        
        with shared_stockfish() as engine:
            # Eval before
            info_before = engine.analyse(board, chess.engine.Limit(time=0.05))
            prev_score = clamp(info_before["score"].white().score(mate_score=10000), -EVAL_CAP, EVAL_CAP)
            
            # Push custom move
            m = chess.Move.from_uci(request.move)
            board.push(m)
                
            # Eval after
            info_after = engine.analyse(board, chess.engine.Limit(time=0.05))
            curr_score = clamp(info_after["score"].white().score(mate_score=10000), -EVAL_CAP, EVAL_CAP)
            
            delta = (prev_score - curr_score) if color == chess.WHITE else (curr_score - prev_score)
            classification = classify_move(delta)
            if board.is_checkmate():
                classification = "Perfect"
            
            best_reply = info_after.get("pv", [None])[0]
            best_reply_uci = best_reply.uci() if best_reply else None
            
            return {
                "eval_pawns": round(curr_score / 100, 2),
                "classification": classification,
                "best_reply": best_reply_uci,
                "fen_after": board.fen(),
                "san": chess.Board(request.fen).san(m),
                "is_checkmate": board.is_checkmate()
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))