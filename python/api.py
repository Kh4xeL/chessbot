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
import chess.engine
import chess.polyglot
import numpy as np
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
# 3 processes x 4 search threads (Lazy SMP: ~+160 Elo over 1 thread at 0.2 s/move).
ENGINE_PROCESSES = int(os.environ.get("ENGINE_PROCESSES", "3"))
ENGINE_THREADS = int(os.environ.get("ENGINE_THREADS", "4"))


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
            [self.path, "--server", "--threads", str(ENGINE_THREADS)], cwd=PROJECT_ROOT, text=True, bufsize=1,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None)
        line = self.proc.stdout.readline().strip()
        if line != "READY":
            self.proc.kill()
            self.proc = None
            raise RuntimeError(f"engine failed to start: {line}")

    def _command_locked(self, line):
        if self.proc is None or self.proc.poll() is not None:
            self._start()
        self.proc.stdin.write(line + "\n")
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

    def command(self, line):
        """Sends one server command; returns its output block (the lines before END)."""
        with self.lock:
            try:
                return self._command_locked(line)
            except (RuntimeError, OSError):
                self.close()
                return self._command_locked(line)  # one retry with a fresh process

    def go(self, fen, elo, depth, time_ms):
        """Returns the engine's output block (BEST_MOVE:..., RANK:..., ...)."""
        return self.command(f"go {int(elo)} {int(depth)} {int(time_ms)} {fen}")

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

    def command(self, line):
        e = self.free.get()
        try:
            return e.command(line)
        finally:
            self.free.put(e)

    def go(self, fen, elo, depth, time_ms):
        return self.command(f"go {int(elo)} {int(depth)} {int(time_ms)} {fen}")

    def newgame(self):
        for e in self.engines:
            e.newgame()

    def close(self):
        for e in self.engines:
            e.close()


ENGINE = EnginePool(ENGINE_PATH, size=ENGINE_PROCESSES)
atexit.register(ENGINE.close)


def position_string(fen, start_fen=None, moves=None):
    """FEN alone, or start FEN + moves so the engine knows the game history (repetitions)."""
    if start_fen and moves:
        return f"{start_fen} moves {' '.join(moves)}"
    return fen


def parse_rank_line(line):
    """'RANK:1|MOVE:e2e4|IDEA:e2e4 e7e5|EVAL:0.31|MATE:2' -> dict. EVAL (the line's final
    evaluation, White's view) and MATE are only present when the engine has an exact score."""
    fields = dict(part.split(":", 1) for part in line.split("|") if ":" in part)
    rank_key = "USER_RANK" if "USER_RANK" in fields else "RANK"
    try:
        entry = {"rank": int(fields[rank_key]), "move": fields["MOVE"], "idea": fields.get("IDEA", "")}
    except (KeyError, ValueError):
        return None
    entry["eval_pawns"] = float(fields["EVAL"]) if "EVAL" in fields else None
    entry["mate"] = int(fields["MATE"]) if "MATE" in fields else None
    return entry


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
        tablebase = None
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
            elif line.startswith("TABLEBASE:"):
                try:
                    tablebase = int(line.split(":")[1].strip())
                except ValueError:
                    tablebase = None
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
                entry = parse_rank_line(line)
                if entry:
                    top_moves.append(entry)
            elif line.startswith("USER_RANK:"):
                entry = parse_rank_line(line)
                if entry:
                    user_top_moves.append(entry)

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
                "tablebase": tablebase,  # 1 / 0 / -1: White wins / draw / loses with best play (5 pieces or fewer)
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


class PositionRequest(BaseModel):
    fen: str
    multipv: int = 3
    time_ms: int = 800


@app.post("/analyze_position")
def analyze_position(request: PositionRequest):
    """Live Stockfish analysis of one position for the analysis board: search depth plus the
    top lines (SAN and UCI) with their evaluations, like the engine panel on chess.com."""
    from .semantic_analysis import shared_stockfish
    try:
        board = chess.Board(request.fen)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if board.is_game_over():
        return {"depth": 0, "lines": [], "game_over": True}
    multipv = max(1, min(5, request.multipv))
    time_s = max(0.05, min(5.0, request.time_ms / 1000))
    try:
        with shared_stockfish() as sf:
            infos = sf.analyse(board, chess.engine.Limit(time=time_s), multipv=multipv)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stockfish error: {e}")
    if isinstance(infos, dict):
        infos = [infos]
    lines, depth = [], 0
    for info in infos:
        pv = info.get("pv", [])
        if not pv:
            continue
        depth = max(depth, info.get("depth", 0))
        score = info["score"].white()
        mate = score.mate()
        b = board.copy()
        san = []
        for mv in pv[:12]:
            san.append(b.san(mv))
            b.push(mv)
        lines.append({
            "uci": [m.uci() for m in pv[:12]],
            "san": san,
            "eval_pawns": round(score.score(mate_score=10000) / 100, 2),
            "mate": mate,
        })
    return {"depth": depth, "lines": lines, "game_over": False,
            "nodes": infos[0].get("nodes") if infos else None}


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

# --- Rating estimate ("Evaluate your Elo") ---
from . import adaptive, elo_estimate, insights, rating_store

MIN_RATED_MOVES = 10       # fewer of the player's moves say too little about their rating
COMBINE_GAMES = 10         # the overall estimate uses this many most recent games
_ELO_MODEL = None
_ANALYSIS = None           # UCI engine for move losses (same settings as the calibration)
_ANALYSIS_LOCK = threading.Lock()


def _elo_model():
    global _ELO_MODEL
    if _ELO_MODEL is None:
        _ELO_MODEL = elo_estimate.load_model()
    return _ELO_MODEL


def _move_losses(start_fen, moves, plies):
    """Expected-score loss of the moves at the given 1-based plies."""
    wanted = set(plies)
    board = chess.Board(start_fen)
    losses = {}
    with _ANALYSIS_LOCK:
        eng = _analysis_engine()
        for i, u in enumerate(moves):
            mv = chess.Move.from_uci(u)
            if i + 1 in wanted:
                losses[i + 1] = elo_estimate.move_loss(eng, board, mv)[0]
            board.push(mv)
    return [losses[p] for p in plies]


def _analysis_engine():
    """The UCI engine for fixed-node analysis (move losses, insights); call with _ANALYSIS_LOCK held."""
    global _ANALYSIS
    if _ANALYSIS is None:
        _ANALYSIS = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH, cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL)
        atexit.register(_ANALYSIS.quit)
    return _ANALYSIS


def _overall():
    games = rating_store.recent_games(COMBINE_GAMES)
    history = [{"ts": g["ts"], "rating": g["rating"], "low": g["low"], "high": g["high"],
                "bot_elo": g["bot_elo"], "result": g["result"]} for g in reversed(games)]
    if not games:
        return None, history
    llp = np.sum([g["llp"] for g in games], axis=0)
    counts = np.sum([g["counts"] for g in games], axis=0)
    overall = elo_estimate.estimate_from_summary(llp, counts, _elo_model())
    overall["games"] = len(games)
    return overall, history


class RateGameRequest(BaseModel):
    moves: List[str]                  # UCI moves of the whole game
    start_fen: Optional[str] = None
    side: str = "w"                   # the human's colour: "w" or "b"
    bot_elo: Optional[int] = None
    result: Optional[str] = None      # "1-0", "0-1", "1/2-1/2"
    save: bool = True
    adaptive: bool = False            # the game was against the adaptive opponent: update its Elo


def _adaptive_state():
    """The adaptive opponent's current base Elo and how many games it has adapted over."""
    state = rating_store.get_adaptive()
    if state:
        return {"elo": state[0], "games": state[1]}
    overall, _ = _overall()
    return {"elo": adaptive.initial_elo(overall["rating"] if overall else None), "games": 0}


@app.post("/rate_game")
def rate_game(request: RateGameRequest):
    """Estimates the rating the player's moves in this game resemble, saves it, and returns it
    with the combined estimate over recent games and the history for a trend line. For a
    finished game against the adaptive opponent, it also moves the opponent's Elo."""
    if request.side not in ("w", "b"):
        raise HTTPException(status_code=400, detail="side must be 'w' or 'b'")
    model = _elo_model()
    start = request.start_fen or chess.STARTING_FEN
    cmd = (f"humangame {','.join(map(str, elo_estimate.RATINGS))} {request.side} {start} "
           f"moves {' '.join(request.moves)}")
    block = ENGINE.command(cmd)
    if block.startswith("ERROR"):
        raise HTTPException(status_code=400, detail=block.split("\n")[0][6:])
    plies, logp = elo_estimate.parse_humangame(block)
    keep = [i for i, p in enumerate(plies) if p - 1 >= model["skip_plies"]]
    before = _adaptive_state()        # what the opponent played at, before this game joins the history
    out = {"rated": len(keep) >= MIN_RATED_MOVES}
    est = None
    if out["rated"]:
        plies = [plies[i] for i in keep]
        losses = _move_losses(start, request.moves, plies)
        llp, counts = elo_estimate.summary(logp[keep], losses)
        est = elo_estimate.estimate_from_summary(llp, counts, model)
        out["game"] = est
        if request.save:
            rating_store.add_game(request.side, start, request.moves, request.bot_elo, request.result, est, llp, counts)
    else:
        out["reason"] = f"Too short to rate: at least {MIN_RATED_MOVES} of your moves are needed."

    score = adaptive.score_from_result(request.result, request.side)
    if request.adaptive and request.save and score is not None:
        new_elo = adaptive.update(before["elo"], before["games"], score, est["rating"] if est else None)
        rating_store.set_adaptive(new_elo, before["games"] + 1)
        out["adaptive_change"] = {"from": before["elo"], "to": new_elo}
    out["overall"], out["history"] = _overall()
    out["adaptive"] = _adaptive_state()
    return out


@app.get("/rating_history")
def rating_history():
    overall, history = _overall()
    return {"overall": overall, "history": history, "adaptive": _adaptive_state()}


@app.post("/adaptive/reset")
def adaptive_reset():
    """Forget the adaptive opponent's Elo; the next game starts again from the rating estimate."""
    rating_store.reset_adaptive()
    return {"adaptive": _adaptive_state()}


# --- Tutor insights from the human policy ---
class InsightsRequest(BaseModel):
    fen: str
    level: int = 1500                 # the player's rating (their estimate, or the adaptive opponent's)
    start_fen: Optional[str] = None
    moves: List[str] = []
    known: List[dict] = []            # the tutor's lines: {move, eval_pawns (White's view), mate}, best first


@app.post("/insights")
def get_insights(request: InsightsRequest):
    """What players at this level play here, compared with the engine: the common choice,
    hard-to-find best moves and natural-looking moves that lose."""
    level = max(600, min(2800, int(request.level)))
    try:
        if request.start_fen is not None or request.moves:
            board = chess.Board(request.start_fen or chess.STARTING_FEN)
            for u in request.moves:
                board.push_uci(u)
            if board.fen() != chess.Board(request.fen).fen():
                board = chess.Board(request.fen)
        else:
            board = chess.Board(request.fen)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if board.is_game_over():
        return {"level": level, "best": None, "popular": [], "insights": []}
    block = ENGINE.command(f"humanprobs {level} {board.fen()}")   # the policy sees the position only
    if block.startswith("ERROR"):
        raise HTTPException(status_code=400, detail=block.split("\n")[0][6:])
    probs = insights.parse_humanprobs(block)
    sign = 1 if board.turn == chess.WHITE else -1
    known = []
    for k in request.known:
        mate, ev = k.get("mate"), k.get("eval_pawns")
        if mate is not None:
            known.append((k["move"], sign * (insights.EVAL_CAP if mate > 0 else -insights.EVAL_CAP)))
        elif ev is not None:
            known.append((k["move"], sign * float(ev)))
    with _ANALYSIS_LOCK:
        return insights.compute(board, level, probs, _analysis_engine(), known=known)
