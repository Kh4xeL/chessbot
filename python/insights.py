"""
Tutor insights from the rating-aware policy combined with the engine:

  * level choice:  "Most ~1500 players play Nf3 here (41%). The engine prefers d4."
  * hard to find:  the engine's move is one players at your level rarely play, and it is
                   clearly better than what they usually play;
  * trap:          a natural-looking move (popular at your level) that loses a lot.

Each popular move and the engine's choice are evaluated by a short fixed-node search, from
the player's side, so every insight rests on both the human policy and the engine.
"""

import math

import chess
import chess.engine

POPULAR_MIN = 0.05         # moves below this share at the player's level aren't "what players play"
TRAP_MIN_SHARE = 0.12      # a trap must look natural: at least this share of players play it
TRAP_LOSS = 1.2            # ... and lose at least this many pawns against the best move
HARD_MAX_SHARE = 0.10      # "hard to find": the best move is played by fewer than this share
HARD_GAIN = 0.7            # ... and is this many pawns better than the most popular move
EVAL_CAP = 20.0            # pawns; mates count as +-EVAL_CAP


def parse_humanprobs(block):
    """The engine's humanprobs output -> [(uci, prob)], most likely first."""
    out = []
    for line in block.splitlines():
        if line.startswith("MOVE:"):
            f = dict(part.split(":", 1) for part in line.split("|"))
            out.append((f["MOVE"], float(f["P"])))
    return out


def _pawns(score, color):
    s = score.pov(color)
    if s.is_mate():
        return EVAL_CAP if s.mate() > 0 else -EVAL_CAP
    return max(-EVAL_CAP, min(EVAL_CAP, s.score() / 100))


def _eval_after(engine, board, move, nodes):
    """Eval in pawns for the side to move in `board` after it plays `move`."""
    me = board.turn
    b = board.copy()
    b.push(move)
    if b.is_checkmate():
        return EVAL_CAP
    if b.is_game_over(claim_draw=True):
        return 0.0
    return _pawns(engine.analyse(b, chess.engine.Limit(nodes=nodes))["score"], me)


def _pct(p):
    return f"{p * 100:.0f}%" if p >= 0.01 else "under 1%"


def compute(board, level, probs, engine, nodes=150_000, known=None):
    """Insights for the side to move in `board` (with its move history, for repetitions).
    level: the player's rating; probs: [(uci, prob)] from the engine's humanprobs at that rating;
    engine: a python-chess UCI engine used for the evaluations; known: optional [(uci, pawns for
    the side to move)], best first, from a deeper search already shown to the player (the
    tutor's lines), so the insights agree with them."""
    if board.is_game_over():
        return {"level": level, "best": None, "popular": [], "insights": []}
    me = board.turn
    known = [(u, max(-EVAL_CAP, min(EVAL_CAP, v))) for u, v in (known or [])
             if chess.Move.from_uci(u) in board.legal_moves]
    if known:
        best_move, best_eval = chess.Move.from_uci(known[0][0]), known[0][1]
    else:
        info = engine.analyse(board, chess.engine.Limit(nodes=nodes))
        best_move, best_eval = info["pv"][0], _pawns(info["score"], me)
    known_eval = dict(known)
    share = dict(probs)

    popular = [(u, p) for u, p in probs[:3] if p >= POPULAR_MIN] or probs[:1]
    rows = []
    for u, p in popular:
        mv = chess.Move.from_uci(u)
        if mv == best_move:
            ev = best_eval
        elif u in known_eval:
            ev = known_eval[u]
        else:
            ev = _eval_after(engine, board, mv, nodes)
        rows.append({"move": u, "san": board.san(mv), "share": round(p, 4), "eval": round(ev, 2),
                     "loss": round(max(0.0, best_eval - ev), 2)})
    best = {"move": best_move.uci(), "san": board.san(best_move), "share": round(share.get(best_move.uci(), 0.0), 4),
            "eval": round(best_eval, 2)}

    insights = []
    top = rows[0]
    top_is_trap = top["share"] >= TRAP_MIN_SHARE and top["loss"] >= TRAP_LOSS
    if top["move"] != best["move"]:
        text = (f"Most ~{level} players play {top['san']} here ({_pct(top['share'])}). "
                f"The engine prefers {best['san']}")
        if top_is_trap:
            text += "."                                   # the trap warning below says why
        elif top["loss"] >= 5:
            text += ", which is much better."
        elif top["loss"] >= 0.3:
            text += f", about {top['loss']:.1f} pawns better."
        else:
            text += ", but both are fine."
        insights.append({"type": "level_choice", "text": text, "move": top["move"]})
    else:
        insights.append({"type": "level_choice", "move": top["move"],
                         "text": f"{top['san']} is both the most common choice at ~{level} ({_pct(top['share'])}) "
                                 f"and the engine's."})
    if best["share"] < HARD_MAX_SHARE and best["move"] != top["move"] and top["loss"] >= HARD_GAIN:
        insights.append({"type": "hard_to_find", "move": best["move"],
                         "text": f"Hard to find: only {_pct(best['share'])} of ~{level} players play {best['san']} here, "
                                 f"and it is the strongest move."})
    for r in rows:
        if r["share"] >= TRAP_MIN_SHARE and r["loss"] >= TRAP_LOSS and r["move"] != best["move"]:
            if r["eval"] <= -EVAL_CAP + 0.01:
                verdict = "walks into a forced loss"
            elif r["loss"] >= 5:
                verdict = "loses a lot of material"
            else:
                verdict = f"loses about {r['loss']:.1f} pawns"
            insights.append({"type": "trap", "move": r["move"],
                             "text": f"Careful: {r['san']} looks natural ({_pct(r['share'])} of ~{level} players "
                                     f"play it) but {verdict}."})
    return {"level": level, "best": best, "popular": rows, "insights": insights}
