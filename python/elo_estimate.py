"""
Estimates a player's rating from their moves.

Each move gives two pieces of evidence, both as a likelihood over a grid of ratings:
  * the rating-aware policy: how likely a player of each rating was to play that move;
  * the move's expected-score loss by engine eval, binned, against how often players of
    each rating lose that much (a table fitted on held-out Lichess games).
The log-likelihoods are weighted (a, b), summed over the moves (and over several games for
a combined estimate), and turned into a posterior with a flat prior. The weights are fitted
for the smallest error, and a width factor (tau) so the 80% range holds 80% of true
ratings (training/fit_elo_estimator.py, assets/elo_model.json).
"""

import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "assets", "elo_model.json")
RATINGS = list(range(600, 2900, 100))                    # 600 .. 2800
LOSS_BINS = [0.0, 0.01, 0.03, 0.06, 0.10, 0.20, 0.35]    # lower edges of expected-score-loss bins


def expected_score(cp):
    """Expected score for the side with an eval of `cp` centipawns (Lichess's win-chance curve)."""
    return 1.0 / (1.0 + np.exp(-0.00368208 * np.asarray(cp, dtype=np.float64)))


def loss_bin(loss):
    return np.clip(np.searchsorted(LOSS_BINS, np.asarray(loss), side="right") - 1, 0, len(LOSS_BINS) - 1)


def load_model(path=MODEL_PATH):
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    m["loss_logp"] = np.array(m["loss_logp"], dtype=np.float64)     # [ratings, bins]
    return m


def move_loss(engine, board, move, nodes=20000, out=None):
    """Expected-score loss of `move` for the side to move, from the engine's (UCI, fixed nodes)
    evaluation before and after it. `board` is not changed. If `out` is a dict, the engine's
    principal variation before the move is stored in out["pv"]."""
    import chess.engine
    limit = chess.engine.Limit(nodes=nodes)
    info = engine.analyse(board, limit)
    before = info["score"].pov(board.turn).score(mate_score=10000)
    if out is not None:
        out["pv"] = info.get("pv", [])
    b = board.copy(stack=False)
    b.push(move)
    if b.is_game_over():
        after = 10000 if b.is_checkmate() else 0          # from the mover's view: mate given, or a draw
    else:
        after = engine.analyse(b, limit)["score"].pov(board.turn).score(mate_score=10000)
    return float(np.clip(expected_score(before) - expected_score(after), 0, 1)), before, after


def summary(logp, loss):
    """What a game contributes to the estimate, in a form that adds up over games:
    the summed policy log-likelihood per rating, and the count of moves per loss bin."""
    logp = np.asarray(logp, dtype=np.float64).reshape(-1, len(RATINGS))
    llp = np.maximum(logp, math.log(1e-4)).sum(axis=0)
    counts = np.bincount(loss_bin(loss), minlength=len(LOSS_BINS)) if len(loss) else np.zeros(len(LOSS_BINS))
    return llp, counts.astype(np.float64)


def log_likelihood(llp, counts, model):
    """Weighted log-likelihood over the rating grid from a (possibly summed) summary."""
    return model["a"] * np.asarray(llp) + model["b"] * (model["loss_logp"] @ np.asarray(counts))


def posterior(ll, tau):
    z = ll / tau
    p = np.exp(z - z.max())
    return p / p.sum()


def summarize(post, low_q=0.1, high_q=0.9):
    """Posterior mean and an 80% range, interpolated on the rating grid."""
    r = np.array(RATINGS, dtype=np.float64)
    cdf = np.cumsum(post)
    mean = float((post * r).sum())
    low = float(np.interp(low_q, cdf, r))
    high = float(np.interp(high_q, cdf, r))
    return {"rating": round(mean), "low": round(low), "high": round(high)}


def estimate_from_summary(llp, counts, model=None):
    """Rating estimate from a game's summary, or several games' summaries added up: the
    posterior mean, and an 80% range from the posterior widened by tau (moves within a game
    aren't independent evidence)."""
    model = model or load_model()
    ll = log_likelihood(llp, counts, model)
    out = {"rating": summarize(posterior(ll, 1.0))["rating"]}
    rng = summarize(posterior(ll, model["tau"]))
    out["low"], out["high"] = rng["low"], rng["high"]
    out["moves"] = int(np.sum(counts))
    return out


def estimate(logp, loss, model=None):
    """Rating estimate for one player's moves: logp [moves, ratings], loss [moves]."""
    return estimate_from_summary(*summary(logp, loss), model)


def parse_humangame(block):
    """The engine's humangame output -> (plies (1-based), log P [moves, ratings])."""
    plies, rows = [], []
    for line in block.splitlines():
        if not line.startswith("PLY:"):
            continue
        f = dict(part.split(":", 1) for part in line.split("|"))
        plies.append(int(f["PLY"]))
        rows.append([math.log(max(float(p), 1e-12)) for p in f["P"].split(",")])
    return plies, np.array(rows, dtype=np.float64).reshape(-1, len(RATINGS))
