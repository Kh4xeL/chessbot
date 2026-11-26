"""
Adaptive opponent: picks the bot's Elo setting so games are won about half the time.

  * Start: the player's overall rating estimate (python/elo_estimate.py), or 1200.
  * After each game: an Elo-style step toward a 50% score (bigger for the first games), plus,
    early on, a pull toward that game's rating estimate.
  * Within a game: when the player is clearly winning the bot plays a bit stronger; when the
    player is collapsing it eases off a little (a smaller step, so the game stays a game).

The functions are pure; python/rating_store.py keeps the state and python/api.py calls them.
"""

MIN_ELO, MAX_ELO = 600, 2800     # below the slider (800) so it can ease off for beginners; 2900+ is full strength
DEFAULT_ELO = 1200
EARLY_GAMES = 5                  # games with the larger step and the pull toward the estimate
K_EARLY, K_LATE = 150, 70        # Elo step per point of score away from 0.5
ESTIMATE_PULL = 0.25             # early games: weight of the game's rating estimate


def clamp(elo):
    return int(round(min(MAX_ELO, max(MIN_ELO, elo))))


def initial_elo(overall_estimate=None):
    return clamp(overall_estimate) if overall_estimate else DEFAULT_ELO


def update(elo, games, score, game_estimate=None):
    """New base Elo after a game. score: the player's result (1 win, 0.5 draw, 0 loss)."""
    k = K_EARLY if games < EARLY_GAMES else K_LATE
    new = elo + k * (score - 0.5) * 2
    if game_estimate is not None and games < EARLY_GAMES:
        new = (1 - ESTIMATE_PULL) * new + ESTIMATE_PULL * game_estimate
    return clamp(new)


def in_game_adjust(player_eval_pawns):
    """Elo offset for the bot's next move from the current eval, in pawns from the player's
    side: up to +250 when the player is far ahead, down to -150 when far behind."""
    if player_eval_pawns is None:
        return 0
    e = player_eval_pawns
    if e > 1.5:
        return int(min(250, (e - 1.5) * 100))
    if e < -1.5:
        return -int(min(150, (-e - 1.5) * 60))
    return 0


def score_from_result(result, side):
    """The player's score from a PGN result and their colour ('w'/'b'); None if unfinished."""
    if result == "1/2-1/2":
        return 0.5
    if result in ("1-0", "0-1"):
        return 1.0 if (result == "1-0") == (side == "w") else 0.0
    return None
