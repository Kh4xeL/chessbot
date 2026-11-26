# Chessbot: Hybrid NNUE Engine & AI Chess Tutor

A chess engine with a built-in tutor. You play it in the browser, and it shows you threats, your best moves and the plan behind them. After the game it explains your mistakes in plain language.

Under the hood it pairs two networks, each where it works best:

* **An NNUE evaluator** sits inside a classical alpha-beta search. It judges positions in about a microsecond, so the engine can look 12–20 moves deep in a second or two.
* **A rating-aware CNN** predicts the move a human of a given rating would play. It runs once at the root. It decides which moves to look at first and, at lower Elo settings, which reasonable move a player of that rating would pick.

With 5 pieces or fewer, **Syzygy endgame tablebases** give the exact result and the fastest winning move.

| | Before (CNN at every leaf) | Now (NNUE + CNN at the root) |
|---|---|---|
| Positions evaluated per second | ~700 | ~2.4M per thread, ~9.7M with 4 threads |
| Depth in 1–2 s | 3–4 | 14–20 |
| Strength vs Stockfish's Elo limiter (1 thread, 0.3 s/move) | not measured | ~3000 (+5 ± 43 vs SF 3000, −194 ± 44 vs SF 3190, the limiter's maximum; 150 games each) |
| Average centipawn loss, same 47 positions from a real game | 32 | 21 |
| Human move prediction (top-1, Lichess games) | 39.2% (original CNN) | 48.9% (rating-aware CNN) |
| Runtime dependencies | LibTorch (~2 GB) | none (plain C++17) |

Full-strength Stockfish is still far ahead: even with 3 s per move against Stockfish's 0.1 s, the engine lost all 10 games, lasting 57 moves on average.

---

## How it works

```
             ┌───────────────────────── web/ (chessboard.js UI) ─────────────────────────┐
             │  play vs bot · threat alerts · top-3 hints · post-game review + move tree │
             └──────────────────────────────────────┬────────────────────────────────────┘
                                                    │ HTTP (JSON)
             ┌──────────────────────── python/api.py (FastAPI) ──────────────────────────┐
             │  opening book · tactic annotations · threat detection (null move)         │
             │  post-game analysis with Stockfish (semantic_analysis.py)                 │
             └──────────────────────────────────────┬────────────────────────────────────┘
                                                    │ stdin/stdout, a pool of long-lived processes
             ┌──────────────────────── engine (src/, C++17) ─────────────────────────────┐
             │  root:   rating-aware CNN (policy_net.hpp) → move order, human-like choice│
             │  search: Lazy SMP, iterative deepening, PVS, TT, SEE, LMR, quiescence     │
             │  leaves: NNUE (nnue.hpp), updated incrementally move by move              │
             │  ≤5 pieces: Syzygy tablebases via Fathom (tablebase.hpp)                  │
             └───────────────────────────────────────────────────────────────────────────┘
```

### The NNUE (`src/nnue.hpp`)
* **Shape:** `768 → 2×512 → 1` with SCReLU. The input is one-hot piece-squares, seen from both the side to move and the opponent.
* **Training data:** 262M quiet positions (not in check, best move not a capture):
  * 76M with Stockfish evaluations from the [Lichess evaluation database](https://database.lichess.org/#evals);
  * 6M from real Lichess games, labelled by Stockfish;
  * 180M from the engine's own self-play games (below).
* **Training target:** 70% the search score (as a win probability), 30% the game result where one is known.
* **Self-play:** `engine --datagen` plays games against itself at 5,000 nodes per move from 8 random opening moves. It records the search score and the game's result for each quiet position. Games end early on a tablebase result or a long run of decisive or drawn scores. Each round adds 30M positions (about 4 hours on 6 cores) and retrains on everything so far:

  | Round | Positions | Elo vs the previous network |
  |---|---|---|
  | 1 | 112M | +107 ± 24 |
  | 2 | 142M | +39 ± 14 |
  | 3 | 172M | +26 ± 11 |
  | 4 | 202M | +17 ± 9 |
  | 5 | 232M | +14 ± 8 |
  | 6 | 262M | +12 ± 8 |

  Each was an SPRT at 20k nodes per move. With 262M positions, training uses 15 of 16 GB of GPU memory.
* **Why it's fast:** a move only changes 2–4 pieces. Instead of recomputing the network, the engine adds and subtracts those few weight rows, in one pass per move. The first layer is int16-quantized, and the output layer uses AVX2. The AVX2 path and the one-pass update took the speed from 1.4M to 2.2M positions per second, worth +57 ± 14 Elo at 10 s + 0.1 s per game.
* **Tried and not adopted:**
  * A hidden layer of 1024 instead of 512: −12 ± 8 Elo at 10 s + 0.1 s per game. Its evaluations are more accurate (validation loss 0.0068 vs 0.0071 on round 3 data), but it is about 30% slower.
  * A "v2" network with king buckets (the board seen relative to each king, mirrored) and separate output layers by piece count: −17 ± 13 Elo at 10 s + 0.1 s per game.
  * A 50/50 mix of score and result as the target instead of 70/30: −33 ± 14 Elo at 20k nodes per move.

  The engine and `train_nnue.py --arch v2 / --hidden` still support these networks.

### The rating-aware policy (`src/policy_net.hpp`)
* **Model:** a 6-block × 96-channel residual CNN, in the spirit of Maia. Its input is:
  * the board from the side to move's view;
  * castling rights;
  * both players' ratings, given as constant input planes.

  A convolutional head scores every from-to move.
* **Training data:** 85.8M positions from 9.4M rated Lichess games (October 2026), excluding bullet and abandoned games. Positions are capped per 100-Elo band, so 600–1000 and 2400+ players are well represented.
* **Accuracy:** it predicts the human's actual move 48.9% of the time, against 39.2% for the original CNN on the same positions. By rating:

  | Rating | 600 | 1000 | 1400 | 1800 | 2200 | 2400 |
  |---|---|---|---|---|---|---|
  | Top-1 | 43.6% | 48.4% | 49.7% | 50.0% | 48.8% | 47.9% |

  The rating input matters. Told that 400–799 players are rated 2500, it predicts their moves 36.8% of the time instead of 42.3%.
* **No LibTorch:** BatchNorm is folded into the convolutions and the forward pass is plain C++ (about 3 ms). The original AlphaZero-style CNN (`policy_cnn.bin`) is still supported as a fallback.
* **Self-test:** `engine --selftest` checks the C++ output against PyTorch (also part of `engine --test`).

### Rating estimate (`python/elo_estimate.py`)
* **What:** after each game, an estimate of the rating your moves resembled, for example "~1550 (1420–1690)". The play panel also shows an overall estimate from your last 10 games and a trend line. Games are stored in SQLite (`data/tutor.db`).
* **How:** each of your moves gives two pieces of evidence over ratings 600–2800:
  * how likely a player of each rating was to play it (the rating-aware CNN, via the engine's `humangame` command);
  * how much it lost by the engine's evaluation, against how often players of each rating lose that much.

  These are combined into a best-fitting rating and an 80% range.
* **Checked on 5,172 held-out Lichess games** from a month the CNN never saw (September 2026), fitted on half and tested on the other half:

  | | Median error | Mean error | Within ±150 | 80% range holds the true rating |
  |---|---|---|---|---|
  | One game | 194 | 255 | 41% | 87% |
  | 5 games | 99 | 126 | 66% | 79% |

  A single game is a small sample, and single-game estimates lean toward the middle (low ratings read high, high ratings low). Over several games the estimate is within about ±100 typically. The move choice carries most of the signal; the eval losses add a little. Lichess ratings come from timed games, so untimed games against the bot may read somewhat higher.

### Adaptive opponent (`python/adaptive.py`)
* **What:** the "Adaptive" box next to the Elo slider. The bot starts at your overall rating estimate (1200 without one) and adjusts so you win about half your games. The play panel shows "Adaptive opponent: playing at ~1600 for you".
* **Between games:** each finished game, including a resignation, moves it toward a 50% score: ±150 for the first 5 games, ±70 after. In those first games it is also pulled a quarter of the way toward the game's rating estimate.
* **Within a game:** up to +250 Elo while you are far ahead, and down to −150 while you are far behind, so a lost position eases off a little without becoming a gift.
* **Range:** 600–2800. The engine's 600 setting picks moves more loosely than the slider's lowest (800), so it can ease off for beginners.
* **Simulated** with the engine at a fixed setting standing in for the player, 40 games each from a start of 1200 (`scripts/simulate_adaptive.py`):

  | Player | Bot settles at | Player's score after game 10 |
  |---|---|---|
  | 800 | ~900 (700–1190) | 50% |
  | 1000 | ~1050 (840–1190) | 50% |
  | 1400 | ~1460 | 50% |
  | 1800 | ~1820 | 52% |
  | 2200 | ~2210 | 48% |

  Below about 1200 the engine's settings play alike (1-ply search), so the bot wanders further there before the score moves. With an 800 floor the 1000 player scored only 33%, which is why the floor is 600.

### Insights at your level (`python/insights.py`)
The tutor compares what players at your rating usually play with the engine's choice. Your level is your rating estimate (or the adaptive opponent's Elo, or 1500). Three kinds of insight:
* **What your level plays**, in the form "Most ~1500 players play Nf3 here (41%). The engine prefers d4, about 0.8 pawns better."
* **Hard to find:** the engine's move is played by under 10% of players at your level, and it is at least 0.7 pawns better than the popular choice. In the Two Knights line 4.Ng5 d5 5.exd5, only 2% of ~1200 players find 5...b5.
* **Trap warnings:** a move at least 12% of your level plays that loses 1.2 pawns or more. In Légal's trap (5.Nxe5), 80% of ~1000 players take the queen with 5...Bxd1 and walk into mate.

Each popular move is checked by a short fixed-node search, and with full hints the insights reuse the tutor's own lines, so the two never disagree. With full hints you see all three; with threat warnings only, the trap warnings. Hints also scale with your level: 3-ply lines and only the good moves below 1200, 5 plies up to 1800, full 8-ply lines above.

### Training (`web/train.html`, <http://127.0.0.1:8000/train>)
* **Your puzzles** (`python/puzzles.py`): when a finished game is rated, each of your moves that gave up 15% or more of your winning chances becomes a puzzle: the position before it, "find a better move". Any move within 5% of the engine's best counts as solved. Solved puzzles come back after 1, 3, 7, 16 and 35 days, then retire; a miss comes back in 10 minutes.
* **Endgames** (`python/endgames.py`): convert a won endgame against perfect tablebase defence. There are eight endings, from queen vs king to rook and pawn vs rook and bishop and knight vs king. Positions are random wins checked with Syzygy. The defence resists longest by DTZ, and "par" is what a tablebase-perfect attacker needs. A move that throws the win away is flagged at once, with Undo.
* **Opening explorer** (`python/explorer.py`): for each candidate move, its share in the opening book, the share of players at your level who play it, and the engine's eval after it. Click a move to follow it.
* **Guess the move** (`python/guess.py`): step through 150 strong Lichess games (both players 2400+, from the held-out month, without player names) and guess one side's moves. The game move scores 10; a move the engine rates as good scores 8; others score less the more they give up. Each answer shows how many 2400 players would play your move and the game's.

### Endgame tablebases (`src/tablebase.hpp`)
* **What:** Syzygy 3–5 piece tables (WDL + DTZ, ~940 MB), probed through [Fathom](https://github.com/jdart1/Fathom) (MIT, in `include/fathom/`).
* **In the search:** win/draw/loss probes after captures and pawn moves.
* **At the root:**
  * only moves that keep the theoretical result are searched;
  * when winning, the engine plays the fastest-converting move. It mates with KRvK in 25 plies from a test position.
* **For the tutor:** the API reports the result (`tablebase`), and the eval bar shows "TB +", "TB =" or "TB −".
* **Below 2200 Elo** the bot ignores the tables and plays endgames by its own judgement.

### The search (`src/main.cpp`)
* **Core:** iterative deepening with Multi-PV at the root, so the tutor's top 3 moves have exact scores, plus aspiration windows.
* **Threads:** Lazy SMP. Helper threads share a lock-free transposition table (`--threads N`). 4 threads gained about +160 Elo over 1.
* **Pruning:** principal variation search, null-move pruning (off with less than a rook or two minor pieces, where zugzwang is common), reverse futility pruning, ProbCut, late move reductions, late move pruning and SEE pruning. Pruning is stricter when the side to move's eval is falling ("improving" flag).
* **Extensions:** check extensions, and singular extensions with multi-cut. When the hash move is much better than every alternative, it is searched one ply deeper.
* **Move ordering:** hash move, captures ranked by static exchange evaluation (SEE), then killers, counter-moves, and history with continuation history (what worked after the previous 1–2 moves).
* **Transposition table:** 64 MB in 4-slot buckets, each one cache line. Entries carry the search that wrote them, so leftovers from earlier moves are replaced first.
* **Quiescence search:** follows captures to the end, skipping losing ones (SEE) and hopeless ones (delta pruning). It reads and writes the hash table.
* **Tuning:** 20 search constants are UCI options (`rfp_margin`, `lmr_div`, …). Their defaults come from 60,000 SPSA games (`scripts/spsa.py`).
* **Measured gains (SPRT, 20k nodes/move, each against the previous step):**

  | Change | Elo |
  |---|---|
  | continuation history + counter-moves | +15 ± 8 |
  | improving flag | +19 ± 9 |
  | singular extensions | +11 ± 7 |
  | hash table in quiescence | +21 ± 10 |
  | ProbCut | +25 ± 11 |
  | TT buckets + aging | +0 ± 4 (kept: it matters once the table fills in longer searches) |
  | SPSA-tuned constants | +58 ± 17 |
  | **all of the above vs the previous main, 10 s + 0.1 s per game** | **+140 ± 23** |

  History-adjusted LMR was tried and dropped (−1 ± 4).
* **Game history:** the API sends the game's moves, so the search sees repetitions.
* **Elo setting:**
  * At 2900 or above it plays at full strength.
  * Below that it caps the depth and picks among near-best moves. Each move is weighted by its score and by how likely a player of that rating is to play it (the rating-aware CNN), so weaker settings play the moves such players actually make rather than random blunders.
  * Calibrated against Stockfish's Elo limiter, 150 games per point at 0.3 s/move (`scripts/calibrate_elo.py`): 1400 → ~1391, 1800 → ~1749, 2200 → ~2268, 2600 → ~2600, each ± 45–59. Below 1400 the setting is extrapolated, because Stockfish can't be limited that low.
  * Limited settings cap the depth: 3 plies at 1400, 7 at 1800, 10 at 2200, 12 at 2600. The self-play networks made every setting 110–245 Elo stronger, so the caps and sampling temperatures were lowered to match.

---

## Getting started

### 1. Build the engine
You need a C and C++17 compiler and CMake, nothing else.

* **CLion:** open the folder and build the `engine` target in the Release profile. The output goes to `cmake-build-release/`.
* **Terminal:**
  ```bash
  cmake -S . -B cmake-build-release -DCMAKE_BUILD_TYPE=Release
  cmake --build cmake-build-release
  ```

By default the build is optimised for your own CPU (`-march=native`). For a binary you want to share, add `-DENGINE_NATIVE=OFF`: it then runs on any x86-64 CPU with AVX2, at about the same speed.

```bash
cmake-build-release/engine --test                  # all self-checks: NNUE, SEE, mates, repetition, CNN
cmake-build-release/engine --bench 2000 --threads 4  # search speed on a few positions
```

### 2. Run the tutor
```bash
pip install -r requirements.txt
uvicorn python.api:app
```

Open <http://127.0.0.1:8000> to play, or <http://127.0.0.1:8000/train> for the trainers.
* The API finds the engine in `cmake-build-release/` or `build/`. It runs `ENGINE_PROCESSES` (default 3) engines with `ENGINE_THREADS` (default 4) threads each, so hints and threat checks don't wait behind the bot's move.
* Stockfish, used only for post-game analysis, is found through `STOCKFISH_PATH`, your `PATH`, `tools/` or Homebrew.

### Endgame tablebases (optional, ~940 MB)
```bash
python scripts/download_syzygy.py        # 3-4-5 piece Syzygy tables into tools/syzygy
```
The engine finds them in `tools/syzygy` (or `--syzygy DIR` / `$SYZYGY_PATH`). Without them it simply searches endgames normally.

### Play online (lichess-bot, optional)
The engine speaks UCI, so [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) can run it on Lichess:
1. Create a **new** Lichess account for the bot. It must not have played any games, and upgrading it to a bot account is permanent.
2. Create an API token for it with the `bot:play` scope, at <https://lichess.org/account/oauth/token>.
3. Clone lichess-bot and install its requirements. Follow its wiki to upgrade the account to a bot account.
4. Copy `deploy/lichess-bot.example.yml` to `config.yml` in the lichess-bot folder. Set the paths to this project, and put the token in the file or in `LICHESS_BOT_TOKEN`. Don't commit a real token.
5. Run `python lichess-bot.py`.

The example plays one game at a time at full strength. For a human-like opponent, set `UCI_LimitStrength: true` and `UCI_Elo` in its `uci_options`.

### 3. Use the engine directly (optional)
The engine speaks **UCI**, so any chess GUI (CuteChess, Arena, Banksia) or match runner can use it. Start it with no arguments. It supports these options: `Threads`, `MultiPV`, `Move Overhead`, `SyzygyPath`, `UCI_LimitStrength` and `UCI_Elo`.

```bash
engine                                        # UCI
engine "<fen>" [elo] [max_depth] [time_ms]   # one search, prints the result block
engine --server [--threads N] [--syzygy DIR]  # persistent, see below
engine --eval "<fen>" ...                     # static NNUE evaluation
```

In server mode the commands are:
* `go <elo> <depth> <ms> <fen> [moves m1 m2 ...]`
* `humanprobs <elo> <fen> [moves ...]`: each legal move's probability for a player of that rating, most likely first
* `humangame <elo,elo,...> <w|b|all> <fen> moves m1 m2 ...`: for each move of that side, the played move's probability and rank at each rating
* `newgame`
* `quit`

Each `go` prints a block of `BEST_MOVE`, `EVAL`, `MATE`, `TABLEBASE`, `DEPTH_REACHED` and `RANK` lines, ending with `END`.

### 4. Run the tests
```bash
cmake-build-release/engine --test
pytest tests/
```

GitHub Actions runs both on every push to `main` and on every pull request (`.github/workflows/ci.yml`).

### 5. Test whether a change gains Elo (SPRT)
`scripts/test_change.py` plays a new build against a base build with [fastchess](https://github.com/Disservin/fastchess) (put it in `tools/fastchess/`).
* **Setup:** fixed nodes per move over 5,000 balanced openings (`tests/openings.epd`), each played with both colours.
* **SPRT:** a sequential test that stops as soon as the result is clear.
* **Inputs:** engines can be `.exe` paths or git refs; refs are built in a worktree automatically.
* **Log:** every result goes to `data/tests.csv`.

```bash
python scripts/test_change.py --new my-branch --base main                # gainer: SPRT [0, 5]
python scripts/test_change.py --new HEAD --base main --elo0 -5 --elo1 0  # "no worse" check
```

**Speed:** about 13,000 games an hour at 20k nodes/move with 6 games in parallel.

**Checks of the harness:**
* the same build against itself: −0.0 ± 11.8 Elo over 2,000 games;
* full strength against the Elo-2400 setting: a gain accepted after 304 games.

`scripts/make_openings.py` rebuilds the opening suite from the Lichess games data.

**Tuning:** `scripts/spsa.py` tunes the search constants with SPSA. It saves its state after every batch, so a run can be stopped and resumed. `--workers 2 --concurrency 6` plays 12 games at once.

```bash
python scripts/spsa.py --engine cmake-build-release/engine.exe --iterations 2500 --workers 2
```

---

## Retraining
Training needs a CUDA GPU. On an RTX 5070 Ti each network trains in about 5–30 minutes; streaming and labelling the data take longer.

**NNUE**
```bash
python training/prepare_data.py --max-lines 110000000      # Lichess eval DB -> data/ (~1 h, ~2.5 GB)
python training/label_fens.py --workers 6                  # optional: label game positions with Stockfish
python training/train_nnue.py --arch v1 --data data        # writes assets/nnue.bin (v2: --arch v2)
python scripts/datagen.py --gen gen7 --procs 6             # self-play with the current net -> data/selfplay/gen7 (~4 h)
python training/train_nnue.py --wdl 0.3 --data data data/games_labeled data/selfplay/gen*/part*
```

**Rating-aware policy**
```bash
python training/prepare_games.py --games 10000000          # stream rated games -> data/policy (~3 GB download)
python training/train_policy.py --epochs 2                 # writes assets/policy_elo.bin (~30 min)
```

**Rating estimate**
```bash
python training/fetch_heldout.py                           # held-out rated games -> data/heldout_2026-09.pgn (~20 MB stream)
python training/heldout_features.py                        # per-move policy (GPU) and eval-loss features (~25 min, 6 processes)
python training/fit_elo_estimator.py                       # writes assets/elo_model.json and prints the held-out accuracy
```

**Measuring strength**
```bash
python training/match.py --elo 2400 --games 40             # vs Stockfish's Elo limiter
python training/match.py --elo 1800 --our-elo 1800         # check one Elo-slider setting
python training/match.py --opponent cmake-build-release/engine.exe --opp-cwd old_nets/ --games 300
                                                           # A/B: the opponent loads old_nets/assets/*
```

`training/export_policy.py` re-exports the original CNN (`assets/policy_cnn.bin`) from `assets/traced_alphazero.pt`.

---

## Project layout
```
src/            engine: main.cpp (search + protocol), nnue.hpp, policy_net.hpp, tablebase.hpp, tests.hpp
include/        chess.hpp (move generation, by Disservin), fathom/ (Syzygy probing, MIT)
python/         api.py (FastAPI bridge), semantic_analysis.py (post-game review),
                elo_estimate.py + rating_store.py (rating estimate and history), adaptive.py, insights.py,
                puzzles.py, endgames.py, explorer.py, guess.py (training)
web/            index.html, train.html, ui.js, tree.js / tree.css (analysis move tree)
training/       prepare_data.py, prepare_games.py, label_fens.py, train_nnue.py, train_policy.py,
                export_policy.py, match.py, fetch_heldout.py, heldout_features.py, fit_elo_estimator.py
scripts/        datagen.py (self-play), test_change.py (SPRT), spsa.py, calibrate_elo.py, simulate_adaptive.py,
                download_syzygy.py, make_openings.py, make_guess_games.py
assets/         nnue.bin, policy_elo.bin, policy_cnn.bin (+ test vectors), traced_alphazero.pt, book.bin,
                elo_model.json, guess_games.pgn
deploy/         lichess-bot.example.yml
tests/          API and UCI tests (pytest), openings.epd
```

## Known limitations
* **Rough Elo numbers:** they come from short matches (±100 Elo) against Stockfish's own Elo limiter, not from a rating list.
* **Tablebases are optional:** they aren't in the repository (940 MB). Without them, endgames are played by search alone.
* **Human-like, not human:** weaker settings pick moves real players of that rating make, but the bot doesn't simulate time pressure or tilt.

## Credits
* A university experiment in hybrid ML and classical game-playing systems.
* **Training data:** the [Lichess evaluation database](https://database.lichess.org/#evals) and [rated game archives](https://database.lichess.org/) (both CC0). The rating-aware policy follows the idea of [Maia](https://maiachess.com/).
* **Libraries and tools:**
  * move generation: [chess-library](https://github.com/Disservin/chess-library);
  * tablebase probing: [Fathom](https://github.com/jdart1/Fathom), with tables from the Lichess mirror;
  * labelling, analysis and benchmarking: [Stockfish](https://stockfishchess.org/).
* Code comments, parts of the frontend, and the NNUE engine rewrite were written with AI assistance.
