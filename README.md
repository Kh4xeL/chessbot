# Chessbot: Hybrid NNUE Engine & AI Chess Tutor

A chess engine with a built-in tutor. You play it in the browser, and it shows you threats, your best moves and the plan behind them. After the game it explains your mistakes in plain language.

Under the hood it pairs two networks, each where it works best:

* **An NNUE evaluator** sits inside a classical alpha-beta search. It judges positions in about a microsecond, so the engine can look 12–20 moves deep in a second or two.
* **A rating-aware CNN** predicts the move a human of a given rating would play. It runs once at the root. It decides which moves to look at first and, at lower Elo settings, which reasonable move a player of that rating would pick.

With 5 pieces or fewer, **Syzygy endgame tablebases** give the exact result and the fastest winning move.

| | Before (CNN at every leaf) | Now (NNUE + CNN at the root) |
|---|---|---|
| Positions evaluated per second | ~700 | ~1.8M per thread, ~7.6M with 4 threads |
| Depth in 1–2 s | 3–4 | 14–20 |
| Strength vs Stockfish's Elo limiter (1 thread, 0.3 s/move) | not measured | ~2860 before the search work below (+140 Elo since) |
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
* **Training data:** 76M positions with Stockfish evaluations from the [Lichess evaluation database](https://database.lichess.org/#evals). Only "quiet" positions are kept: not in check, and the best move is not a capture.
* **Why it's fast:** a move only changes 2–4 pieces. Instead of recomputing the network, the engine adds and subtracts those few weight rows. The first layer is int16-quantized.
* **Tried and not adopted:** two changes that usually help, each A/B-tested over 300 games:
  * A "v2" network with king buckets (the board seen relative to each king, mirrored) and separate output layers by piece count.
  * 5.75M extra positions from real Lichess games, labelled by Stockfish and blended with the game results.

  Neither beat this network: every variant landed within ±20 Elo of it, inside the margin of error. The engine and `train_nnue.py --arch v2` still support v2 networks.

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
  * Calibrated against Stockfish's Elo limiter, 40 games per point: 1400 → ~1490, 1800 → ~1870, 2200 → ~2190, 2600 → ~2630. Below 1400 the setting is extrapolated, because Stockfish can't be limited that low.

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

Open <http://127.0.0.1:8000>.
* The API finds the engine in `cmake-build-release/` or `build/`. It runs `ENGINE_PROCESSES` (default 3) engines with `ENGINE_THREADS` (default 4) threads each, so hints and threat checks don't wait behind the bot's move.
* Stockfish, used only for post-game analysis, is found through `STOCKFISH_PATH`, your `PATH`, `tools/` or Homebrew.

### Endgame tablebases (optional, ~940 MB)
```bash
python scripts/download_syzygy.py        # 3-4-5 piece Syzygy tables into tools/syzygy
```
The engine finds them in `tools/syzygy` (or `--syzygy DIR` / `$SYZYGY_PATH`). Without them it simply searches endgames normally.

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
```

**Rating-aware policy**
```bash
python training/prepare_games.py --games 10000000          # stream rated games -> data/policy (~3 GB download)
python training/train_policy.py --epochs 2                 # writes assets/policy_elo.bin (~30 min)
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
python/         api.py (FastAPI bridge), semantic_analysis.py (post-game review)
web/            index.html, tree.js / tree.css (analysis move tree)
training/       prepare_data.py, prepare_games.py, label_fens.py, train_nnue.py, train_policy.py,
                export_policy.py, match.py
scripts/        download_syzygy.py, make_openings.py, test_change.py (SPRT)
assets/         nnue.bin, policy_elo.bin, policy_cnn.bin (+ test vectors), traced_alphazero.pt, book.bin
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
