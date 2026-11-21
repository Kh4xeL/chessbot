# Chessbot: Hybrid NNUE Engine & AI Chess Tutor

A chess engine with a built-in tutor. You play it in the browser, and it shows you threats, your best moves and the plan behind them. After the game it explains your mistakes in plain language.

Under the hood it pairs two networks, each where it works best:

* **An NNUE evaluator** sits inside a classical alpha-beta search. It judges positions in about a microsecond, so the engine can look 12–20 moves deep in a second or two.
* **An AlphaZero-style CNN** (our original model) supplies "intuition" at the root. It decides which moves to look at first and, at lower Elo settings, which reasonable move a human would pick.

| | Before (CNN at every leaf) | Now (NNUE + CNN at the root) |
|---|---|---|
| Positions evaluated per second | ~700 | ~1,800,000 |
| Depth in 1–2 s | 3–4 | 11–19 |
| Strength vs Stockfish limited to 2400 Elo (0.3 s/move) | not measured | 80% (+16 =0 −4) |
| Strength vs Stockfish limited to 2700 Elo (0.3 s/move) | not measured | 45% (+6 =6 −8) |
| Average centipawn loss, same 47 positions from a real game | 32 | 21 |
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
             │  opening book · tactic annotations · threat detection (null move)        │
             │  post-game analysis with Stockfish (semantic_analysis.py)                │
             └──────────────────────────────────────┬────────────────────────────────────┘
                                                    │ stdin/stdout, one long-lived process
             ┌──────────────────────── engine (src/, C++17) ─────────────────────────────┐
             │  root:   CNN policy (policy_net.hpp) → move order, human-like choice      │
             │  search: iterative deepening, PVS, TT, null move, LMR, quiescence        │
             │  leaves: NNUE (nnue.hpp), updated incrementally move by move             │
             └───────────────────────────────────────────────────────────────────────────┘
```

### The NNUE (`src/nnue.hpp`)
* **Shape:** `768 → 2×512 → 1` with SCReLU. The input is one-hot piece-squares, seen from both the side to move and the opponent.
* **Training data:** 76M positions with Stockfish evaluations from the [Lichess evaluation database](https://database.lichess.org/#evals). Only "quiet" positions are kept: not in check, and the best move is not a capture.
* **Why it's fast:** a move only changes 2–4 pieces. Instead of recomputing the network, the engine adds and subtracts those few weight rows. The first layer is int16-quantized.

### The CNN policy (`src/policy_net.hpp`)
* **Model:** our original 4-block residual CNN (13×8×8 input), trained on human games. Only its policy head is used now.
* **No LibTorch:** BatchNorm is folded into the convolutions and the forward pass is plain C++. It runs once per move in about 3 ms.
* **Self-test:** `engine --selftest` checks its output against PyTorch.

### The search (`src/main.cpp`)
* **Core:** iterative deepening with Multi-PV at the root, so the tutor's top 3 moves have exact scores.
* **Pruning:** principal variation search, a transposition table, null-move pruning, reverse futility pruning, late move reductions and late move pruning.
* **Move ordering:** killer and history heuristics, plus check extensions.
* **Quiescence search:** follows captures to the end with delta pruning.
* **Elo setting:**
  * At 2900 or above it plays at full strength.
  * Below that it caps the depth and picks among near-best moves. Each move is weighted by its score and by how natural the CNN thinks it is, so weaker settings play plausible human moves rather than random blunders.
  * Set to 1500, it scored about 1410 against Stockfish limited to 1500.

---

## Getting started

### 1. Build the engine
You need a C++17 compiler and CMake, nothing else.

* **CLion:** open the folder and build the `engine` target in the Release profile. The output goes to `cmake-build-release/`.
* **Terminal:**
  ```bash
  cmake -S . -B cmake-build-release -DCMAKE_BUILD_TYPE=Release
  cmake --build cmake-build-release
  ```

The build uses `-march=native`, so build on the machine that will run it.

```bash
cmake-build-release/engine --selftest     # CNN matches PyTorch, NNUE loads
cmake-build-release/engine --bench 2000   # search speed on a few positions
```

### 2. Run the tutor
```bash
pip install fastapi uvicorn chess matplotlib
uvicorn python.api:app
```

Open <http://127.0.0.1:8000>.
* The API finds the engine in `cmake-build-release/` or `build/`.
* Stockfish, used only for post-game analysis, is found through `STOCKFISH_PATH`, your `PATH`, `tools/` or Homebrew.

### 3. Use the engine directly (optional)
```bash
engine "<fen>" [elo] [max_depth] [time_ms]   # one search, prints the result block
engine --server                               # persistent: "go <elo> <depth> <ms> <fen>", "newgame", "quit"
```

---

## Retraining the NNUE
This needs a CUDA GPU. On an RTX 5070 Ti, training takes about 5 minutes; streaming the data takes about an hour.

```bash
python training/prepare_data.py --max-lines 110000000   # streams the Lichess eval DB into data/ (~2.5 GB)
python training/train_nnue.py --epochs 15               # writes assets/nnue.bin
python training/match.py --elo 2400 --games 40          # strength check vs limited Stockfish
python training/match.py --elo 0 --ms 3000 --sf-ms 100  # vs full-strength Stockfish
```

`training/export_policy.py` re-exports `assets/policy_cnn.bin` from `assets/traced_alphazero.pt`.

---

## Project layout
```
src/            engine: main.cpp (search + protocol), nnue.hpp, policy_net.hpp
include/        chess.hpp (move generation, by Disservin)
python/         api.py (FastAPI bridge), semantic_analysis.py (post-game review)
web/            index.html, tree.js / tree.css (analysis move tree)
training/       prepare_data.py, train_nnue.py, export_policy.py, match.py
assets/         nnue.bin, policy_cnn.bin (+ test vectors), traced_alphazero.pt, book.bin
```

## Known limitations
* **No game history:** the engine only receives the current position, so it can't see repetitions from earlier in the game.
* **Single-threaded search:** the API serializes requests to the one engine process.
* **Rough Elo numbers:** they come from short matches against Stockfish's own Elo limiter, not from a rating list.

## Credits
* A university experiment in hybrid ML and classical game-playing systems.
* Training data: the [Lichess evaluation database](https://database.lichess.org/#evals) (CC0). Move generation: [chess-library](https://github.com/Disservin/chess-library). Analysis and benchmarking: [Stockfish](https://stockfishchess.org/).
* Code comments, parts of the frontend, and the NNUE engine rewrite were written with AI assistance.
