***

# AlphaZero-Style Hybrid Chess Engine & AI Tutor

A high-performance chess engine and interactive tutor combining classical **Alpha-Beta search** with a **Deep Convolutional Neural Network (CNN)** (via LibTorch). 

This project features a C++ core for heavy tactical calculation, a Python FastAPI bridge for semantic move analysis, and a responsive web-based UI that acts as a real-time chess tutor.

## 📁 Project Structure

* **`src/`**: Core C++ engine logic (Minimax, Iterative Deepening, Policy Pruning).
* **`include/`**: C++ header files (`chess.hpp`).
* **`python/`**: FastAPI backend (`api.py`) that bridges the web UI to the C++ executable and annotates tactical motifs.
* **`web/`**: Frontend chessboard interface (`index.html`) using `chessboardjs` and `chess.js`.
* **`assets/`**: Contains the PyTorch model (`traced_alphazero.pt`) and the Polyglot opening book (`book.bin`).
* **`CMakeLists.txt`**: Build configuration for the C++ engine.

## 🧠 Core Engine Architecture (C++)
* **13-Channel CNN Evaluation**: Positional judgment is powered by a PyTorch CNN trained on 155+ million positions from Grandmaster games. The board is converted into a 13-channel tensor to evaluate deep positional concepts (King safety, outposts) that traditional material-counting engines miss.
* **Tapered Policy Pruning (Beam Search)**: To solve the Horizon Effect, the engine uses the Neural Network's raw intuition to aggressively prune unpromising branches before search begins. 
  * **Midgame**: Keeps the top 50% of intuitive moves to ensure defensive safety nets.
  * **Endgame**: Tapers down to the top 25% of moves, drastically accelerating search depth to solve complex endgames.
* **Search Optimizations**: Implements Transposition Tables (TT) for memory preservation, Quiescence Search for noisy captures, and Time-Managed Iterative Deepening.

## 🎓 AI Tutor Features (Python API)
Instead of forcing the C++ engine to handle string formatting, the Python API acts as a "Color Commentator," translating raw C++ calculations into human-readable advice.
* **Tactical Motif Annotator**: Python simulates the engine's suggested moves to dynamically generate warnings like `💥 Captures Knight` or `🔱 Creates a Fork/Double Attack!`.
* **Level 1 Assistance (Threat Detection)**: Uses **Null-Move Pruning** at the API level. By intentionally passing the human's turn, the API asks the C++ engine to find the opponent's most dangerous sequence, alerting the user to hanging pieces or imminent forks.
* **Level 2 Assistance (Deep PV)**: Displays the top 3 best calculated moves, complete with expected continuation strings (Principal Variation) and tactical annotations.

## 🚀 Getting Started

### 1. Build the C++ Engine
Ensure you have **LibTorch** and **CMake** installed on your system.
```bash
mkdir build && cd build
cmake ..
make
```

### 2. Start the API
Run the FastAPI server from the project root:
```bash
uvicorn python.api:app --reload
```

### 3. Play and Learn
Open your browser and navigate to `http://127.0.0.1:8000` to interact with the AI Tutor. Adjust the depth and Elo sliders dynamically to test the engine's strength!

## 🤖 Attribution
* **Architecture**: Developed as a university experiment in Hybrid ML-Classical systems.
* **Code Comments**: The explanatory comments throughout the C++ and Python source files were generated with the assistance of AI to ensure clarity for new contributors.
* **Frontend**: The initial `index.html` structure and the integration logic for the chessboard UI were AI-assisted.

***
