# 🧠 Little Go Competitive Engine

A compact **5×5 Go AI engine** implemented in **Python + NumPy**, featuring **Alpha–Beta pruning with iterative deepening, aspiration windows, and bitboard compression**.  
This engine was developed for competitive mini-Go environments and achieves over **90% win rate** against standard baselines.

---

## 📘 Overview

**Little Go Competitive Engine** is a lightweight yet high-performance AI designed to play 5×5 Go.  
It leverages Alpha–Beta search combined with heuristic move ordering, symmetry reduction, and dynamic evaluation to play competitively within strict time constraints.

---

## ⚙️ Technical Highlights

- **Iterative Deepening + Aspiration Window Search** (1–5 plies, self-tuned per time limit)
- **Multi-layer Move Ordering:** killer move, history heuristic, transposition table, top-10 filter (≈35% node cut)
- **Bitboard Compression:** 5×5 board compressed into 50-bit canonical form (≈90% smaller keys)
- **Dynamic Evaluation Function:** territory, liberty, influence, and ko-risk weighting by game phase
- **Transposition Table & Symmetry Canonicalization** to reduce repeated state evaluation
- **Tournament Performance:** ~90% win rate vs greedy, aggressive, and vanilla alpha-beta opponents

---

## 🧩 Project Structure

```
Little-Go-Competitive-Engine/
│
├── AlphabetaPlayer.py      # Alpha-Beta pruning player class
├── Board.py                # Board representation and state manipulation
├── chest_alpha.py          # Experimental Alpha-Beta testing script
├── chest.py                # Basic game state control logic
├── input.txt               # Input file containing side and current/previous board states
├── LICENSE                 # License file
├── my_player3.py           # Main AI entry point (iterative deepening + aspiration window)
├── output.txt              # Output file for AI move results
├── qlearnerExp.py          # Q-learning experiment (optional extension)
├── qtable.txt              # Pretrained Q-table values for reinforcement learning variant
├── RandomPlayer.py         # Baseline random move generator for benchmarking
├── README.md               # Project documentation (this file)
├── rule.txt                # Game rule summary for 5×5 Little Go
└── test.py                 # Unit tests and simulation scripts
```

---

## 🚀 How to Run

### 1️⃣ Prepare the Input File (`input.txt`)

```
1
00000
00000
00000
00000
00000
00000
00000
00000
00000
00000
```
- First line: current side (`1` = Black, `2` = White)  
- Next 10 lines: previous and current board states

### 2️⃣ Run the Engine

```bash
python3 my_player3.py
```

### 3️⃣ Output Example (`output.txt`)
```
2,3
```
This indicates the AI plays at row 2, column 3.

---

## 🧠 Core Class — AlphabetaPlayer

The core of the AI is implemented in `my_player3.py` as the `AlphabetaPlayer` class.

### Key Methods

| Method | Description |
|--------|--------------|
| `alphabeta()` | Minimax with alpha–beta pruning and killer/history heuristics |
| `choose_move()` | Iterative deepening with aspiration windows and time control |
| `evaluate_board()` | Weighted evaluation combining group structure, territory, liberties, and ko-risk |
| `limit_moves()` | Multi-level move ordering and pruning (top 10 heuristic filtering) |
| `canonical_board()` | Symmetry normalization using 8 rotations/flips |

---

## 📊 Performance Summary

| Metric | Value |
|--------|-------|
| Board size | 5 × 5 |
| Average search depth | 1–5 plies (adaptive) |
| Node reduction | ≈ 35% |
| Transposition table compression | 90% smaller keys |
| Tournament win rate | 90% vs. baselines |
| Average move time | ≤ 6 seconds |

---

## 🧩 Future Work

- Integrate **Monte Carlo Tree Search (MCTS)** hybrid strategy  
- Add **Zobrist hashing** for faster state lookups  
- Develop a **visual Go board GUI**  
- Incorporate a **neural policy network** for adaptive move ordering  

---

## 👨‍💻 Author

**Aaron Lin**  
University of Southern California (USC)  
CS561 – Artificial Intelligence / Competitive Game Agent Project  
[GitHub: @w224lin](https://github.com/w224lin)

---
