#  CheckersAI – Python Draughts Game with Neural Network Opponent

A fully featured **Checkers (Draughts) game written in Python**, where the player competes against an **AI powered by a neural network built with PyTorch**.  
Designed for clarity, extensibility, and experimentation with modern machine learning techniques.

---

##  What is CheckersAI?

CheckersAI is an interactive **turn-based Checkers game** featuring:

- A robust Python game engine implementing standard American Checkers rules
- A neural network opponent trained using PyTorch
- A modular architecture separating game logic, AI, training, and UI
- Support for self-play and learning-based experimentation

This project is intended for **educational, research, and portfolio purposes**.

---

##  Features

-  Complete checkers rules (mandatory captures, multi-jumps, king promotion)
-  Neural network–based AI opponent
-  Clean, modular project structure
-  CLI gameplay (with optional graphical interface)
-  Self-play and supervised training support
-  Clear documentation and extensible codebase

---

##  Project Structure

```text
checkers-ai/
├── ai/
│   ├── model.py           # PyTorch neural network definitions
│   ├── trainer.py         # Training logic and loops
│   ├── encoder.py         # Board → tensor encoding
│   └── agent.py           # AI decision-making logic
│
├── game/
│   ├── board.py           # Board and piece representation
│   ├── rules.py           # Rules and move validation
│   ├── engine.py          # Game flow and state transitions
│   └── utils.py           # Helper utilities
│
├── ui/
│   ├── cli.py             # Command-line interface
│   └── gui.py             # Optional graphical interface (pygame)
│
├── train.py               # Neural network training script
├── main.py                # Application entry point
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🛠Prerequisites

- Python **3.10+**
- PyTorch
- NumPy
- (Optional) pygame for graphical UI
- (Optional) matplotlib for training visualizations

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ️ How to Play

Run the game from the project root:

```bash
python main.py
```

You will be prompted to choose your color and enter moves via the command-line interface.  
Illegal moves are rejected with clear feedback, and the board is displayed after each turn.

---

##  Training the AI

To train or improve the neural network:

```bash
python train.py
```

The training pipeline supports:

- Self-play game generation
- Policy and/or value network training
- Configurable hyperparameters (learning rate, epochs, batch size)

Training behavior can be adjusted in `trainer.py`.

---

##  Architecture Overview

The project follows strict separation of concerns:

- **Game Engine**: deterministic, testable rule enforcement
- **AI Layer**: PyTorch-based neural networks for evaluation and move selection
- **Training Pipeline**: data generation, loss computation, optimization
- **UI Layer**: human interaction via CLI or optional GUI

This design allows easy experimentation with advanced techniques such as Minimax, MCTS, or AlphaZero-style learning.

---

##  Development & Extensions

Possible extensions include:

- Monte Carlo Tree Search integration
- AlphaZero-style policy/value training
- Web-based or graphical interfaces
- Model benchmarking and evaluation tools

---

##  License

This project is released under the **MIT License**.  
You are free to use, modify, and distribute it for educational or commercial purposes.

---

##  Acknowledgements

Inspired by classic board game AI research and modern PyTorch practices.  
README structure inspired by the *SwissTransitMap* project.
