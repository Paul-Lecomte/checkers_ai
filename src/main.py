"""Entry point for the checkers game
Run: python -m src.main
"""

from src.engine.board import Board
from src.ui import PygameUI


def main():
    board = Board.setup_start()
    ui = PygameUI(board)
    try:
        ui.run()
    except Exception as e:
        print("Error running UI:", e)
        print("If this is an ImportError for pygame, install it with: python -m pip install pygame")
        raise


if __name__ == "__main__":
    main()
