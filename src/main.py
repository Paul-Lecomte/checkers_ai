"""Entry point for the checkers game
Run: python -m src.main [--ai-white] [--ai-black] [--ai-both]
"""

import sys
from typing import Set

from src.engine.board import Board
from src.ui import PygameUI


def parse_ai_args(argv) -> Set[str]:
    ai: Set[str] = set()
    if '--ai-both' in argv:
        ai.add('white')
        ai.add('black')
    else:
        if '--ai-white' in argv:
            ai.add('white')
        if '--ai-black' in argv:
            ai.add('black')
    return ai


def main(argv=None):
    argv = argv or sys.argv[1:]
    ai_players = parse_ai_args(argv)

    board = Board.setup_start()
    ui = PygameUI(board, ai_players=ai_players)
    try:
        ui.run()
    except Exception as e:
        print("Error running UI:", e)
        print("If this is an ImportError for pygame, install it with: python -m pip install pygame")
        raise


if __name__ == "__main__":
    main()
