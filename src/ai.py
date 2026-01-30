# AI module for choosing moves in a chess game.
from typing import Optional
import random

from src.engine.move import Move
from src.engine.rules import generate_legal_moves
from src.engine.board import Board


def choose_random_move(board: Board, color: str) -> Optional[Move]:
    """Return a randomly chosen legal Move for color, or None if no legal moves."""
    moves = generate_legal_moves(board, color)
    if not moves:
        return None
    return random.choice(moves)

