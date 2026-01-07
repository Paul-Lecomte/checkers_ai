import pytest
from src.engine.board import Board
from src.engine.piece import Piece


def test_start_position_counts():
    b = Board.setup_start()
    assert b.count_pieces('white') == 12
    assert b.count_pieces('black') == 12


def test_clone_independence():
    b = Board.setup_start()
    b2 = b.clone()
    # remove a piece from b2 and ensure b unchanged
    # find any black piece
    pos = None
    for r in range(8):
        for c in range(8):
            p = b2.get_piece((r, c))
            if p and p.color == 'black':
                pos = (r, c)
                break
        if pos:
            break
    assert pos is not None
    b2.set_piece(pos, None)
    assert b.get_piece(pos) is not None

