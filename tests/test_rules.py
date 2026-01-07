from src.engine.board import Board
from src.engine.piece import Piece
from src.engine.rules import generate_legal_moves, apply_move
from src.engine.move import Move


def test_simple_move():
    b = Board()
    # place a white piece at (5,1) with empty (4,0)
    b.set_piece((5, 1), Piece('white'))
    moves = generate_legal_moves(b, 'white')
    assert any(m.frm == (5,1) and m.to == (4,0) for m in moves)


def test_single_capture():
    b = Board()
    b.set_piece((5, 1), Piece('white'))
    b.set_piece((4, 2), Piece('black'))
    # landing square (3,3) empty
    moves = generate_legal_moves(b, 'white')
    caps = [m for m in moves if m.is_capture()]
    assert len(caps) == 1
    m = caps[0]
    assert m.frm == (5,1) and m.to == (3,3) and m.captures == [(4,2)]


def test_multi_jump():
    b = Board()
    b.set_piece((5, 1), Piece('white'))
    b.set_piece((4, 2), Piece('black'))
    b.set_piece((2, 4), Piece('black'))
    # ensure landing (3,3) after first jump, then (1,5)
    moves = generate_legal_moves(b, 'white')
    caps = [m for m in moves if m.is_capture()]
    # expect one multi-jump move
    assert any(len(m.captures) == 2 for m in caps)

