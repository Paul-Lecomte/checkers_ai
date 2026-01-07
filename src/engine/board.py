from copy import deepcopy
from typing import List, Optional, Tuple
from src.engine.piece import Piece
from src.engine.move import Move, Pos


BoardArray = List[List[Optional[Piece]]]


class Board:
    SIZE = 8

    def __init__(self):
        # initialize empty board
        self.grid: BoardArray = [[None for _ in range(self.SIZE)] for _ in range(self.SIZE)]

    @classmethod
    def setup_start(cls):
        b = cls()
        # Place black pieces on rows 0..2 on dark squares
        for r in range(3):
            for c in range(cls.SIZE):
                if (r + c) % 2 == 1:
                    b.grid[r][c] = Piece('black')
        # Place white pieces on rows 5..7
        for r in range(5, 8):
            for c in range(cls.SIZE):
                if (r + c) % 2 == 1:
                    b.grid[r][c] = Piece('white')
        return b

    def get_piece(self, pos: Pos) -> Optional[Piece]:
        r, c = pos
        if 0 <= r < self.SIZE and 0 <= c < self.SIZE:
            return self.grid[r][c]
        return None

    def set_piece(self, pos: Pos, piece: Optional[Piece]):
        r, c = pos
        if 0 <= r < self.SIZE and 0 <= c < self.SIZE:
            self.grid[r][c] = piece
        else:
            raise IndexError("Position out of board")

    def clone(self) -> 'Board':
        newb = Board()
        newb.grid = deepcopy(self.grid)
        return newb

    def count_pieces(self, color: str) -> int:
        return sum(1 for r in range(self.SIZE) for c in range(self.SIZE) if self.grid[r][c] and self.grid[r][c].color == color)

    def __repr__(self):
        rows = []
        for r in range(self.SIZE):
            row = []
            for c in range(self.SIZE):
                p = self.grid[r][c]
                if p is None:
                    row.append('.')
                else:
                    row.append('b' if p.color == 'black' else 'w')
            rows.append(''.join(row))
        return '\n'.join(rows)

