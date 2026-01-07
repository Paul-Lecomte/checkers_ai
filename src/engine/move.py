from dataclasses import dataclass
from typing import List, Tuple

Pos = Tuple[int, int]  # (row, col)


@dataclass
class Move:
    """Represents a move from one square to another.
    captures: list of positions captured during the move (for jumps)
    """
    frm: Pos
    to: Pos
    captures: List[Pos] = None

    def is_capture(self):
        return bool(self.captures)

    def __repr__(self):
        return f"Move({self.frm} -> {self.to}, captures={self.captures})"

