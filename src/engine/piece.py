from dataclasses import dataclass


@dataclass
class Piece:
    """Represents a checkers piece.
    color: 'white' or 'black'
    king: whether the piece is promoted
    """
    color: str
    king: bool = False

    def promote(self):
        self.king = True

    def __repr__(self):
        return f"Piece(color={self.color!r}, king={self.king})"

