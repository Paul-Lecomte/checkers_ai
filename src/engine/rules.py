from typing import List, Tuple, Optional
from src.engine.board import Board
from src.engine.move import Move, Pos
from src.engine.piece import Piece


def in_bounds(pos: Pos) -> bool:
    r, c = pos
    return 0 <= r < Board.SIZE and 0 <= c < Board.SIZE


def opponent(color: str) -> str:
    return 'black' if color == 'white' else 'white'


def _directions_for_piece(piece: Piece) -> List[Tuple[int, int]]:
    # For normal piece, forward only; for king both directions
    if piece.king:
        return [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    if piece.color == 'white':
        return [(-1, -1), (-1, 1)]
    else:
        return [(1, -1), (1, 1)]


def _find_jumps_from(board: Board, start: Pos) -> List[Move]:
    """Return all jumping Move sequences starting from start (captures list is ordered).
    Uses cloning per branch to simplify state management.
    """
    piece = board.get_piece(start)
    if piece is None:
        return []

    results: List[Move] = []

    def dfs(b: Board, current_pos: Pos, captured_positions: List[Pos]):
        found_any = False
        p = b.get_piece(current_pos)
        dirs = _directions_for_piece(p)
        for dr, dc in dirs:
            mid = (current_pos[0] + dr, current_pos[1] + dc)
            land = (current_pos[0] + 2 * dr, current_pos[1] + 2 * dc)
            if not in_bounds(mid) or not in_bounds(land):
                continue
            mid_piece = b.get_piece(mid)
            land_piece = b.get_piece(land)
            if mid_piece and mid_piece.color == opponent(p.color) and land_piece is None:
                # perform jump on a clone
                nb = b.clone()
                # remove captured
                nb.set_piece(mid, None)
                # move piece
                nb.set_piece(current_pos, None)
                nb.set_piece(land, Piece(p.color, p.king))
                # continue searching from land
                dfs(nb, land, captured_positions + [mid])
                found_any = True
        if not found_any and captured_positions:
            # no further jumps, record move ending here
            results.append(Move(start, current_pos, captures=captured_positions))

    dfs(board.clone(), start, [])
    return results


def _find_simple_moves(board: Board, start: Pos) -> List[Move]:
    piece = board.get_piece(start)
    if piece is None:
        return []
    moves: List[Move] = []
    for dr, dc in _directions_for_piece(piece):
        to = (start[0] + dr, start[1] + dc)
        if in_bounds(to) and board.get_piece(to) is None:
            moves.append(Move(start, to, captures=[]))
    return moves


def generate_legal_moves(board: Board, color: str) -> List[Move]:
    """Generate all legal moves for player `color`.
    Captures are mandatory: if any capture exists, only capture moves are returned.
    """
    all_captures: List[Move] = []
    all_simples: List[Move] = []
    for r in range(Board.SIZE):
        for c in range(Board.SIZE):
            p = board.get_piece((r, c))
            if p and p.color == color:
                caps = _find_jumps_from(board, (r, c))
                if caps:
                    all_captures.extend(caps)
                else:
                    all_simples.extend(_find_simple_moves(board, (r, c)))
    return all_captures if all_captures else all_simples


def apply_move(board: Board, move: Move) -> Board:
    nb = board.clone()
    piece = nb.get_piece(move.frm)
    if piece is None:
        raise ValueError("No piece at move.frm")
    # move piece
    nb.set_piece(move.frm, None)
    nb.set_piece(move.to, Piece(piece.color, piece.king))
    # remove captures
    if move.captures:
        for pos in move.captures:
            nb.set_piece(pos, None)
    # handle promotion
    r_to, _ = move.to
    if piece.color == 'white' and r_to == 0:
        nb.get_piece(move.to).promote()
    if piece.color == 'black' and r_to == Board.SIZE - 1:
        nb.get_piece(move.to).promote()
    return nb


def has_any_moves(board: Board, color: str) -> bool:
    """Return True if player `color` has any legal move."""
    moves = generate_legal_moves(board, color)
    return bool(moves)


def count_pieces(board: Board, color: str) -> int:
    """Count pieces of a given color on the board."""
    return board.count_pieces(color)


def determine_winner(board: Board, current_player: str = None) -> Optional[str]:
    """Determine the winner of the game if any.

    Returns:
      - 'white' or 'black' if that player has won,
      - 'draw' if neither player has moves/pieces,
      - None if the game is not finished.

    Logic used:
      - If a player has zero pieces -> opponent wins.
      - If a player has zero legal moves -> opponent wins.
      - If both players have no moves -> 'draw'.

    The optional `current_player` arg is not strictly necessary for the result here,
    but can be useful for checking draw conditions.
    """
    white_pieces = count_pieces(board, 'white')
    black_pieces = count_pieces(board, 'black')

    white_has_moves = has_any_moves(board, 'white')
    black_has_moves = has_any_moves(board, 'black')

    # If a player has 0 pieces -> opponent wins
    if white_pieces == 0 and black_pieces == 0:
        return 'draw'
    if white_pieces == 0:
        return 'black'
    if black_pieces == 0:
        return 'white'

    # If a player has no legal moves -> opponent wins
    if not white_has_moves and not black_has_moves:
        return 'draw'
    if not white_has_moves:
        return 'black'
    if not black_has_moves:
        return 'white'

    # No winner yet
    return None
