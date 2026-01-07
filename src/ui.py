from typing import Callable, Optional, List, Set

import pygame
import time

from src.engine.board import Board
from src.engine.move import Move, Pos
from src.engine.rules import generate_legal_moves, apply_move
from src.ai import choose_random_move


class PygameUI:
    """Minimal Pygame UI for the checkers board.

    - board: instance of Board
    - square_size: pixels per square
    - margin: outer margin in pixels
    - on_advance: optional callable Board -> Board called when SPACE is pressed
    - ai_players: optional set of colors that are played by AI (e.g., {'white'})
    """

    def __init__(self, board: Board, square_size: int = 80, margin: int = 20, on_advance: Optional[Callable] = None, ai_players: Optional[Set[str]] = None):
        self.board = board
        self.square_size = square_size
        self.margin = margin
        self.on_advance = on_advance
        self.ai_players = ai_players or set()

        # Interaction state
        self.selected: Optional[Pos] = None
        self.legal_moves: List[Move] = []
        # current player to move: start with black (top side)
        self.current_player = 'black'

    def set_on_advance(self, callback: Callable[[Board], Board]):
        self.on_advance = callback

    def _mouse_to_board(self, mouse_pos: tuple) -> Optional[Pos]:
        mx, my = mouse_pos
        rel_x = mx - self.margin
        rel_y = my - self.margin
        if rel_x < 0 or rel_y < 0:
            return None
        col = rel_x // self.square_size
        row = rel_y // self.square_size
        if 0 <= row < Board.SIZE and 0 <= col < Board.SIZE:
            return (int(row), int(col))
        return None

    def _legal_moves_for(self, pos: Pos) -> List[Move]:
        # generate legal moves for current player and filter those starting from pos
        all_moves = generate_legal_moves(self.board, self.current_player)
        return [m for m in all_moves if m.frm == pos]

    def _is_move_destination(self, pos: Pos) -> bool:
        return any(m.to == pos for m in self.legal_moves)

    def _perform_move(self, move: Move) -> None:
        try:
            new_board = apply_move(self.board, move)
            self.board = new_board
            # swap current player
            self.current_player = 'white' if self.current_player == 'black' else 'black'
        except Exception as e:
            print("Failed to apply move:", e)

    def run(self) -> None:
        """Start the pygame loop and render the board.

        Handles QUIT, ESC/q to quit, SPACE to call on_advance.
        Also handles mouse clicks for selecting and moving pieces.
        """
        try:
            pygame.init()
        except Exception as e:
            raise RuntimeError("Failed to initialize pygame") from e

        size_px = self.margin * 2 + self.square_size * Board.SIZE
        screen = pygame.display.set_mode((size_px, size_px))
        pygame.display.set_caption("Checkers")
        clock = pygame.time.Clock()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        if self.on_advance:
                            try:
                                new_board = self.on_advance(self.board)
                                if isinstance(new_board, Board):
                                    self.board = new_board
                            except Exception as e:
                                print("on_advance callback error:", e)
                        else:
                            print("Advance pressed (no callback set)")
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    board_pos = self._mouse_to_board(event.pos)
                    if board_pos is None:
                        # clicked outside board
                        continue
                    piece = self.board.get_piece(board_pos)
                    if self.selected is None:
                        # try selecting a piece belonging to current player
                        if piece and piece.color == self.current_player:
                            self.selected = board_pos
                            self.legal_moves = self._legal_moves_for(board_pos)
                            print(f"Selected {board_pos}, {len(self.legal_moves)} moves")
                        else:
                            # nothing to select
                            pass
                    else:
                        # if clicking destination of a legal move, perform it
                        if self._is_move_destination(board_pos):
                            move = next(m for m in self.legal_moves if m.to == board_pos)
                            self._perform_move(move)
                            # clear selection after move
                            self.selected = None
                            self.legal_moves = []
                        else:
                            # if clicked another own piece, change selection
                            if piece and piece.color == self.current_player:
                                self.selected = board_pos
                                self.legal_moves = self._legal_moves_for(board_pos)
                                print(f"Reselected {board_pos}, {len(self.legal_moves)} moves")
                            else:
                                # clicked empty or opponent piece: clear selection
                                self.selected = None
                                self.legal_moves = []

            # AI move handling
            if self.current_player in self.ai_players:
                time.sleep(0.5)  # short delay for visibility
                move = choose_random_move(self.board, self.current_player)
                if move:
                    self._perform_move(move)

            # draw
            screen.fill((50, 50, 50))
            self._draw_board(screen)
            self._draw_highlights(screen)
            self._draw_pieces(screen)
            pygame.display.flip()
            clock.tick(30)

        pygame.quit()

    def _draw_board(self, surface: pygame.Surface) -> None:
        light = (240, 217, 181)
        dark = (181, 136, 99)
        for r in range(Board.SIZE):
            for c in range(Board.SIZE):
                x = self.margin + c * self.square_size
                y = self.margin + r * self.square_size
                color = light if (r + c) % 2 == 0 else dark
                pygame.draw.rect(surface, color, (x, y, self.square_size, self.square_size))

    def _draw_highlights(self, surface: pygame.Surface) -> None:
        # draw selection and legal move hints
        if self.selected is None:
            return
        # translucent overlay surface
        overlay = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 0))

        # highlight selected square (blueish)
        r, c = self.selected
        x = self.margin + c * self.square_size
        y = self.margin + r * self.square_size
        # draw border rectangle
        pygame.draw.rect(surface, (30, 144, 255), (x, y, self.square_size, self.square_size), width=4)

        # draw possible destination markers
        for m in self.legal_moves:
            tr, tc = m.to
            cx = self.margin + tc * self.square_size + self.square_size // 2
            cy = self.margin + tr * self.square_size + self.square_size // 2
            pygame.draw.circle(surface, (34, 139, 34), (cx, cy), max(6, self.square_size // 8))

    def _draw_pieces(self, surface: pygame.Surface) -> None:
        for r in range(Board.SIZE):
            for c in range(Board.SIZE):
                p = self.board.get_piece((r, c))
                if p is None:
                    continue
                x = self.margin + c * self.square_size
                y = self.margin + r * self.square_size
                cx = x + self.square_size // 2
                cy = y + self.square_size // 2
                radius = int(self.square_size * 0.4)

                # border + fill for a nicer look
                if p.color == 'black':
                    border_color = (20, 20, 20)
                    fill_color = (40, 40, 40)
                else:
                    border_color = (230, 230, 230)
                    fill_color = (255, 255, 255)

                pygame.draw.circle(surface, border_color, (cx, cy), radius)
                pygame.draw.circle(surface, fill_color, (cx, cy), max(1, radius - 6))

                if getattr(p, 'king', False):
                    crown_color = (212, 175, 55)
                    pygame.draw.circle(surface, crown_color, (cx, cy), radius // 3)
