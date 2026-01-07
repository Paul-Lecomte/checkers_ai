from typing import Callable, Optional, List, Set

import pygame
import time

from src.engine.board import Board
from src.engine.move import Move, Pos
from src.engine.rules import generate_legal_moves, apply_move, determine_winner, count_pieces
from src.ai import choose_random_move

try:
    # optional import; score_moves is defined only if nn module is available
    from src.nn import score_moves
except Exception:
    score_moves = None


class PygameUI:
    """Pygame UI with animated moves and an optional sidebar showing NN info.

    - ai_players: set of colors controlled by AI
    - model: optional ML model used by `score_moves` (passed via score_moves closure)
    """

    def __init__(self, board: Board, square_size: int = 80, margin: int = 20, on_advance: Optional[Callable] = None, ai_players: Optional[Set[str]] = None, model: Optional[object] = None, sidebar_width: int = 240, anim_seconds: float = 0.35):
        self.board = board
        self.square_size = square_size
        self.margin = margin
        self.on_advance = on_advance
        self.ai_players = ai_players or set()
        self.model = model
        self.sidebar_width = sidebar_width
        self.anim_seconds = anim_seconds

        # Interaction state
        self.selected: Optional[Pos] = None
        self.legal_moves: List[Move] = []
        self.current_player = 'black'

        # Animation state
        self.animating = False
        self.anim_move: Optional[Move] = None
        self.anim_start_px = (0, 0)
        self.anim_end_px = (0, 0)
        self.anim_elapsed = 0.0
        self.anim_piece_color = None
        self.board_before_anim: Optional[Board] = None

    def set_on_advance(self, callback: Callable[[Board], Board]):
        self.on_advance = callback

    def _mouse_to_board(self, mouse_pos: tuple) -> Optional[Pos]:
        mx, my = mouse_pos
        board_left = self.margin
        board_top = self.margin
        rel_x = mx - board_left
        rel_y = my - board_top
        if rel_x < 0 or rel_y < 0:
            return None
        col = rel_x // self.square_size
        row = rel_y // self.square_size
        if 0 <= row < Board.SIZE and 0 <= col < Board.SIZE:
            return (int(row), int(col))
        return None

    def _legal_moves_for(self, pos: Pos) -> List[Move]:
        all_moves = generate_legal_moves(self.board, self.current_player)
        return [m for m in all_moves if m.frm == pos]

    def _is_move_destination(self, pos: Pos) -> bool:
        return any(m.to == pos for m in self.legal_moves)

    def _start_move_animation(self, move: Move) -> None:
        """Prepare animation: store move, piece color, compute pixel coordinates, and mark animating.
        The actual board update will be applied when animation completes.
        """
        if self.animating:
            return
        piece = self.board.get_piece(move.frm)
        if piece is None:
            return
        self.animating = True
        self.anim_move = move
        self.anim_elapsed = 0.0
        self.anim_piece_color = piece.color
        self.board_before_anim = self.board
        # compute start/end centers in pixels
        sx = self.margin + move.frm[1] * self.square_size + self.square_size // 2
        sy = self.margin + move.frm[0] * self.square_size + self.square_size // 2
        ex = self.margin + move.to[1] * self.square_size + self.square_size // 2
        ey = self.margin + move.to[0] * self.square_size + self.square_size // 2
        self.anim_start_px = (sx, sy)
        self.anim_end_px = (ex, ey)
        # visually remove piece from source during animation by using a clone for drawing
        # we won't modify self.board until the animation completes

    def _finish_move_animation(self) -> None:
        """Apply the move to the logical board and finish animation state."""
        if not self.animating or self.anim_move is None:
            return
        try:
            # apply move to the current logical board (board_before_anim)
            self.board = apply_move(self.board, self.anim_move)
        except Exception as e:
            print("Error applying move after animation:", e)
        # swap player
        self.current_player = 'white' if self.current_player == 'black' else 'black'
        # clear animation
        self.animating = False
        self.anim_move = None
        self.anim_elapsed = 0.0
        self.anim_piece_color = None
        self.board_before_anim = None
        # clear selection/legals
        self.selected = None
        self.legal_moves = []

    def _perform_move_or_animate(self, move: Move) -> None:
        # start animation and let it finish to apply move
        self._start_move_animation(move)

    def run(self) -> None:
        try:
            pygame.init()
        except Exception as e:
            raise RuntimeError("Failed to initialize pygame") from e

        board_px = self.margin * 2 + self.square_size * Board.SIZE
        total_width = board_px + self.sidebar_width
        total_height = board_px

        screen = pygame.display.set_mode((total_width, total_height))
        pygame.display.set_caption("Checkers")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 20)
        large_font = pygame.font.SysFont(None, 26)

        running = True
        while running:
            dt = clock.tick(60) / 1000.0  # seconds since last frame
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
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not self.animating:
                    board_pos = self._mouse_to_board(event.pos)
                    if board_pos is None:
                        continue
                    piece = self.board.get_piece(board_pos)
                    if self.selected is None:
                        if piece and piece.color == self.current_player:
                            self.selected = board_pos
                            self.legal_moves = self._legal_moves_for(board_pos)
                    else:
                        if self._is_move_destination(board_pos):
                            move = next(m for m in self.legal_moves if m.to == board_pos)
                            self._perform_move_or_animate(move)
                        else:
                            if piece and piece.color == self.current_player:
                                self.selected = board_pos
                                self.legal_moves = self._legal_moves_for(board_pos)
                            else:
                                self.selected = None
                                self.legal_moves = []

            # AI turn: if it's AI's turn and not animating, play
            if not self.animating and self.current_player in self.ai_players:
                # small pause to make it visible
                time.sleep(0.2)
                move = choose_random_move(self.board, self.current_player)
                if move:
                    self._perform_move_or_animate(move)

            # update animation
            if self.animating and self.anim_move is not None:
                self.anim_elapsed += dt
                if self.anim_elapsed >= self.anim_seconds:
                    # finish
                    self._finish_move_animation()

            # draw
            screen.fill((40, 40, 40))
            # board area
            board_surface = screen.subsurface((0, 0, board_px, total_height))
            self._draw_board(board_surface)
            # during animation, draw from board_before_anim with the moving piece drawn separately
            if self.animating and self.board_before_anim is not None:
                self._draw_pieces(board_surface, self.board_before_anim, hide_from=self.anim_move.frm)
                # draw moving piece at interpolated position
                t = min(1.0, (self.anim_elapsed / max(1e-6, self.anim_seconds)))
                sx, sy = self.anim_start_px
                ex, ey = self.anim_end_px
                cx = int(sx + (ex - sx) * t)
                cy = int(sy + (ey - sy) * t)
                self._draw_piece_at(board_surface, (cx, cy), self.anim_piece_color)
            else:
                self._draw_pieces(board_surface, self.board)

            # sidebar
            sidebar_rect = pygame.Rect(board_px, 0, self.sidebar_width, total_height)
            pygame.draw.rect(screen, (60, 60, 60), sidebar_rect)
            self._draw_sidebar(screen, sidebar_rect, font, large_font)

            pygame.display.flip()

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

    def _draw_pieces(self, surface: pygame.Surface, board: Board, hide_from: Optional[Pos] = None) -> None:
        for r in range(Board.SIZE):
            for c in range(Board.SIZE):
                if hide_from is not None and hide_from == (r, c):
                    continue
                p = board.get_piece((r, c))
                if p is None:
                    continue
                x = self.margin + c * self.square_size
                y = self.margin + r * self.square_size
                cx = x + self.square_size // 2
                cy = y + self.square_size // 2
                self._draw_piece_at(surface, (cx, cy), p.color, king=getattr(p, 'king', False))

    def _draw_piece_at(self, surface: pygame.Surface, center: tuple, color: str, king: bool = False) -> None:
        cx, cy = center
        radius = int(self.square_size * 0.4)
        if color == 'black':
            border_color = (20, 20, 20)
            fill_color = (40, 40, 40)
        else:
            border_color = (230, 230, 230)
            fill_color = (255, 255, 255)
        pygame.draw.circle(surface, border_color, (cx, cy), radius)
        pygame.draw.circle(surface, fill_color, (cx, cy), max(1, radius - 6))
        if king:
            crown_color = (212, 175, 55)
            pygame.draw.circle(surface, crown_color, (cx, cy), radius // 3)

    def _draw_highlights(self, surface: pygame.Surface) -> None:
        if self.selected is None:
            return
        r, c = self.selected
        x = self.margin + c * self.square_size
        y = self.margin + r * self.square_size
        pygame.draw.rect(surface, (30, 144, 255), (x, y, self.square_size, self.square_size), width=4)
        for m in self.legal_moves:
            tr, tc = m.to
            cx = self.margin + tc * self.square_size + self.square_size // 2
            cy = self.margin + tr * self.square_size + self.square_size // 2
            pygame.draw.circle(surface, (34, 139, 34), (cx, cy), max(6, self.square_size // 8))

    def _draw_sidebar(self, screen: pygame.Surface, rect: pygame.Rect, font: pygame.font.Font, large_font: pygame.font.Font) -> None:
        x0 = rect.x + 8
        y = 8
        # title
        title = large_font.render("Game Info", True, (255, 255, 255))
        screen.blit(title, (x0, y))
        y += 32
        # current player
        cp_text = font.render(f"Current: {self.current_player}", True, (255, 255, 255))
        screen.blit(cp_text, (x0, y))
        y += 24
        # counts
        wcount = count_pieces(self.board, 'white')
        bcount = count_pieces(self.board, 'black')
        wc = font.render(f"White pieces: {wcount}", True, (255, 255, 255))
        bc = font.render(f"Black pieces: {bcount}", True, (255, 255, 255))
        screen.blit(wc, (x0, y)); y += 20
        screen.blit(bc, (x0, y)); y += 28

        # winner check
        winner = determine_winner(self.board)
        if winner is not None:
            win_text = large_font.render(f"Result: {winner}", True, (255, 215, 0))
            screen.blit(win_text, (x0, y)); y += 32

        # model info
        if self.model is not None:
            try:
                # try to show simple model info
                params = sum(p.numel() for p in self.model.parameters())
                mi = font.render(f"Model params: {params}", True, (200, 200, 255))
                screen.blit(mi, (x0, y)); y += 20
            except Exception:
                pass

        # If a piece is selected, show legal moves and model scores (if available)
        if self.selected is not None:
            sel_text = font.render(f"Selected: {self.selected}", True, (255, 255, 255))
            screen.blit(sel_text, (x0, y)); y += 20
            # list legal moves
            for m in self.legal_moves:
                mv_text = font.render(f"{m.frm}->{m.to} cap:{len(m.captures) if m.captures else 0}", True, (220, 220, 220))
                screen.blit(mv_text, (x0, y)); y += 18
            y += 6
            if self.model is not None and score_moves is not None and len(self.legal_moves) > 0:
                # compute scores for legal moves using the model
                try:
                    scored = score_moves(self.board, self.legal_moves, self.model)
                    # sort by score desc
                    scored.sort(key=lambda t: t[1], reverse=True)
                    for m, s in scored:
                        s_text = font.render(f"{m.frm}->{m.to}: {s:.3f}", True, (200, 255, 200))
                        screen.blit(s_text, (x0, y)); y += 18
                except Exception as e:
                    err = font.render(f"Model eval error", True, (255, 100, 100))
                    screen.blit(err, (x0, y)); y += 18

        # draw small legend
        y = rect.y + rect.h - 80
        l1 = font.render("Esc/q: Quit    Space: Advance", True, (180, 180, 180))
        screen.blit(l1, (x0, y))
