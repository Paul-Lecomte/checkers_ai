from typing import Callable, Optional, List, Set, Tuple
import threading

import pygame
import time

from src.engine.board import Board
from src.engine.move import Move, Pos
from src.engine.rules import generate_legal_moves, apply_move, determine_winner, count_pieces
from src.ai import choose_random_move

# optional nn utilities
try:
    from src.nn import score_moves, train_random_positions, CheckersNet, load_model
except Exception:
    score_moves = None
    train_random_positions = None
    CheckersNet = None
    load_model = None


class PygameUI:
    """Pygame UI with animated moves, sidebar and model controls.

    Buttons added in sidebar:
      - Reset: reset the board
      - Toggle Model AI: cycle which colors the model controls (None/White/Black/Both)
      - Train Model: start a background training run (uses train_random_positions)
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

        # model control state: which colors are controlled by the neural network
        self.model_control: Set[str] = set()
        # human-visible model status message shown in sidebar
        self.model_message: str = ""

        # training state
        self.training = False
        self.training_status = ""
        self._training_thread: Optional[threading.Thread] = None
        self._model_lock = threading.Lock()

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
                    mx, my = event.pos
                    board_left = 0
                    board_top = 0
                    board_px_local = board_px
                    # check if click is in sidebar
                    if mx >= board_px_local:
                        # handle sidebar button clicks
                        rel_x = mx - board_px_local
                        rel_y = my
                        self._handle_sidebar_click(rel_x, rel_y, font)
                        continue
                    # otherwise handle board clicks
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
            # Consider both simple ai_players and model_control (model-controlled colors)
            if not self.animating and (self.current_player in self.ai_players or self.current_player in self.model_control):
                 # small pause to make it visible
                 time.sleep(0.2)
                 move = None
                 # prefer model-controlled move if configured
                 if self.current_player in self.model_control and self.model is not None and score_moves is not None:
                     try:
                         scored = score_moves(self.board, generate_legal_moves(self.board, self.current_player), self.model)
                         if scored:
                             # choose best score
                             scored.sort(key=lambda t: t[1], reverse=True)
                             move = scored[0][0]
                     except Exception as e:
                         print("Model move error, falling back to random:", e)
                         move = None
                 if move is None:
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

    def _handle_sidebar_click(self, rel_x: int, rel_y: int, font: pygame.font.Font) -> None:
        """Handle clicks in the sidebar area. Coordinates are relative to the sidebar origin."""
        click_point = (rel_x, rel_y)
        # Use cached button rects (relative to sidebar) if available
        btns = getattr(self, '_sidebar_buttons', None)
        if btns:
            if btns.get('reset') and btns['reset'].collidepoint(click_point):
                self._do_reset()
                return
            if btns.get('toggle') and btns['toggle'].collidepoint(click_point):
                self._cycle_model_control()
                return
            if btns.get('train') and btns['train'].collidepoint(click_point):
                self._start_training_thread()
                return
        # fallback: ignore click if no cached buttons
        return

    def _do_reset(self) -> None:
        # reset the logical board and UI state
        self.board = Board.setup_start()
        self.selected = None
        self.legal_moves = []
        self.animating = False
        self.anim_move = None
        self.anim_elapsed = 0.0
        self.current_player = 'black'
        self.training_status = ""
        print("Board reset")

    def _cycle_model_control(self) -> None:
        # cycle through: none -> white -> black -> both -> none
        if not self.model_control:
            self.model_control = {'white'}
        elif self.model_control == {'white'}:
            self.model_control = {'black'}
        elif self.model_control == {'black'}:
            self.model_control = {'white', 'black'}
        else:
            self.model_control = set()
        # ensure a model exists if the user assigned model control
        if self.model_control and self.model is None:
            # try to (re)import NN utilities dynamically in case they were not available at module load
            try:
                from src import nn as _nnmod
                # expose function references at module scope
                globals()['score_moves'] = getattr(_nnmod, 'score_moves', None)
                globals()['train_random_positions'] = getattr(_nnmod, 'train_random_positions', None)
                globals()['CheckersNet'] = getattr(_nnmod, 'CheckersNet', None)
                globals()['load_model'] = getattr(_nnmod, 'load_model', None)
            except Exception:
                # leave existing values as-is
                pass

            if 'CheckersNet' in globals() and globals()['CheckersNet'] is not None:
                with self._model_lock:
                    if self.model is None:
                        try:
                            self.model = globals()['CheckersNet']()
                            self.model_message = "Model instantiated"
                            print("Instantiated new CheckersNet for model control")
                        except Exception as e:
                            self.model_message = f"Failed to instantiate model: {e}"
                            print("Failed to instantiate model:", e)
            else:
                self.model_message = "Model unavailable: install PyTorch or use Train/--model"
                print("Model controls set, but PyTorch/NN utilities not available. Use Train or --model to provide a model.")
        print("Model now controls:", self.model_control)

    def _start_training_thread(self) -> None:
        if train_random_positions is None or CheckersNet is None:
            print("Training not available: PyTorch or training utilities missing.")
            return
        if self.training:
            print("Training already running")
            return
        # ensure a model exists
        with self._model_lock:
            if self.model is None:
                self.model = CheckersNet()
        # start thread
        self.training = True
        self.training_status = "Starting..."

        def _train():
            def progress_cb(epoch, loss):
                self.training_status = f"Epoch {epoch}: loss={loss:.4f}"
            try:
                train_random_positions(self.model, epochs=5, n_positions=64, progress_callback=progress_cb)
                self.training_status = "Training done"
            except Exception as e:
                self.training_status = f"Training error: {e}"
            finally:
                self.training = False

        t = threading.Thread(target=_train, daemon=True)
        t.start()
        self._training_thread = t

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
         model_label = "None"
         if self.model is not None:
             try:
                 params = sum(p.numel() for p in self.model.parameters())
                 model_label = f"params={params}"
             except Exception:
                 model_label = "model"
         ml = font.render(f"Model: {model_label}", True, (200, 200, 255))
         screen.blit(ml, (x0, y)); y += 20

         # buttons
         btn_w = self.sidebar_width - 16
         btn_h = 28
         # Reset
         reset_rect = pygame.Rect(x0, y, btn_w, btn_h)
         pygame.draw.rect(screen, (100, 80, 80), reset_rect)
         reset_text = font.render("Reset", True, (255, 255, 255))
         screen.blit(reset_text, (x0 + 8, y + 6))
         y += btn_h + 8
         # Toggle Model AI
         toggle_rect = pygame.Rect(x0, y, btn_w, btn_h)
         pygame.draw.rect(screen, (80, 100, 80), toggle_rect)
         mc_label = ",".join(sorted(self.model_control)) if self.model_control else "None"
         toggle_text = font.render(f"Model controls: {mc_label}", True, (255, 255, 255))
         screen.blit(toggle_text, (x0 + 8, y + 6))
         y += btn_h + 8
         # Train Model
         train_rect = pygame.Rect(x0, y, btn_w, btn_h)
         train_color = (80, 80, 120) if not self.training else (120, 120, 80)
         pygame.draw.rect(screen, train_color, train_rect)
         train_text = font.render("Train Model", True, (255, 255, 255))
         screen.blit(train_text, (x0 + 8, y + 6))
         y += btn_h + 12

         # cache button rects in sidebar-local coordinates for click handling
         try:
             rel_reset = pygame.Rect(reset_rect.x - rect.x, reset_rect.y - rect.y, reset_rect.w, reset_rect.h)
             rel_toggle = pygame.Rect(toggle_rect.x - rect.x, toggle_rect.y - rect.y, toggle_rect.w, toggle_rect.h)
             rel_train = pygame.Rect(train_rect.x - rect.x, train_rect.y - rect.y, train_rect.w, train_rect.h)
             self._sidebar_buttons = {'reset': rel_reset, 'toggle': rel_toggle, 'train': rel_train}
         except Exception:
             self._sidebar_buttons = None
