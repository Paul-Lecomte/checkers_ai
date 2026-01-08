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
      - NN: White / NN: Black: toggle model control per color
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

        # Theme/colors
        self.theme = {
            'bg': (36, 36, 38),
            'board_border': (50, 50, 52),
            'light': (235, 219, 188),
            'dark': (181, 136, 99),
            'light2': (245, 228, 199),
            'dark2': (171, 126, 89),
            'highlight': (30, 144, 255),
            'move_dot': (34, 139, 34),
            'last_move_from': (255, 215, 0, 70),
            'last_move_to': (60, 179, 113, 80),
            'sidebar': (58, 58, 60),
            'text': (240, 240, 240),
            'muted': (200, 200, 200),
            'danger': (180, 60, 60),
            'ok': (60, 160, 80),
            'info': (70, 90, 160),
        }

        # Interaction state
        self.selected: Optional[Pos] = None
        self.legal_moves: List[Move] = []
        self.current_player = 'black'
        self.last_move: Optional[Move] = None

        # Mouse/hover
        self._mouse_pos: Tuple[int, int] = (0, 0)

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
        # model activity visualization
        self.model_thinking: bool = False
        self.last_model_scores: Optional[List[tuple]] = None  # list of (move, score)
        self.model_think_start: float = 0.0

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
        # store last move for highlight
        self.last_move = self.anim_move
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
                elif event.type == pygame.MOUSEMOTION:
                    self._mouse_pos = event.pos
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
                         # indicate thinking
                         self.model_thinking = True
                         self.model_think_start = time.time()
                         moves = generate_legal_moves(self.board, self.current_player)
                         scored = score_moves(self.board, moves, self.model)
                         # store for visualization
                         self.last_model_scores = sorted(scored, key=lambda t: t[1], reverse=True)
                         if self.last_model_scores:
                             # choose best score
                             move = self.last_model_scores[0][0]
                         # short visible pause so user sees the thinking/visual
                         time.sleep(0.35)
                     except Exception as e:
                         print("Model move error, falling back to random:", e)
                         move = None
                     finally:
                         self.model_thinking = False
                 if move is None:
                     move = choose_random_move(self.board, self.current_player)
                 if move:
                     # highlight chosen move in last_model_scores if present
                     try:
                         if self.last_model_scores is not None:
                             # move may not be in list if random fallback; ensure it's first
                             if not any(m is move or (m.frm == move.frm and m.to == move.to) for m, _ in self.last_model_scores):
                                 # prepend move with a neutral score
                                 self.last_model_scores = [(move, 0.0)] + (self.last_model_scores or [])
                     except Exception:
                         pass
                     self._perform_move_or_animate(move)

            # update animation
            if self.animating and self.anim_move is not None:
                self.anim_elapsed += dt
                if self.anim_elapsed >= self.anim_seconds:
                    # finish
                    self._finish_move_animation()

            # draw
            screen.fill(self.theme['bg'])
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

            # overlays: last move and highlights
            self._draw_last_move(board_surface)
            self._draw_highlights(board_surface)
            self._draw_hover(board_surface)

            # sidebar
            sidebar_rect = pygame.Rect(board_px, 0, self.sidebar_width, total_height)
            pygame.draw.rect(screen, self.theme['sidebar'], sidebar_rect)
            self._draw_sidebar(screen, sidebar_rect, font, large_font)

            pygame.display.flip()

        pygame.quit()

    def _draw_board(self, surface: pygame.Surface) -> None:
        # draw outer border
        pygame.draw.rect(surface, self.theme['board_border'], (self.margin - 6, self.margin - 6, self.square_size * Board.SIZE + 12, self.square_size * Board.SIZE + 12), border_radius=6)
        # wood-like alternating tones with subtle gradient per square
        for r in range(Board.SIZE):
            for c in range(Board.SIZE):
                x = self.margin + c * self.square_size
                y = self.margin + r * self.square_size
                base = self.theme['light'] if (r + c) % 2 == 0 else self.theme['dark']
                alt = self.theme['light2'] if (r + c) % 2 == 0 else self.theme['dark2']
                # gradient vertical
                rect = pygame.Rect(x, y, self.square_size, self.square_size)
                self._fill_vertical_gradient(surface, rect, alt, base)
        # grid outline
        for i in range(Board.SIZE + 1):
            x = self.margin + i * self.square_size
            y = self.margin + i * self.square_size
            pygame.draw.line(surface, (0, 0, 0, 40), (self.margin, y), (self.margin + self.square_size * Board.SIZE, y))
            pygame.draw.line(surface, (0, 0, 0, 40), (x, self.margin), (x, self.margin + self.square_size * Board.SIZE))

    def _fill_vertical_gradient(self, surface: pygame.Surface, rect: pygame.Rect, top_color: Tuple[int, int, int], bottom_color: Tuple[int, int, int]):
        # simple vertical gradient fill
        h = rect.height
        if h <= 1:
            pygame.draw.rect(surface, top_color, rect)
            return
        r1, g1, b1 = top_color
        r2, g2, b2 = bottom_color
        for i in range(h):
            t = i / (h - 1)
            r = int(r1 + (r2 - r1) * t)
            g = int(g1 + (g2 - g1) * t)
            b = int(b1 + (b2 - b1) * t)
            pygame.draw.line(surface, (r, g, b), (rect.x, rect.y + i), (rect.x + rect.width, rect.y + i))

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
        radius = int(self.square_size * 0.42)
        # shadow
        shadow_offset = (int(self.square_size * 0.06), int(self.square_size * 0.06))
        shadow_color = (0, 0, 0, 80)
        self._draw_circle(surface, (cx + shadow_offset[0], cy + shadow_offset[1]), radius, shadow_color)
        # base colors
        if color == 'black':
            border_color = (25, 25, 25)
            fill_color = (55, 55, 58)
            rim_color = (95, 95, 100)
            highlight = (255, 255, 255, 50)
        else:
            border_color = (210, 210, 210)
            fill_color = (250, 250, 250)
            rim_color = (220, 220, 220)
            highlight = (255, 255, 255, 90)
        # piece body (border + fill)
        pygame.draw.circle(surface, border_color, (cx, cy), radius)
        pygame.draw.circle(surface, fill_color, (cx, cy), max(1, radius - 5))
        # inner rim
        pygame.draw.circle(surface, rim_color, (cx, cy), max(1, radius - 10), width=2)
        # specular highlight
        self._draw_circle(surface, (cx - radius // 3, cy - radius // 3), radius // 2, highlight)
        # king marker
        if king:
            crown_color = (212, 175, 55)
            pygame.draw.circle(surface, crown_color, (cx, cy), radius // 3)
            pygame.draw.circle(surface, (180, 140, 30), (cx, cy), radius // 3, width=2)

    def _draw_circle(self, surface: pygame.Surface, center: Tuple[int, int], radius: int, color: Tuple[int, int, int, int]):
        # draw a circle with optional alpha by using a temporary surface
        diameter = radius * 2
        temp = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
        pygame.draw.circle(temp, color, (radius, radius), radius)
        surface.blit(temp, (center[0] - radius, center[1] - radius))

    def _draw_highlights(self, surface: pygame.Surface) -> None:
        # selection square
        if self.selected is not None:
            r, c = self.selected
            x = self.margin + c * self.square_size
            y = self.margin + r * self.square_size
            pygame.draw.rect(surface, self.theme['highlight'], (x + 2, y + 2, self.square_size - 4, self.square_size - 4), width=3, border_radius=6)
            # legal move dots
            for m in self.legal_moves:
                tr, tc = m.to
                cx = self.margin + tc * self.square_size + self.square_size // 2
                cy = self.margin + tr * self.square_size + self.square_size // 2
                dot_radius = max(6, self.square_size // 10)
                pygame.draw.circle(surface, self.theme['move_dot'], (cx, cy), dot_radius)

    def _draw_last_move(self, surface: pygame.Surface) -> None:
        if not self.last_move:
            return
        # semi-transparent overlays
        def square_rect(pos: Pos) -> pygame.Rect:
            r, c = pos
            x = self.margin + c * self.square_size
            y = self.margin + r * self.square_size
            return pygame.Rect(x, y, self.square_size, self.square_size)
        frm_rect = square_rect(self.last_move.frm)
        to_rect = square_rect(self.last_move.to)
        # from in gold tint, to in green tint
        self._fill_rect_alpha(surface, frm_rect, self.theme['last_move_from'])
        self._fill_rect_alpha(surface, to_rect, self.theme['last_move_to'])

    def _fill_rect_alpha(self, surface: pygame.Surface, rect: pygame.Rect, color: Tuple[int, int, int, int]):
        temp = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        temp.fill(color)
        surface.blit(temp, rect.topleft)

    def _draw_hover(self, surface: pygame.Surface) -> None:
        # highlight hovered dark square subtly
        pos = self._mouse_to_board(self._mouse_pos)
        if pos is None:
            return
        r, c = pos
        x = self.margin + c * self.square_size
        y = self.margin + r * self.square_size
        self._fill_rect_alpha(surface, pygame.Rect(x, y, self.square_size, self.square_size), (255, 255, 255, 25))

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
            if btns.get('nn_white') and btns['nn_white'].collidepoint(click_point):
                self._toggle_model_color('white')
                return
            if btns.get('nn_black') and btns['nn_black'].collidepoint(click_point):
                self._toggle_model_color('black')
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
        self.last_move = None
        self.training_status = ""
        print("Board reset")

    def _ensure_nn_utils(self):
        # try import of nn utils if not already present
        if globals().get('CheckersNet') is None or globals().get('score_moves') is None:
            try:
                from src import nn as _nnmod
                globals()['score_moves'] = getattr(_nnmod, 'score_moves', None)
                globals()['train_random_positions'] = getattr(_nnmod, 'train_random_positions', None)
                globals()['CheckersNet'] = getattr(_nnmod, 'CheckersNet', None)
                globals()['load_model'] = getattr(_nnmod, 'load_model', None)
            except Exception:
                pass

    def _toggle_model_color(self, color: str) -> None:
        # Toggle a single color in model_control and ensure model is ready when enabling
        if color not in {'white', 'black'}:
            return
        if color in self.model_control:
            self.model_control.remove(color)
        else:
            self.model_control.add(color)
            # ensure utilities and model exist
            self._ensure_nn_utils()
            if globals().get('CheckersNet') is not None:
                with self._model_lock:
                    if self.model is None:
                        try:
                            self.model = globals()['CheckersNet']()
                            self.model_message = "Model instantiated"
                        except Exception as e:
                            self.model_message = f"Failed to instantiate model: {e}"
            else:
                self.model_message = "Model unavailable: install PyTorch"
        print("Model now controls:", self.model_control)

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
            self._ensure_nn_utils()
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
         x0 = rect.x + 12
         y = 10
         # title
         title = large_font.render("Game Info", True, self.theme['text'])
         screen.blit(title, (x0, y))
         y += 32
         # current player
         cp_text = font.render(f"Current: {self.current_player}", True, self.theme['text'])
         screen.blit(cp_text, (x0, y))
         y += 24
         # counts
         wcount = count_pieces(self.board, 'white')
         bcount = count_pieces(self.board, 'black')
         wc = font.render(f"White pieces: {wcount}", True, self.theme['text'])
         bc = font.render(f"Black pieces: {bcount}", True, self.theme['text'])
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
         # model message (status)
         if self.model_message:
             mm = font.render(self.model_message, True, (200, 180, 120))
             screen.blit(mm, (x0, y)); y += 18

         # training status
         if self.training_status:
             tstatus = font.render(self.training_status, True, self.theme['muted'])
             screen.blit(tstatus, (x0, y))
             y += 24

         # model activity visualization
         if self.model_thinking:
             think_text = font.render("Model: thinking...", True, (255, 255, 0))
             screen.blit(think_text, (x0, y)); y += 20
         if self.last_model_scores:
             # draw small bar chart of scores (normalized)
             scores = [s for (_m, s) in self.last_model_scores]
             min_s = min(scores)
             max_s = max(scores)
             span = max(1e-6, max_s - min_s)
             # limit to top 8 moves
             for m, s in self.last_model_scores[:8]:
                 label = f"{m.frm}->{m.to}"
                 score_text = font.render(label, True, (200, 255, 200))
                 screen.blit(score_text, (x0, y))
                 # bar
                 bar_x = x0 + 88
                 bar_y = y + 4
                 bar_w = self.sidebar_width - 16 - 96
                 normalized = (s - min_s) / span if span > 0 else 0.5
                 fill_w = int(bar_w * normalized)
                 pygame.draw.rect(screen, (30, 200, 30), (bar_x, bar_y, fill_w, 12), border_radius=3)
                 pygame.draw.rect(screen, (80, 80, 80), (bar_x, bar_y, bar_w, 12), 1, border_radius=3)
                 y += 18
             y += 6

         # buttons
         btn_w = self.sidebar_width - 24
         btn_h = 30
         # Reset
         reset_rect = pygame.Rect(x0, y, btn_w, btn_h)
         pygame.draw.rect(screen, (100, 80, 80), reset_rect, border_radius=6)
         reset_text = font.render("Reset", True, (255, 255, 255))
         screen.blit(reset_text, (x0 + 10, y + 6))
         y += btn_h + 8
         # Toggle Model AI
         toggle_rect = pygame.Rect(x0, y, btn_w, btn_h)
         pygame.draw.rect(screen, (80, 100, 80), toggle_rect, border_radius=6)
         mc_label = ",".join(sorted(self.model_control)) if self.model_control else "None"
         toggle_text = font.render(f"Model controls: {mc_label}", True, (255, 255, 255))
         screen.blit(toggle_text, (x0 + 10, y + 6))
         y += btn_h + 8
         # Train Model
         train_rect = pygame.Rect(x0, y, btn_w, btn_h)
         train_color = (80, 80, 120) if not self.training else (120, 120, 80)
         pygame.draw.rect(screen, train_color, train_rect, border_radius=6)
         train_text = font.render("Train Model", True, (255, 255, 255))
         screen.blit(train_text, (x0 + 10, y + 6))
         y += btn_h + 8

         # NN per-color toggles
         nn_row_h = btn_h
         col_w = (btn_w - 6) // 2
         nn_white_rect = pygame.Rect(x0, y, col_w, nn_row_h)
         nn_black_rect = pygame.Rect(x0 + col_w + 6, y, col_w, nn_row_h)
         w_on = 'white' in self.model_control
         b_on = 'black' in self.model_control
         pygame.draw.rect(screen, (90, 120, 180) if w_on else (70, 70, 90), nn_white_rect, border_radius=6)
         pygame.draw.rect(screen, (90, 120, 180) if b_on else (70, 70, 90), nn_black_rect, border_radius=6)
         w_text = font.render("NN: White" + (" ✓" if w_on else ""), True, (255, 255, 255))
         b_text = font.render("NN: Black" + (" ✓" if b_on else ""), True, (255, 255, 255))
         screen.blit(w_text, (nn_white_rect.x + 8, nn_white_rect.y + 6))
         screen.blit(b_text, (nn_black_rect.x + 8, nn_black_rect.y + 6))
         y += nn_row_h + 12

         # cache button rects in sidebar-local coordinates for click handling
         try:
             rel_reset = pygame.Rect(reset_rect.x - rect.x, reset_rect.y - rect.y, reset_rect.w, reset_rect.h)
             rel_toggle = pygame.Rect(toggle_rect.x - rect.x, toggle_rect.y - rect.y, toggle_rect.w, toggle_rect.h)
             rel_train = pygame.Rect(train_rect.x - rect.x, train_rect.y - rect.y, train_rect.w, train_rect.h)
             rel_nn_white = pygame.Rect(nn_white_rect.x - rect.x, nn_white_rect.y - rect.y, nn_white_rect.w, nn_white_rect.h)
             rel_nn_black = pygame.Rect(nn_black_rect.x - rect.x, nn_black_rect.y - rect.y, nn_black_rect.w, nn_black_rect.h)
             self._sidebar_buttons = {'reset': rel_reset, 'toggle': rel_toggle, 'train': rel_train, 'nn_white': rel_nn_white, 'nn_black': rel_nn_black}
         except Exception:
             self._sidebar_buttons = None
