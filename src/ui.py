from typing import Callable, Optional

import pygame

from src.engine.board import Board


class PygameUI:
    """Minimal Pygame UI for the checkers board.

    - board: instance of Board
    - square_size: pixels per square
    - margin: outer margin in pixels
    - on_advance: optional callable Board -> Board called when SPACE is pressed
    """

    def __init__(self, board: Board, square_size: int = 80, margin: int = 20, on_advance: Optional[Callable] = None):
        self.board = board
        self.square_size = square_size
        self.margin = margin
        self.on_advance = on_advance

    def set_on_advance(self, callback: Callable[[Board], Board]):
        self.on_advance = callback

    def run(self) -> None:
        """Start the pygame loop and render the board.

        Handles QUIT, ESC/q to quit, SPACE to call on_advance.
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

            # draw
            screen.fill((50, 50, 50))
            self._draw_board(screen)
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
