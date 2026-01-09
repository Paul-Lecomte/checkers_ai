"""Neural network utilities for Checkers using PyTorch.

This module provides:
- board_to_tensor(board): convert a Board to a torch tensor input
- CheckersNet: small convnet returning a scalar value estimate for a position
- score_moves(board, moves, model): evaluate candidate moves by applying them and scoring resulting positions
- save_model / load_model helpers
- train_step: a single training step for supervised value regression

Note: PyTorch must be installed in the environment to import and use this module.
"""
from typing import List, Optional, Callable
from random import randint, choice

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.engine.board import Board
from src.engine.move import Move
from src.engine.rules import apply_move, generate_legal_moves, count_pieces


def board_to_tensor(board: Board, device: Optional[torch.device] = None) -> torch.Tensor:
    """Convert a Board into a tensor of shape (1, 4, 8, 8).

    Channels: [white_man, white_king, black_man, black_king]
    Values are 1.0 where a piece exists, else 0.0.
    Returns a batch tensor (batch_size=1).
    """
    t = torch.zeros((4, Board.SIZE, Board.SIZE), dtype=torch.float32)
    for r in range(Board.SIZE):
        for c in range(Board.SIZE):
            p = board.get_piece((r, c))
            if p is None:
                continue
            if p.color == 'white':
                if p.king:
                    t[1, r, c] = 1.0
                else:
                    t[0, r, c] = 1.0
            else:
                if p.king:
                    t[3, r, c] = 1.0
                else:
                    t[2, r, c] = 1.0
    t = t.unsqueeze(0)  # add batch dim
    if device is not None:
        t = t.to(device)
    return t


class CheckersNet(nn.Module):
    """Small convolutional network that outputs a scalar value for a board position."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * Board.SIZE * Board.SIZE, 256)
        self.fc2 = nn.Linear(256, 1)
        # Ajout pour stocker les activations
        self.activations = {}

    def forward(self, x: torch.Tensor, return_activations: bool = False):
        acts = {}
        x1 = F.relu(self.conv1(x))
        acts['conv1'] = x1.detach().cpu().numpy()
        x2 = F.relu(self.conv2(x1))
        acts['conv2'] = x2.detach().cpu().numpy()
        x3 = F.relu(self.conv3(x2))
        acts['conv3'] = x3.detach().cpu().numpy()
        x_flat = x3.view(x3.size(0), -1)
        x4 = F.relu(self.fc1(x_flat))
        acts['fc1'] = x4.detach().cpu().numpy()
        x5 = self.fc2(x4)
        acts['fc2'] = x5.detach().cpu().numpy()
        if return_activations:
            return x5.squeeze(-1), acts
        return x5.squeeze(-1)


def score_moves(board: Board, moves: List[Move], model: nn.Module, device: Optional[torch.device] = None) -> List[tuple]:
    """Return list of (move, score) where score is the model evaluation of the resulting board.

    Higher score means better for white by convention (we don't enforce sign here; you can train accordingly).
    """
    model.eval()
    results = []
    with torch.no_grad():
        for m in moves:
            nb = apply_move(board, m)
            inp = board_to_tensor(nb, device=device)
            out = model(inp)
            # out can be tensor of shape (1,) or scalar
            score = float(out.cpu().numpy().item()) if isinstance(out, torch.Tensor) else float(out)
            results.append((m, score))
    return results


def score_moves_with_activations(board: Board, moves: List[Move], model: nn.Module, device: Optional[torch.device] = None):
    """Retourne (move, score, activations) pour chaque coup."""
    model.eval()
    results = []
    with torch.no_grad():
        for m in moves:
            nb = apply_move(board, m)
            inp = board_to_tensor(nb, device=device)
            out, acts = model(inp, return_activations=True)
            score = float(out.cpu().numpy().item()) if isinstance(out, torch.Tensor) else float(out)
            results.append((m, score, acts))
    return results


def save_model(model: nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)


def load_model(model: nn.Module, path: str, device: Optional[torch.device] = None) -> nn.Module:
    map_location = device if device is not None else None
    model.load_state_dict(torch.load(path, map_location=map_location))
    model.eval()
    return model


def train_step(model: nn.Module, optimizer: torch.optim.Optimizer, boards: torch.Tensor, targets: torch.Tensor) -> float:
    """Perform one training step on a batch.

    boards: tensor shape (B, 4, 8, 8)
    targets: tensor shape (B,) scalar target values
    Returns the loss value (float)
    """
    model.train()
    preds = model(boards)
    loss = F.mse_loss(preds, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def train_random_positions(model: nn.Module, epochs: int = 10, lr: float = 1e-3, n_positions: int = 32, max_playout: int = 12, device: Optional[torch.device] = None, progress_callback: Optional[Callable[[int, float], None]] = None) -> None:
    """Train model for a few epochs on randomly generated positions.

    This is a lightweight demo trainer: it generates random playouts from the start
    position (random lengths up to max_playout), computes a simple material-based
    target (white pieces - black pieces normalized), and trains the network using MSE.

    - model is modified in-place.
    - progress_callback(epoch, avg_loss) called after each epoch if provided.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        losses = []
        boards_batch = []
        targets = []
        for i in range(n_positions):
            # generate random playout
            b = Board.setup_start()
            current = 'black'
            steps = randint(0, max_playout)
            for _ in range(steps):
                moves = generate_legal_moves(b, current)
                if not moves:
                    break
                mv = choice(moves)
                b = apply_move(b, mv)
                current = 'white' if current == 'black' else 'black'
            # target = normalized material balance (white - black)/12
            wc = count_pieces(b, 'white')
            bc = count_pieces(b, 'black')
            target = float(wc - bc) / 12.0
            boards_batch.append(board_to_tensor(b, device=device).squeeze(0))
            targets.append(target)

        # create tensors
        boards_tensor = torch.stack(boards_batch, dim=0)
        targets_tensor = torch.tensor(targets, dtype=torch.float32, device=device)

        loss = train_step(model, optimizer, boards_tensor, targets_tensor)
        if progress_callback:
            try:
                progress_callback(epoch, loss)
            except Exception:
                pass

    # done training
    model.to('cpu')
