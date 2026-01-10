"""Neural network utilities and PPO agent for Checkers using PyTorch.

This module now provides:
- board_to_tensor(board): convert a Board to a torch tensor input
- move_to_features(move, board): compact move feature vector
- PolicyValueNet: shared conv encoder with policy and value heads
- PPOAgent: rollout buffer, action selection over variable legal moves, PPO update
- score_moves helpers (still available) and legacy CheckersNet preserved for compatibility

Note: PyTorch must be installed in the environment to import and use this module.
"""
from typing import List, Optional, Callable, Tuple, Dict
from dataclasses import dataclass
from random import randint, choice

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.engine.board import Board
from src.engine.move import Move
from src.engine.rules import apply_move, generate_legal_moves, count_pieces


# -------------------- Encoding --------------------

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


def move_to_features(move: Move, board: Board) -> torch.Tensor:
    """Encode a Move into a small feature vector.

    Features (length 10):
    - from_row, from_col (normalized to [0,1])
    - to_row, to_col (normalized)
    - delta_row, delta_col (normalized to [-1,1] then scaled to [0,1])
    - is_capture (0/1)
    - capture_count normalized by 12
    - becomes_king (0/1)
    - current_player_is_white (0/1)
    """
    fr_r, fr_c = move.start
    to_r, to_c = move.end
    # normalize to [0,1]
    def norm_xy(x: int) -> float:
        return float(x) / (Board.SIZE - 1)

    dr = to_r - fr_r
    dc = to_c - fr_c
    # map deltas to [-1,1] then to [0,1]
    def norm_delta(d: int) -> float:
        # maximum step in checkers moves is up to 2 per single jump; multi-jump not represented directly
        d = max(-2, min(2, d))
        return (d + 2) / 4.0

    is_capture = 1.0 if (move.captures and len(move.captures) > 0) else 0.0
    cap_count = (len(move.captures) if move.captures else 0) / 12.0

    # heuristic: does move result in kinging
    becomes_king = 0.0
    p = board.get_piece(move.start)
    if p is not None and not p.king:
        if p.color == 'white' and to_r == 0:
            becomes_king = 1.0
        if p.color == 'black' and to_r == Board.SIZE - 1:
            becomes_king = 1.0

    current_white = 1.0 if p is not None and p.color == 'white' else 0.0

    feats = torch.tensor([
        norm_xy(fr_r), norm_xy(fr_c), norm_xy(to_r), norm_xy(to_c),
        norm_delta(dr), norm_delta(dc),
        is_capture, cap_count,
        becomes_king,
        current_white,
    ], dtype=torch.float32)
    return feats


# -------------------- Policy-Value Network --------------------

class PolicyValueNet(nn.Module):
    """Shared convolutional encoder over board with policy and value heads.

    Policy head produces per-move logits given move features and board embedding.
    Value head produces scalar state value.
    """

    def __init__(self, board_emb_dim: int = 256, move_emb_dim: int = 64):
        super().__init__()
        # Encoder
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc_enc = nn.Linear(64 * Board.SIZE * Board.SIZE, board_emb_dim)
        # Value head
        self.val_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(board_emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # Move embedding and policy scorer
        self.move_mlp = nn.Sequential(
            nn.Linear(10, move_emb_dim),
            nn.ReLU(),
            nn.Linear(move_emb_dim, move_emb_dim),
            nn.ReLU(),
        )
        self.policy_scorer = nn.Sequential(
            nn.Linear(board_emb_dim + move_emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # for UI activations (optional)
        self.activations: Dict[str, torch.Tensor] = {}

    def encode_board(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        self.activations['conv1'] = x.detach()
        x = F.relu(self.conv2(x))
        self.activations['conv2'] = x.detach()
        x = F.relu(self.conv3(x))
        self.activations['conv3'] = x.detach()
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_enc(x))
        self.activations['enc'] = x.detach()
        return x

    def forward(self, boards: torch.Tensor, moves_batch: Optional[List[List[torch.Tensor]]] = None, return_activations: bool = False):
        """If moves_batch is provided (list per item of list of feature tensors), returns (logits_list, values).
        Otherwise returns only values.
        """
        emb = self.encode_board(boards)
        values = self.val_head(emb).squeeze(-1)
        if moves_batch is None:
            return (None, values) if not return_activations else ((None, values), self._collect_acts())
        # For each item in batch, compute per-move logits
        logits_list: List[torch.Tensor] = []
        for i, moves in enumerate(moves_batch):
            if len(moves) == 0:
                logits_list.append(torch.empty((0,), device=boards.device))
                continue
            mv_feats = torch.stack(moves, dim=0).to(boards.device)
            mv_emb = self.move_mlp(mv_feats)
            # broadcast board emb to each move
            board_i = emb[i].unsqueeze(0).expand(mv_emb.size(0), -1)
            joint = torch.cat([board_i, mv_emb], dim=1)
            logits = self.policy_scorer(joint).squeeze(-1)
            logits_list.append(logits)
        if return_activations:
            return (logits_list, values), self._collect_acts()
        return logits_list, values

    def _collect_acts(self) -> Dict[str, torch.Tensor]:
        # convert to cpu numpy where useful would be done by UI side
        return {k: v.detach().cpu() for k, v in self.activations.items()}


# -------------------- PPO Agent --------------------

@dataclass
class Transition:
    board_tensor: torch.Tensor
    move_feats: List[torch.Tensor]
    action_idx: int
    logprob: float
    value: float
    reward: float
    done: bool


class RolloutBuffer:
    def __init__(self):
        self.storage: List[Transition] = []

    def add(self, t: Transition):
        self.storage.append(t)

    def clear(self):
        self.storage.clear()

    def __len__(self):
        return len(self.storage)


class PPOAgent:
    """PPO agent handling variable action spaces via per-state move lists.

    Use select_action(board, legal_moves) during play and call update() after collecting rollouts.
    """

    def __init__(self, model: Optional[PolicyValueNet] = None, device: Optional[torch.device] = None,
                 lr: float = 3e-4, gamma: float = 0.99, gae_lambda: float = 0.95, clip_eps: float = 0.2,
                 entropy_coef: float = 0.01, value_coef: float = 0.5, max_grad_norm: float = 0.5):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model or PolicyValueNet()
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.buffer = RolloutBuffer()

    def select_action(self, board: Board, legal_moves: List[Move]) -> Tuple[Optional[Move], Dict]:
        """Return chosen move and info dict; stores transition in buffer (without reward yet).
        If no legal moves, returns (None, {}).
        """
        if not legal_moves:
            return None, {}
        b_t = board_to_tensor(board, device=self.device)
        moves_feats = [move_to_features(m, board) for m in legal_moves]
        with torch.no_grad():
            logits_list, values = self.model(b_t, [moves_feats])
            logits = logits_list[0]
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            action_idx = dist.sample().item()
            logprob = float(dist.log_prob(torch.tensor(action_idx, device=self.device)).item())
            value = float(values[0].item())
        chosen = legal_moves[action_idx]
        info = {
            'probs': probs.detach().cpu().numpy(),
            'logits': logits.detach().cpu().numpy(),
            'value': value,
        }
        # Store placeholder reward/done to be filled externally via "record_outcome"
        self.buffer.add(Transition(b_t.squeeze(0).detach(), moves_feats, action_idx, logprob, value, 0.0, False))
        return chosen, info

    def record_outcome(self, reward: float, done: bool) -> None:
        """Attach reward/done to the last stored transition."""
        if len(self.buffer.storage) == 0:
            return
        self.buffer.storage[-1].reward = reward
        self.buffer.storage[-1].done = done

    def update(self, epochs: int = 4, batch_size: int = 32) -> Dict[str, float]:
        """Perform PPO update over collected buffer."""
        if len(self.buffer) == 0:
            return {'loss': 0.0}
        # Prepare flattened tensors
        boards = torch.stack([t.board_tensor for t in self.buffer.storage], dim=0).to(self.device)
        # Variable-length moves; we recompute logits per transition
        actions = torch.tensor([t.action_idx for t in self.buffer.storage], dtype=torch.long, device=self.device)
        old_logprobs = torch.tensor([t.logprob for t in self.buffer.storage], dtype=torch.float32, device=self.device)
        rewards = torch.tensor([t.reward for t in self.buffer.storage], dtype=torch.float32, device=self.device)
        values = torch.tensor([t.value for t in self.buffer.storage], dtype=torch.float32, device=self.device)
        dones = torch.tensor([t.done for t in self.buffer.storage], dtype=torch.float32, device=self.device)
        # Compute returns and advantages (GAE)
        returns = torch.zeros_like(rewards)
        advs = torch.zeros_like(rewards)
        last_gae = 0.0
        last_return = 0.0
        for t in reversed(range(len(self.buffer.storage))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * (values[t + 1] if t + 1 < len(values) else 0.0) * mask - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * mask * last_gae
            advs[t] = last_gae
            last_return = rewards[t] + self.gamma * (last_return if mask > 0 else 0.0)
            returns[t] = last_return
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        # Training epochs
        losses = []
        n = len(self.buffer.storage)
        indices = torch.arange(n)
        for _ in range(epochs):
            # simple full-batch or mini-batch shuffle
            perm = torch.randperm(n)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                batch_boards = boards[idx]
                batch_moves = [self.buffer.storage[i].move_feats for i in idx.tolist()]
                batch_actions = actions[idx]
                batch_old_logprobs = old_logprobs[idx]
                batch_advs = advs[idx]
                batch_returns = returns[idx]

                logits_list, new_values = self.model(batch_boards, batch_moves)
                # gather logprobs and entropy
                logprobs = []
                entropies = []
                for i, logits in enumerate(logits_list):
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs=probs)
                    a = batch_actions[i]
                    logprobs.append(dist.log_prob(a))
                    entropies.append(dist.entropy())
                logprobs = torch.stack(logprobs)
                entropy = torch.stack(entropies).mean()

                # PPO objective
                ratios = torch.exp(logprobs - batch_old_logprobs)
                surr1 = ratios * batch_advs
                surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch_advs
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(new_values, batch_returns)

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                losses.append(float(loss.item()))

        # clear buffer after update
        self.buffer.clear()
        return {
            'loss': sum(losses) / max(1, len(losses)),
        }


# -------------------- Legacy value-only network and helpers --------------------

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


# -------------------- IO & Training helpers --------------------

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
