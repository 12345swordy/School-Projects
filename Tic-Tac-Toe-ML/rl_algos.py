import numpy as np
import torch
import torch.nn as nn
from dataset import Board

def action_space_n(board):
    return int(np.prod(board.dimensions))

def flatten_board(board):
    return board.state.reshape(-1)

def coord_to_index(board, coord):
    return int(np.ravel_multi_index(tuple(coord), board.dimensions, order="C"))

def index_to_coord(board, idx):
    return tuple(int(x) for x in np.unravel_index(int(idx), board.dimensions, order="C"))

def legal_mask_from_board(board):
    A = action_space_n(board)
    mask = np.zeros(A, dtype=bool)
    empties = board.possible_moves()
    if empties.size > 0:
        flat_idx = np.ravel_multi_index(empties.T, board.dimensions, order="C")
        mask[flat_idx] = True
    return torch.from_numpy(mask)

def encode_state_from_board(board, player_relative=True, include_turn_bit=True):
    flat = flatten_board(board).astype(np.float32)
    if player_relative:
        cur = board.turn
        opp = 2 if cur == 1 else 1
        rel = np.zeros_like(flat, dtype=np.float32)
        rel[flat == cur] = 1.0
        rel[flat == opp] = -1.0
        out = rel
    else:
        out = (flat - 1.0) / 1.0
    if include_turn_bit:
        tbit = np.array([+1.0 if board.turn == 1 else -1.0], dtype=np.float32)
        out = np.concatenate([out, tbit], axis=0)
    return torch.from_numpy(out)

def epsilon_greedy(q_vals, mask=None, epsilon=0.1):
    q = q_vals.clone()
    if mask is not None:
        q[~mask] = -1e9
        if not mask.any():
            raise ValueError("epsilon_greedy: no legal actions in mask")
    if torch.rand(()) < epsilon:
        if mask is None:
            return torch.randint(q.numel(), ()).item()
        valid = torch.nonzero(mask, as_tuple=False).view(-1)
        return valid[torch.randint(valid.numel(), ()).item()].item()
    return torch.argmax(q).item()

def _reward_from_result(mover, result):
    if result is None:
        return 0.0
    if result == 0:
        return 0.0
    return 1.0 if result == mover else -1.0

def step_board(board, action_idx, player_relative=True, include_turn_bit=True):
    mask_now = legal_mask_from_board(board)
    if action_idx < 0 or action_idx >= mask_now.numel():
        raise ValueError("step_board: action index out of range")
    if not mask_now[action_idx]:
        raise ValueError("step_board: illegal action (cell occupied)")
    mover = board.turn
    coord = index_to_coord(board, action_idx)
    board.push(coord)
    result = board.result()
    reward = _reward_from_result(mover, result)
    done = (result is not None)
    next_state = encode_state_from_board(board, player_relative, include_turn_bit)
    next_mask = legal_mask_from_board(board)
    info = {"mask": next_mask, "result": result, "mover": mover, "coord": coord}
    return next_state, reward, done, info

def _ensure_batch_state(x):
    if x.ndim == 1:
        return x.unsqueeze(0), False
    return x, True

class _BaseLearner:
    def __init__(self, model, optimizer, gamma=0.99, loss_fn=None, grad_clip=1.0, device=None):
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()
        self.grad_clip = grad_clip
        self.device = device if device is not None else next(model.parameters()).device
        self.model.to(self.device)

    @torch.no_grad()
    def act_on_board(self, board, epsilon=0.1, player_relative=True, include_turn_bit=True):
        self.model.eval()
        s = encode_state_from_board(board, player_relative, include_turn_bit).to(self.device)
        m = legal_mask_from_board(board).to(self.device)
        q = self.model(s.unsqueeze(0))[0]
        a = epsilon_greedy(q, m, epsilon)
        return a

    def _gstep(self, loss):
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip is not None and self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

class QLearningLearner(_BaseLearner):
    def update(self, state, action, reward, next_state, done, mask=None, next_mask=None):
        self.model.train()
        s = state.to(self.device)
        ns = next_state.to(self.device)
        s, _ = _ensure_batch_state(s)
        ns, _ = _ensure_batch_state(ns)
        if isinstance(action, int):
            action = torch.tensor([action], device=self.device, dtype=torch.long)
        else:
            action = action.to(self.device).long()
        if isinstance(reward, (int, float)):
            reward = torch.tensor([reward], device=self.device, dtype=torch.float32)
        else:
            reward = reward.to(self.device).float()
        if isinstance(done, bool):
            done = torch.tensor([done], device=self.device, dtype=torch.bool)
        else:
            done = done.to(self.device)
        q_all = self.model(s)
        q_sa = q_all.gather(1, action.view(-1, 1)).squeeze(1)
        with torch.no_grad():
            q_next = self.model(ns)
            if next_mask is not None:
                nm = next_mask.to(self.device).bool()
                if nm.ndim == 1:
                    nm = nm.unsqueeze(0).expand(q_next.size(0), -1)
                elif nm.ndim == 2 and nm.size(0) != q_next.size(0):
                    nm = nm.expand(q_next.size(0), -1)
                q_next = q_next.clone()
                q_next[~nm] = -1e9
            q_next_max, _ = q_next.max(dim=1)
            targets = reward + (~done).float() * self.gamma * q_next_max
        loss = self.loss_fn(q_sa, targets)
        self._gstep(loss)
        return float(loss.detach().cpu().item())

    def update_from_boards(self, board_before, action_idx, board_after, reward, done,
                           player_relative=True, include_turn_bit=True):
        s = encode_state_from_board(board_before, player_relative, include_turn_bit)
        ns = encode_state_from_board(board_after, player_relative, include_turn_bit)
        next_mask = legal_mask_from_board(board_after)
        return self.update(s, action_idx, reward, ns, done, next_mask=next_mask)

class SarsaLearner(_BaseLearner):
    def update(self, state, action, reward, next_state, next_action, done, mask=None, next_mask=None):
        self.model.train()
        s = state.to(self.device)
        ns = next_state.to(self.device)
        s, _ = _ensure_batch_state(s)
        ns, _ = _ensure_batch_state(ns)
        if isinstance(action, int):
            action = torch.tensor([action], device=self.device, dtype=torch.long)
        else:
            action = action.to(self.device).long()
        if isinstance(next_action, int):
            next_action = torch.tensor([next_action], device=self.device, dtype=torch.long)
        else:
            next_action = next_action.to(self.device).long()
        if isinstance(reward, (int, float)):
            reward = torch.tensor([reward], device=self.device, dtype=torch.float32)
        else:
            reward = reward.to(self.device).float()
        if isinstance(done, bool):
            done = torch.tensor([done], device=self.device, dtype=torch.bool)
        else:
            done = done.to(self.device)
        q_all = self.model(s)
        q_sa = q_all.gather(1, action.view(-1, 1)).squeeze(1)
        with torch.no_grad():
            q_next = self.model(ns)
            if next_mask is not None:
                nm = next_mask.to(self.device).bool()
                if nm.ndim == 1:
                    nm = nm.unsqueeze(0).expand(q_next.size(0), -1)
                elif nm.ndim == 2 and nm.size(0) != q_next.size(0):
                    nm = nm.expand(q_next.size(0), -1)
                q_next = q_next.clone()
                q_next[~nm] = -1e9

            q_next_ap = q_next.gather(1, next_action.view(-1, 1)).squeeze(1)
            targets = reward + (~done).float() * self.gamma * q_next_ap

        loss = self.loss_fn(q_sa, targets)
        self._gstep(loss)
        return float(loss.detach().cpu().item())


    def update_from_boards(self, board_before, action_idx, board_after, next_action_idx, reward, done,
                           player_relative=True, include_turn_bit=True):
        s = encode_state_from_board(board_before, player_relative, include_turn_bit)
        ns = encode_state_from_board(board_after, player_relative, include_turn_bit)
        return self.update(s, action_idx, reward, ns, next_action_idx, done)
