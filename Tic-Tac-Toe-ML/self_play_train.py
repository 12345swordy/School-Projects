# self_play_train.py
import os
import re
import numpy as np
import torch
import torch.optim as optim
import csv

from net_models import SmallNet, MediumNet, LargeNet
from dataset import Board
from rl_algos import (
    encode_state_from_board,
    legal_mask_from_board,
    epsilon_greedy,
    step_board,
    QLearningLearner,
    SarsaLearner,
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
SELF_PLAY_COUNTS = [10, 50, 100]
ALGOS = ["q", "sarsa"]

LR = 1e-4
WEIGHT_DECAY = 0.0
GAMMA = 0.99
EPS_START = 0.20
EPS_END = 0.05
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

def parse_shape_from_name(name):
    m = re.search(r'(\d+)x(\d+)(?:x(\d+))?', name)
    if not m:
        return None
    a = int(m.group(1)); b = int(m.group(2)); c = m.group(3)
    if c is None:
        return (False, (a, b))
    else:
        return (True, (a, b, int(c)))

def pick_model_class(name):
    if name.startswith("small"): return SmallNet
    if name.startswith("medium"): return MediumNet
    if name.startswith("large"): return LargeNet
    return None

def make_board(is_3d, dims):
    if is_3d:
        D, H, W = dims
        side = D
        return Board((D, H, W), x_in_a_row=side)
    else:
        H, W = dims
        side = H
        return Board((H, W), x_in_a_row=side)

def eps_for_episode(ep_idx, total_eps):
    if total_eps <= 1:
        return EPS_END
    frac = ep_idx / (total_eps - 1)
    return float(EPS_START + (EPS_END - EPS_START) * frac)

def run_self_play_q(learner, board, episodes):
    wins_x = wins_o = draws = 0
    for ep in range(episodes):
        board.state[:] = 0
        board.turn = 1
        eps = eps_for_episode(ep, episodes)
        while True:
            s = encode_state_from_board(board, player_relative=True, include_turn_bit=True).to(learner.device)
            m = legal_mask_from_board(board).to(learner.device)
            q = learner.model(s.unsqueeze(0))[0]
            a = epsilon_greedy(q, m, epsilon=eps)
            next_state, reward, done, info = step_board(board, a, player_relative=True, include_turn_bit=True)
            next_state = next_state.to(learner.device)
            learner.update(s, a, reward, next_state, done, next_mask=info["mask"].to(learner.device))
            if done:
                res = info["result"]
                if res == 2: wins_x += 1
                elif res == 1: wins_o += 1
                else: draws += 1
                break
    return wins_x, wins_o, draws

def run_self_play_sarsa(learner, board, episodes):
    wins_x = wins_o = draws = 0
    for ep in range(episodes):
        board.state[:] = 0
        board.turn = 1
        eps = eps_for_episode(ep, episodes)

        s = encode_state_from_board(board, player_relative=True, include_turn_bit=True).to(learner.device)
        m = legal_mask_from_board(board).to(learner.device)
        q = learner.model(s.unsqueeze(0))[0]
        a = epsilon_greedy(q, m, epsilon=eps)

        while True:
            next_state, reward, done, info = step_board(board, a, player_relative=True, include_turn_bit=True)
            next_state = next_state.to(learner.device)

            if done:
                learner.update(s, a, reward, next_state, a, True)
                res = info["result"]
                if res == 2: wins_x += 1
                elif res == 1: wins_o += 1
                else: draws += 1
                break

            next_mask = info["mask"].to(learner.device)
            q_next = learner.model(next_state.unsqueeze(0))[0]
            a_next = epsilon_greedy(q_next, next_mask, epsilon=eps)

            learner.update(s, a, reward, next_state, a_next, done)

            s = next_state
            a = a_next
    return wins_x, wins_o, draws

def is_selfplay_file(base):
    return re.search(r'-(q|sarsa)-sp(\d+)\.pt$', base) is not None

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(MODEL_DIR, exist_ok=True)

    for fname in os.listdir(MODEL_DIR):
        if not fname.endswith(".pt"): continue
        if is_selfplay_file(fname):   continue

        shape_info = parse_shape_from_name(fname)
        if shape_info is None:
            print(f"Skip (no shape in name): {fname}")
            continue
        is_3d, dims = shape_info

        model_cls = pick_model_class(fname)
        if model_cls is None:
            print(f"Skip (unknown model class): {fname}")
            continue

        # Inline: build model for shape
        num_cells = int(np.prod(dims))
        in_size = num_cells + 1
        out_size = num_cells
        model = model_cls(in_size, out_size)

        path = os.path.join(MODEL_DIR, fname)
        try:
            state = torch.load(path, map_location="cpu")
            model.load_state_dict(state, strict=False)
        except Exception as e:
            print(f"Failed to load {fname}: {e}")
            continue
        model.to(device)

        board = make_board(is_3d, dims)

        for n_sp in SELF_PLAY_COUNTS:
            for algo in ALGOS:
                suffix = f"-{algo}-sp{n_sp}.pt"
                sp_name = fname[:-3] + suffix
                sp_path = os.path.join(MODEL_DIR, sp_name)
                # if os.path.exists(sp_path):
                #     print(f"Exists, skip: {sp_name}")
                #     continue

                opt = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

                if algo == "q":
                    learner = QLearningLearner(model, opt, gamma=GAMMA, device=device)
                    wx, wo, dr = run_self_play_q(learner, board, episodes=n_sp)
                else:
                    learner = SarsaLearner(model, opt, gamma=GAMMA, device=device)
                    wx, wo, dr = run_self_play_sarsa(learner, board, episodes=n_sp)

                torch.save(model.state_dict(), sp_path)
                result_line = {
                    "model": sp_name,
                    "algorithm": algo.upper(),
                    "games": n_sp,
                    "wins_X": wx,
                    "wins_O": wo,
                    "draws": dr,
                }
                print(
                    f"Saved {sp_name}  |  {algo.upper()} self-play: {n_sp} games  "
                    f"|  W(X)/W(O)/D = {wx}/{wo}/{dr}"
                )

                csv_path = "self_play_results.csv"
                file_exists = os.path.exists(csv_path)
                with open(csv_path, "a", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=["model", "algorithm", "games", "wins_X", "wins_O", "draws"],
                    )
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(result_line)




if __name__ == "__main__":
    main()