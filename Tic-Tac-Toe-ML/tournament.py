import os
import re
import csv
import numpy as np
import torch

from net_models import SmallNet, MediumNet, LargeNet
from dataset import Board
from rl_algos import encode_state_from_board, legal_mask_from_board

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
OUT_CSV   = os.path.join(os.path.dirname(__file__), "tournament_results.csv")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def parse_shape_from_name(name):
    m = re.search(r'(\d+)x(\d+)(?:x(\d+))?', name)
    if not m:
        return None
    a = int(m.group(1)); b = int(m.group(2)); c = m.group(3)
    return (False, (a, b)) if c is None else (True, (a, b, int(c)))

def pick_model_class(name):
    if name.startswith("small"):  return SmallNet
    if name.startswith("medium"): return MediumNet
    if name.startswith("large"):  return LargeNet
    return None

def build_model_from_file(path):
    fname = os.path.basename(path)
    shape_info = parse_shape_from_name(fname)
    if shape_info is None:
        return None, None, None, None
    is_3d, dims = shape_info
    model_cls = pick_model_class(fname)
    if model_cls is None:
        return None, None, None, None
    num_cells = int(np.prod(dims))
    in_size  = num_cells + 1
    out_size = num_cells
    model = model_cls(in_size, out_size)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    return model, is_3d, dims, fname

def make_board(is_3d, dims):
    if is_3d:
        D, H, W = dims
        return Board((D, H, W), x_in_a_row=D)
    else:
        H, W = dims
        return Board((H, W), x_in_a_row=H)

@torch.no_grad()
def greedy_action(model, board, device):
    s = encode_state_from_board(board, player_relative=True, include_turn_bit=True).to(device)
    q = model(s.unsqueeze(0))[0]  # [A]
    mask = legal_mask_from_board(board).to(device)  # [A]
    if not mask.any():
        raise RuntimeError("No legal actions.")
    q = q.clone()
    q[~mask] = -1e9
    return int(torch.argmax(q).item())

def play_one_game(model_X, model_O, board, device):
    board.state[:] = 0
    board.turn = 1  # X starts
    while True:
        if board.turn == 1:
            a = greedy_action(model_X, board, device)
        else:
            a = greedy_action(model_O, board, device)
        coord = np.unravel_index(a, board.dimensions, order="C")
        board.push(coord)
        res = board.result()  # 2=X win, 1=O win, 0=draw, None=ongoing
        if res is not None:
            return int(res)

def round_robin(models_for_shape, is_3d, dims, games_per_pair=10, device="cpu"):
    board = make_board(is_3d, dims)
    n = len(models_for_shape)
    results = []
    for i in range(n):
        for j in range(i + 1, n):
            name_a, model_a = models_for_shape[i]
            name_b, model_b = models_for_shape[j]
            model_a.to(device).eval()
            model_b.to(device).eval()

            wins_a = wins_b = draws = 0
            for g in range(games_per_pair):
                if g % 2 == 0:
                    # A is X, B is O
                    res = play_one_game(model_a, model_b, board, device)
                    if res == 2: wins_a += 1
                    elif res == 1: wins_b += 1
                    else: draws += 1
                else:
                    # B is X, A is O
                    res = play_one_game(model_b, model_a, board, device)
                    if res == 2: wins_b += 1
                    elif res == 1: wins_a += 1
                    else: draws += 1

            shape_label = f"{dims[0]}x{dims[1]}" if not is_3d else f"{dims[0]}x{dims[1]}x{dims[2]}"
            results.append({
                "shape":   shape_label,
                "model_a": name_a,
                "model_b": name_b,
                "games":   games_per_pair,
                "wins_a":  wins_a,
                "wins_b":  wins_b,
                "draws":   draws,
            })
            print(f"[{shape_label}] {name_a} vs {name_b} -> A:{wins_a} B:{wins_b} D:{draws}")
    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    by_shape = {}

    for fname in os.listdir(MODEL_DIR):
        if not fname.endswith(".pt"):
            continue
        full = os.path.join(MODEL_DIR, fname)
        model, is_3d, dims, name = build_model_from_file(full)
        if model is None:
            continue
        key = (is_3d, dims)
        by_shape.setdefault(key, []).append((name, model))

    all_rows = []
    for (is_3d, dims), model_list in by_shape.items():
        # only run if we have at least 2 models for this shape
        if len(model_list) < 2:
            continue
        rows = round_robin(model_list, is_3d, dims, games_per_pair=10, device=device)
        all_rows.extend(rows)

    # write CSV
    if all_rows:
        with open(OUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["shape", "model_a", "model_b", "games", "wins_a", "wins_b", "draws"],
            )
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nSaved tournament results -> {OUT_CSV}")
    else:
        print("No shape had >=2 models. Nothing to evaluate.")

if __name__ == "__main__":
    main()
