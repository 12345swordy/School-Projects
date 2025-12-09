import os
import sqlite3
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from net_models import SmallNet, MediumNet, LargeNet

DB_PATH = os.path.join(os.path.dirname(__file__), "dataset.db")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
IMAGE_DIR = os.path.join(os.path.dirname(__file__), "images")

LR = 1e-3
WEIGHT_DECAY = 0.0
SEED = 42

# new: sweep settings
EPOCHS_LIST = [20, 50, 100]
BATCH_SIZES = [64, 128, 256]

SHAPES_2D = [(1, n, n) for n in (3, 4, 5)]
SHAPES_3D = [(n, n, n) for n in (3, 4, 5)]
ALL_SHAPES = SHAPES_2D + SHAPES_3D

torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

def state_encode_player_relative(board_blob, D, H, W, x_turn, include_turn_bit=True):
    arr = np.frombuffer(board_blob, dtype=np.int8).reshape(D, H, W)
    flat = arr.reshape(-1)
    rel = np.zeros_like(flat, dtype=np.float32)
    if x_turn == 1:
        rel[flat == 2] = 1.0
        rel[flat == 1] = -1.0
    else:
        rel[flat == 1] = 1.0
        rel[flat == 2] = -1.0
    if include_turn_bit:
        tbit = np.array([+1.0 if x_turn == 1 else -1.0], dtype=np.float32)
        rel = np.concatenate([rel.astype(np.float32), tbit], axis=0)
    return rel

def dims_to_label(D, H, W):
    return f"{H}x{W}" if D == 1 else f"{D}x{H}x{W}"

def fetch_rows_by_shape(conn, D, H, W):
    cur = conn.cursor()
    cur.execute(
        "SELECT GameID, TurnID, X_Turn, Board, Win, Moves "
        "FROM Game WHERE D=? AND H=? AND W=? "
        "ORDER BY GameID ASC, TurnID ASC",
        (D, H, W),
    )
    rows = cur.fetchall()
    games = {}
    for gid, tid, xturn, board, win, moves in rows:
        games.setdefault(gid, []).append((tid, xturn, board, win, moves))
    for gid in list(games.keys()):
        games[gid].sort(key=lambda r: r[0])
    return games

def build_xy_for_shape(conn, D, H, W):
    games = fetch_rows_by_shape(conn, D, H, W)
    X, y = [], []
    flat_dims = (H, W) if D == 1 else (D, H, W)
    for gid, rows in games.items():
        T = len(rows)
        if T < 2:
            continue
        for i in range(T - 1):
            _, xturn_t, board_blob_t, _, _ = rows[i]
            _, _, _, _, moves_blob_tp1 = rows[i + 1]
            if not moves_blob_tp1:
                continue
            a = int(np.frombuffer(moves_blob_tp1, dtype=np.int64)[0])
            s = state_encode_player_relative(board_blob_t, D, H, W, xturn_t, include_turn_bit=True)
            num_cells = int(np.prod(flat_dims))
            if not (0 <= a < num_cells):
                continue
            X.append(s)
            y.append(a)
    if not X:
        return None, None
    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y

def train_one(model, loader, device, epochs):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    hist_loss = []
    hist_acc = []
    for _ in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running_loss += float(loss.detach().cpu().item()) * yb.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.size(0))
        epoch_loss = running_loss / max(1, total)
        epoch_acc = correct / max(1, total)
        hist_loss.append(epoch_loss)
        hist_acc.append(epoch_acc)
    return {"loss": hist_loss, "acc": hist_acc}

def plot_history(history, title, out_png_path):
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["loss"])
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy")
    axes[1].plot(epochs, history["acc"])
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Top-1 Acc")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_png_path, dpi=120)
    plt.close(fig)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conn = sqlite3.connect(DB_PATH)
    model_factories = {
        "small": lambda in_size, out_size: SmallNet(in_size, out_size),
        "medium": lambda in_size, out_size: MediumNet(in_size, out_size),
        "large": lambda in_size, out_size: LargeNet(in_size, out_size),
    }

    for (D, H, W) in ALL_SHAPES:
        label = dims_to_label(D, H, W)
        print(f"\n=== Shape {label} ===")
        X, y = build_xy_for_shape(conn, D, H, W)
        if X is None:
            print("No samples found; skipping.")
            continue
        in_size = X.shape[1]
        out_size = int(np.prod((H, W))) if D == 1 else int(np.prod((D, H, W)))

        for bs in BATCH_SIZES:
            ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
            loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=False)
            for epochs in EPOCHS_LIST:
                for name, factory in model_factories.items():
                    print(f"Training {name} on {label} (epochs={epochs}, batch={bs}) "
                          f"(in={in_size}, out={out_size}, samples={len(ds)})")
                    model = factory(in_size, out_size).to(device)
                    history = train_one(model, loader, device, epochs)
                    out_stem = f"{name}-{label}-e{epochs}-b{bs}"
                    out_path = os.path.join(MODEL_DIR, out_stem + ".pt")
                    torch.save(model.state_dict(), out_path)
                    print(f"Saved -> {out_path}")
                    plot_path = os.path.join(IMAGE_DIR, out_stem + ".png")
                    plot_history(history, title=out_stem, out_png_path=plot_path)
                    print(f"Plot  -> {plot_path}")

    conn.close()

if __name__ == "__main__":
    main()