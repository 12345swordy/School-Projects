# read_dataset.py
import argparse
import sqlite3
from collections import defaultdict
from typing import Iterable, List, Optional, Tuple

import numpy as np

Coord3 = Tuple[int, int, int]  # (d, h, w) with shapes (D,H,W)

# ---------------- helpers that match the writer ----------------

def flatten_index(coord: Tuple[int, ...], dims: Tuple[int, ...]) -> int:
    """Row-major flattening for arbitrary dims."""
    idx = 0
    for i, c in enumerate(coord):
        stride = 1
        for d in dims[i + 1:]:
            stride *= d
        idx += c * stride
    return idx

def unflatten_index(idx: int, dims: Tuple[int, ...]) -> Tuple[int, ...]:
    """Inverse of flatten_index."""
    coords = []
    for i in range(len(dims) - 1):
        stride = int(np.prod(dims[i + 1:]))
        q, idx = divmod(idx, stride)
        coords.append(q)
    coords.append(idx)
    return tuple(coords)

def decode_board(board_blob: bytes, D: int, H: int, W: int) -> np.ndarray:
    """
    Writer stored board as arr[d,h,w] with dtype int8, values: 0 empty, 1=O, 2=X.
    We preserve that exact memory layout.
    """
    arr = np.frombuffer(board_blob, dtype=np.int8).copy()
    return arr.reshape(D, H, W)

def encode_board(arr: np.ndarray) -> bytes:
    return arr.astype(np.int8, copy=False).tobytes(order="C")

def board_to_strings(arr: np.ndarray) -> List[str]:
    """
    Pretty print by depth layers (d=0..D-1). For 2D boards, D==1.
    Values: 0='.', 1='O', 2='X'
    """
    D, H, W = arr.shape
    out: List[str] = []
    for d in range(D):
        out.append(f" d={d}")
        for h in range(H):
            row = []
            for w in range(W):
                v = int(arr[d, h, w])
                row.append('.' if v == 0 else ('O' if v == 1 else 'X'))
            out.append(" " + " ".join(row))
        out.append("")  # blank line after each layer
    return out

def in_bounds(coord: Tuple[int, int, int], D: int, H: int, W: int) -> bool:
    d, h, w = coord
    return (0 <= d < D) and (0 <= h < H) and (0 <= w < W)

def find_directions_3d() -> List[Coord3]:
    """
    All unique directions in {-1,0,1}^3 \ {(0,0,0)} using a canonical rule:
    keep directions whose first nonzero component is +1.
    """
    dirs: List[Coord3] = []
    for dd in (-1, 0, 1):
        for dh in (-1, 0, 1):
            for dw in (-1, 0, 1):
                if (dd, dh, dw) == (0, 0, 0):
                    continue
                first = next((v for v in (dd, dh, dw) if v != 0), 0)
                if first == +1:
                    dirs.append((dd, dh, dw))
    return dirs

def find_winning_line(board_arr: np.ndarray, x_in_a_row: int) -> Optional[Tuple[int, List[Coord3]]]:
    """
    Return (player, [(d,h,w)*x_in_a_row]) for a winning line, or None if no win.
    board_arr values: 0 empty, 1=O, 2=X.
    """
    D, H, W = board_arr.shape
    dirs = find_directions_3d()

    for d in range(D):
        for h in range(H):
            for w in range(W):
                p = int(board_arr[d, h, w])
                if p == 0:
                    continue
                for dd, dh, dw in dirs:
                    line = [(d, h, w)]
                    cd, ch, cw = d, h, w
                    ok = True
                    for _ in range(1, x_in_a_row):
                        cd += dd; ch += dh; cw += dw
                        if not in_bounds((cd, ch, cw), D, H, W):
                            ok = False; break
                        if int(board_arr[cd, ch, cw]) != p:
                            ok = False; break
                        line.append((cd, ch, cw))
                    if ok:
                        return p, line
    return None

# -------------- DB iteration & reconstruction ------------------

def iter_games(conn: sqlite3.Connection):
    """
    Yields (GameID, (D,H,W), [rows]) grouped & ordered by TurnID.
    Row tuple layout from DB: (TurnID, X_Turn, Board, Win, Moves)
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT GameID, TurnID, X_Turn, Board, Win, Moves, D, H, W "
        "FROM Game ORDER BY GameID ASC, TurnID ASC"
    )
    by_game = defaultdict(list)
    shapes = {}
    for gid, tid, xturn, board, win, moves, D, H, W in cur.fetchall():
        by_game[gid].append((tid, xturn, board, win, moves))
        shapes[gid] = (int(D), int(H), int(W))

    for gid, rows in by_game.items():
        rows.sort(key=lambda r: r[0])
        yield gid, shapes[gid], rows

def reconstruct_final(shape: Tuple[int, int, int], rows) -> Tuple[np.ndarray, Optional[Coord3], int, int]:
    """
    Replays the game and returns:
      final_board_arr[D,H,W], last_move_coord (if inferred), total_turns, winner
    Strategy:
      - We have pre-move boards for each turn. For turns 1..T-1 we can use the
        Moves blob from row t+1 to know move t precisely. We assert the diff.
      - For the final turn T, we infer based on winner and legality.
    """
    D, H, W = shape
    T = len(rows)
    if T == 0:
        return np.zeros((D, H, W), dtype=np.int8), None, 0, 0

    # winner is constant across the game's rows
    winner = int(rows[0][3])  # 0 draw, 1 O, 2 X
    k = H  # your rule: win length equals side length

    board = np.zeros((D, H, W), dtype=np.int8)

    # Apply moves 1..T-1 using "Moves" from row t+1
    for t in range(1, T):  # t is 1-based turn index
        mark = 2 if (t % 2 == 1) else 1  # X on odd turns, O on even

        # row t+1 contains Moves = move at turn t (writer stored int64)
        moves_blob = rows[t][4]  # rows[t] is (t+1)th row
        if not moves_blob:
            # Should not happen for t>=1; fallback to diff vs next pre-move board
            next_pre = decode_board(rows[t][2], D, H, W)
            diff = np.argwhere(board != next_pre)
            if diff.shape[0] != 1:
                raise RuntimeError(f"Cannot infer move t={t}: ambiguous diff size {diff.shape[0]}")
            d, h, w = map(int, diff[0])
        else:
            a = int(np.frombuffer(moves_blob, dtype=np.int64)[0])
            # For 2D games writer used flat_dims=(H,W); for 3D flat_dims=(D,H,W)
            flat_dims = (H, W) if D == 1 else (D, H, W)
            coords = unflatten_index(a, flat_dims)
            if D == 1:
                h, w = coords  # 2D case
                d = 0
            else:
                d, h, w = coords  # 3D case

        if board[d, h, w] != 0:
            raise RuntimeError(f"Illegal replay at t={t}: cell already filled at {(d,h,w)}")

        board[d, h, w] = mark

        # Optional consistency check against the next pre-move board
        next_pre = decode_board(rows[t][2], D, H, W)
        if not np.array_equal(board, next_pre):
            raise RuntimeError(f"Consistency mismatch after t={t}: replay differs from stored pre-board.")

    # Turn T (last) — infer last move
    last_move: Optional[Coord3] = None
    mark_T = 2 if (T % 2 == 1) else 1  # side to move at last row

    pre_last = decode_board(rows[T - 1][2], D, H, W)  # pre-move board for turn T
    if not np.array_equal(board, pre_last):
        # If earlier replay failed to match last row pre-board, trust the DB
        board = pre_last.copy()

    empties = np.argwhere(board == 0)

    if winner == 0:
        # draw: place deterministically into the first empty (if any)
        if empties.shape[0] > 0:
            d, h, w = map(int, empties[0])
            board[d, h, w] = mark_T
            last_move = (d, h, w)
    else:
        # try all empty cells; pick one that yields a winning line for mark_T
        chosen = None
        for (d, h, w) in empties:
            d = int(d); h = int(h); w = int(w)
            board[d, h, w] = mark_T
            wl = find_winning_line(board, x_in_a_row=k)
            if wl and wl[0] == mark_T:
                chosen = (d, h, w)
                last_move = chosen
                break
            board[d, h, w] = 0  # revert
        if chosen is None and empties.shape[0] > 0:
            d, h, w = map(int, empties[0])
            board[d, h, w] = mark_T
            last_move = (d, h, w)

    return board, last_move, T, winner

# ------------------------ CLI preview --------------------------

def iter_preview(conn: sqlite3.Connection, k_games: int, show_boards: bool, only_game: Optional[int]) -> None:
    shown = 0
    for gid, shape, rows in iter_games(conn):
        if only_game is not None and gid != only_game:
            continue
        if only_game is None and shown >= k_games:
            break

        D, H, W = shape
        board, last_move, total_turns, winner = reconstruct_final(shape, rows)
        k = H  # consistent rule

        print(f"\n=== Game {gid} (D/H/W={D}/{H}/{W}, k={k}) ===")
        print(f"  Total turns: {total_turns}")
        print(f"  Winner: {'X' if winner == 2 else ('O' if winner == 1 else 'Draw')}")
        print(f"  Last move: {last_move}")
        if show_boards:
            print("  Final board:")
            for line in board_to_strings(board):
                print("  " + line)

        wl = find_winning_line(board, x_in_a_row=k)
        if wl:
            p, coords = wl
            who = 'X' if p == 2 else 'O'
            print(f"  Winning line for {who}: {coords}")

        shown += 1

def main():
    parser = argparse.ArgumentParser(description="Preview N-in-a-row games from the dataset (supports 2D and 3D).")
    parser.add_argument("--db", required=True, help="Path to SQLite database (dataset.db)")
    parser.add_argument("--preview", action="store_true", help="Print a preview of games")
    parser.add_argument("-k", type=int, default=3, help="How many games to preview (ignored if --game is used)")
    parser.add_argument("--no-boards", action="store_true", help="Do not print final boards")
    parser.add_argument("--game", type=int, help="Preview only this GameID")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    try:
        if args.preview or args.game is not None:
            iter_preview(conn, k_games=args.k, show_boards=not args.no_boards, only_game=args.game)
        else:
            # default action if no flags: show first 3 games
            iter_preview(conn, k_games=3, show_boards=True, only_game=None)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
