import os
import random
import sqlite3
import numpy as np

# ---------------- user config ----------------
DB_PATH = os.path.join(os.path.dirname(__file__), "dataset.db")
NUM_GAMES_PER_SHAPE = 10000          # how many games for each shape below
BATCH_ROWS = 20000                  # insert batch size for speed
SHAPES_2D = [(n, n) for n in (3, 4, 5)]             # 3x3 .. 5x5
SHAPES_3D = [(n, n, n) for n in (3, 4, 5)]          # 3x3x3 .. 5x5x5
ALL_SHAPES = SHAPES_2D + SHAPES_3D
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
# ---------------------------------------------


# ----- minimal N-in-a-row board -----
class Board:
    def __init__(self, dimensions, x_in_a_row):
        self.dimensions = tuple(int(d) for d in dimensions)
        self.nd = len(self.dimensions)
        self.x_in_a_row = int(x_in_a_row)
        self.state = np.zeros(self.dimensions, dtype=np.int8)
        self.turn = 1  # 1=X, 2=O
        self._dirs = self._make_dirs()

    def _make_dirs(self):
        # unique directions from {-1,0,1}^nd \ {0}, keeping those whose first nonzero > 0
        from itertools import product
        dirs = []
        for v in product((-1, 0, 1), repeat=self.nd):
            if all(c == 0 for c in v):
                continue
            # canonical filter
            keep = False
            for c in v:
                if c < 0:
                    keep = False
                    break
                if c > 0:
                    keep = True
                    break
            if keep:
                dirs.append(tuple(v))
        return dirs

    def possible_moves(self):
        return np.argwhere(self.state == 0).astype(np.int64, copy=False)

    def push(self, coord):
        pos = tuple(int(p) for p in coord)
        if self.state[pos] != 0:
            raise ValueError("occupied")
        self.state[pos] = np.int8(self.turn)
        self.turn = 2 if self.turn == 1 else 1

    def _won(self, p):
        k, dims, A = self.x_in_a_row, self.dimensions, self.state
        for v in self._dirs:
            # start ranges per axis
            starts = []
            for i, d in enumerate(dims):
                step = v[i]
                if step == 0:
                    starts.append(range(0, d))
                elif step == 1:
                    starts.append(range(0, d - k + 1))
                else:
                    starts.append(range(k - 1, d))
            for idx in np.ndindex(*(len(r) for r in starts)):
                s = [starts[i][idx[i]] for i in range(self.nd)]
                ok = True
                for t in range(k):
                    q = tuple(s[i] + t * v[i] for i in range(self.nd))
                    if A[q] != p:
                        ok = False
                        break
                if ok:
                    return True
        return False

    def result(self):
        if self._won(1):
            return 2  # dataset encoding uses X=2
        if self._won(2):
            return 1  # and O=1
        if (self.state == 0).any():
            return None
        return 0

    def board_blob(self):
        # convert internal 0/1/2 to dataset encoding X=2, O=1
        arr = np.zeros_like(self.state, dtype=np.int8)
        it = np.nditer(self.state, flags=['multi_index'])
        while not it.finished:
            v = int(it[0])
            if v == 1:
                arr[it.multi_index] = 2
            elif v == 2:
                arr[it.multi_index] = 1
            it.iternext()
        return arr.tobytes(order="C")


# ----- indexing helpers -----
def flatten_index(move, dims):
    idx = 0
    for i, c in enumerate(move):
        stride = 1
        for d in dims[i + 1:]:
            stride *= d
        idx += c * stride
    return idx

def unflatten_index(idx, dims):
    coords = []
    for i in range(len(dims) - 1):
        stride = int(np.prod(dims[i + 1:]))
        q, idx = divmod(idx, stride)
        coords.append(q)
    coords.append(idx)
    return tuple(coords)


# ----- self-play (random) -----
def play_random(dimensions, k):
    b = Board(dimensions, k)
    seq = []
    dims = b.dimensions
    while True:
        moves = b.possible_moves()
        if moves.size == 0:
            return 0, seq
        m = tuple(map(int, random.choice(moves)))
        seq.append(flatten_index(m, dims))
        b.push(m)
        r = b.result()
        if r is not None:
            return r, seq


# ----- row builder -----
def rows_for_game(gid, win_val, moves_idx, dims, k):
    rows = []
    # normalize dims to (D,H,W) for storage (2D -> D=1, H,W)
    if len(dims) == 2:
        D, H, W = 1, dims[0], dims[1]
        flat_dims = (H, W)
    else:
        D, H, W = dims
        flat_dims = (D, H, W)

    b = Board(dimensions=dims, x_in_a_row=k)
    for t, a in enumerate(moves_idx, start=1):
        x_turn = 1 if (t % 2 == 1) else 0
        prev_blob = b"" if t == 1 else np.asarray([moves_idx[t - 2]], dtype=np.int64).tobytes()
        board_blob = b.board_blob()  # pre-move board
        rows.append((gid, t, x_turn, board_blob, win_val, prev_blob, D, H, W))
        # advance
        coord = unflatten_index(int(a), flat_dims)
        b.push(coord)
    return rows


# ----- main -----
def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # continue GameID
    cur.execute("SELECT COALESCE(MAX(GameID), 0) FROM Game;")
    next_gid = int(cur.fetchone()[0]) + 1

    batch, total_rows, games = [], 0, 0

    def flush():
        nonlocal batch, total_rows
        if not batch:
            return
        cur.executemany(
            "INSERT INTO Game (GameID, TurnID, X_Turn, Board, Win, Moves, D, H, W) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            batch
        )
        conn.commit()
        total_rows += len(batch)
        batch.clear()

    for dims in ALL_SHAPES:
        side = dims[0]  # n in n×n or n×n×n
        k = side        # win length equals side length
        for _ in range(NUM_GAMES_PER_SHAPE):
            win_val, moves_idx = play_random(dims, k)
            gid = next_gid
            next_gid += 1
            games += 1
            batch.extend(rows_for_game(gid, win_val, moves_idx, dims, k))
            if len(batch) >= BATCH_ROWS:
                flush()

    flush()
    conn.close()
    print(f"Done. Inserted {total_rows} rows across {games} games for {len(ALL_SHAPES)} shapes.")

if __name__ == "__main__":
    main()
