import numpy as np
from tetris_gymnasium.envs.tetris import Tetris
from tetris_gymnasium.mappings.actions import ActionsMapping
import cv2
import time

A = ActionsMapping()

# ------------------------
# Helper functions
# ------------------------
def trim_piece(piece):
    rows = np.any(piece, axis=1)
    cols = np.any(piece, axis=0)
    return piece[np.ix_(rows, cols)]

def get_unique_rotations(tetromino):
    rotations = []
    seen = []
    mat = tetromino.matrix.copy()
    for _ in range(4):
        trimmed = trim_piece(mat)
        if not any(np.array_equal(trimmed, s) for s in seen):
            seen.append(trimmed)
            rotations.append(trimmed)
        mat = np.rot90(mat, -1)
    return rotations

def piece_bounds(piece):
    cols = np.any(piece, axis=0)
    left = np.argmax(cols)
    right = len(cols) - 1 - np.argmax(cols[::-1])
    return left, right

def check_collision(board, piece, x, y):
    H, W = board.shape
    h, w = piece.shape
    for py in range(h):
        for px in range(w):
            if piece[py, px]:
                bx = x + px
                by = y + py
                if bx < 0 or bx >= W or by < 0 or by >= H:
                    return True
                if board[by, bx]:
                    return True
    return False

def place_piece(board, piece, x, y):
    new_board = board.copy()
    h, w = piece.shape
    for py in range(h):
        for px in range(w):
            if piece[py, px]:
                new_board[y + py, x + px] = 1
    return new_board

# Offsets to convert trimmed-x → env-x
# Format: piece_index : [offset_per_rotation]
X_OFFSETS = {
    0: [0, -1],         # I (horizontal, vertical)
    1: [0],             # O
    2: [0, 0, 0, -1],   # T
    3: [0, 0],          # S
    4: [0, 0],          # Z
    5: [0, 0, 0, -1],   # J
    6: [0, 0, 0, -1],    # L
}


def compute_action_sequence(env, tetro_idx, tetro, target_rot_id, target_x):
    actions = []
    rotations = get_unique_rotations(tetro)

    # --- Rotation ---
    curr_mat = trim_piece(tetro.matrix.copy())
    curr_rot_id = 0
    for i, r in enumerate(rotations):
        if np.array_equal(curr_mat, r):
            curr_rot_id = i
            break

    n_rot = (target_rot_id - curr_rot_id) % len(rotations)
    actions += [A.rotate_clockwise] * n_rot

    # --- Apply per-piece per-rotation offset ---
    offset = X_OFFSETS[tetro_idx][target_rot_id]
    corrected_target_x = target_x + offset

    start_x = env.unwrapped.x
    dx = corrected_target_x - start_x

    if dx > 0:
        actions += [A.move_right] * dx
    elif dx < 0:
        actions += [A.move_left] * (-dx)

    actions.append(A.hard_drop)
    return actions


def get_all_board_states(env, tetro_idx, tetro):
    board = env.unwrapped.board.copy()
    H, W = board.shape
    rotations = get_unique_rotations(tetro)
    results = []

    pad = 4
    playable_x_min = pad
    playable_x_max = W - pad

    for rot_id, piece in enumerate(rotations):
        h, w = piece.shape
        left, right = piece_bounds(piece)

        # x is the LEFTMOST occupied column in board coords
        min_x = playable_x_min - left
        max_x = playable_x_max - (right + 1)

        for x in range(min_x, max_x + 1):
            y = 0
            while not check_collision(board, piece, x, y):
                y += 1
            y -= 1
            if y < 0:
                continue

            new_board = place_piece(board, piece, x, y)

            actions = compute_action_sequence(
                env,
                tetro_idx,  
                tetro,
                rot_id,
                x
            )

            results.append({
                "rotation": rot_id,
                "x": x,
                "y": y,
                "board": new_board,
                "actions": actions
            })

    return results

# ------------------------
# Save placements to file
# ------------------------
def save_placements_to_file(placements, tetro_idx):
    filename = f"tetro_{tetro_idx}_placements.txt"
    with open(filename, "w") as f:
        for p in placements:
            f.write(f"=== Rotation {p['rotation']} ===\n")
            f.write(f"Placed at x={p['x']}, y={p['y']}:\n")
            
            # Convert board to text using 1 and .
            board_str = ""
            for row in p['board']:
                board_str += " ".join(str(int(cell)) if cell else '.' for cell in row) + "\n"
            f.write(board_str)
            
            f.write(f"Actions: {p['actions']}\n")
            f.write("-" * 30 + "\n")
    print(f"Placements saved to {filename}")

# ------------------------
# Main
# ------------------------
env = Tetris(render_mode="rgb_array")
env.reset(seed=42)

# Select tetromino
tetro_idx = int(input("Enter tetromino index (0-6): "))
tetro = env.unwrapped.tetrominoes[tetro_idx]

# Compute all placements
placements = get_all_board_states(env, tetro_idx, tetro)

# Save all placements to file
save_placements_to_file(placements, tetro_idx)

placement_idx = int(input(f"Enter placement index (0-{len(placements)-1}): "))
p = placements[placement_idx]

# --- SIMULATE FIRST BLOCK ---
# Manually set the first piece to the chosen tetromino
env.unwrapped.active_tetromino = tetro

# Step through precomputed actions
for a in p['actions']:
    obs, reward, terminated, truncated, info = env.step(a)

    # OpenCV rendering
    frame = env.render()
    if frame is not None:
        frame = cv2.resize(frame, (frame.shape[1]*4, frame.shape[0]*4), interpolation=cv2.INTER_NEAREST)
        cv2.putText(frame, f"Placement idx: {placement_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imshow("Tetris Simulation", frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    if terminated or truncated:
        break

input("Press Enter to close...")
cv2.destroyAllWindows()
env.close()
