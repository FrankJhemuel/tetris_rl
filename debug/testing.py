import sys
import os
# Add parent directory to path for imports when in debug folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tetris_gymnasium.envs.tetris import Tetris
from tetris_gymnasium.mappings.actions import ActionsMapping
from tetris_dqn import extract_features
import cv2
import time

A = ActionsMapping()

# ------------------------
# Helper functions from placement script
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
    0: [0, -2],         # I (horizontal, vertical)
    1: [0],             # O
    2: [0, -1, 0, 0],   # T
    3: [0, -1],          # S
    4: [0, -1],          # Z
    5: [0, -1, 0, 0],   # J
    6: [0, -1, 0, 0],   # L
}


def compute_action_sequence(env, tetro_idx, tetro, target_rot_id, target_x):
    actions = []
    
    # The piece always spawns at rotation 0, so target_rot_id directly tells us
    # how many clockwise rotations to perform
    n_rot = target_rot_id
    actions += [A.rotate_counterclockwise] * n_rot

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
    
    # CRITICAL FIX: Clear any full lines in the board snapshot FIRST
    # The environment may not have cleared lines from the previous piece yet
    playable_snapshot = board[:H-4, 4:W-4]
    rows_to_clear = []
    for row in range(playable_snapshot.shape[0]):
        if np.all(playable_snapshot[row] > 0):
            rows_to_clear.append(row)
    
    if rows_to_clear:
        # Remove full rows and add empty rows at top
        playable_snapshot = np.delete(playable_snapshot, rows_to_clear, axis=0)
        empty_rows = np.zeros((len(rows_to_clear), playable_snapshot.shape[1]), dtype=playable_snapshot.dtype)
        playable_snapshot = np.vstack([empty_rows, playable_snapshot])
        # Reconstruct full board with padding
        board = np.zeros((H, W), dtype=board.dtype)
        board[:H-4, 4:W-4] = playable_snapshot
    
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
            
            # Extract playable area for feature calculation
            H_full, W_full = new_board.shape
            playable_board = new_board[:H_full-4, 4:W_full-4]  # Remove padding
            
            # Calculate lines that would be cleared AND simulate line clearing
            lines_cleared = 0
            rows_to_clear = []
            for row in range(playable_board.shape[0]):
                if np.all(playable_board[row] > 0):
                    lines_cleared += 1
                    rows_to_clear.append(row)
            
            # Simulate line clearing: remove full rows and add empty rows at top
            if rows_to_clear:
                playable_board_after_clear = np.delete(playable_board, rows_to_clear, axis=0)
                empty_rows = np.zeros((lines_cleared, playable_board.shape[1]), dtype=playable_board.dtype)
                playable_board = np.vstack([empty_rows, playable_board_after_clear])

            # Extract features
            features = extract_features(playable_board)
            # Create 5-feature vector
            feature_vector = np.array([
                features['aggregate_height'] / 200.0,  # 0-200 → 0-1
                features['complete_lines'] / 4.0,      # 0-4 → 0-1
                features['holes'] / 10.0,              # 0-10 → 0-1
                features['min_height'] / 20.0,         # 0-20 → 0-1
                features['max_height'] / 20.0          # 0-20 → 0-1
            ], dtype=np.float32)
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
                "features": feature_vector,  # Now returns feature vector instead of board
                "actions": actions,
                "board": playable_board  # For debugging
            })

    return results

# ------------------------
# Save placements to file
# ------------------------
def save_placements_to_file(placements, tetro_idx):
    # Create placements directory if it doesn't exist
    os.makedirs("placements", exist_ok=True)
    
    filename = f"placements/tetro_{tetro_idx}_placements.txt"
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
