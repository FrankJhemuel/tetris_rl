import cv2
import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

from tetris_gymnasium.envs.tetris import Tetris
from tetris_gymnasium.mappings.actions import ActionsMapping

# ------------------------
# Directories & Config
# ------------------------
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
GRAPH_FILE = "tetris_training_graph.png"
SAVE_EVERY = 100
MAX_CHECKPOINTS = 10

with open("hyperparams.yaml", "r") as f:
    config = yaml.safe_load(f)

gamma = config["gamma"]
epsilon = config["epsilon"]
epsilon_decay = config["epsilon_decay"]
epsilon_min = config["epsilon_min"]
lr = config["lr"]
batch_size = config["batch_size"]
memory_size = config["memory_size"]
target_update_freq = config["target_update_freq"]
num_episodes = config["num_episodes"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

A = ActionsMapping()

# ------------------------
# DQN Network
# ------------------------
class DQN(nn.Module):
    def __init__(self, input_shape):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU()
        )
        conv_out_size = 64 * input_shape[1] * input_shape[2]
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ------------------------
# Environment
# ------------------------
env = gym.make("tetris_gymnasium/Tetris", render_mode=None)
obs, info = env.reset()

# ------------------------
# Helper Functions
# ------------------------
def cleanup_checkpoints(folder, max_keep):
    files = sorted(
        [f for f in os.listdir(folder) if f.endswith(".pth")],
        key=lambda f: os.path.getmtime(os.path.join(folder, f))
    )
    while len(files) > max_keep:
        old_file = files.pop(0)
        os.remove(os.path.join(folder, old_file))
        print(f"🧹 Removed old checkpoint: {old_file}")

def piece_placed(prev_board, env):
    current_board = env.unwrapped.board
    H, W = current_board.shape
    prev_playable = prev_board[:H-4, 4:W-4]
    curr_playable = current_board[:H-4, 4:W-4]
    return not np.array_equal(prev_playable, curr_playable)

def get_state_channels(env: Tetris) -> np.ndarray:
    board = env.unwrapped.board.astype(np.float32)
    return board[np.newaxis, :, :]

def shape_reward(env, prev_board=None, info=None):
    env = env.unwrapped
    reward = 1.0  # base reward for placing a piece (survival)

    board = env.board.copy()
    H, W = board.shape
    playable_board = board[:H-4, 4:W-4]
    H_p, W_p = playable_board.shape

    # ------------------------
    # Reward for lines cleared
    # ------------------------
    lines = info.get("lines_cleared", 0) if info else 0
    reward += lines * 50  # line clearing is the main goal

    # ------------------------
    # Compute holes and column heights
    # ------------------------
    holes = 0
    column_heights = []
    for c in range(W_p):
        col = playable_board[:, c]
        found_block = False
        for row in range(H_p):
            if col[row] > 0:
                found_block = True
            elif found_block and col[row] == 0:
                holes += 1
        filled_rows = np.where(col > 0)[0]
        height = H_p - filled_rows[0] if len(filled_rows) > 0 else 0
        column_heights.append(height)

    smoothness = sum(abs(column_heights[i] - column_heights[i+1]) for i in range(W_p-1))
    max_height = max(column_heights)

    # ------------------------
    # Reward based on improvement from previous board
    # ------------------------
    if prev_board is not None:
        prev_playable = prev_board[:H-4, 4:W-4]

        prev_holes = 0
        prev_column_heights = []
        for c in range(W_p):
            col = prev_playable[:, c]
            found_block = False
            for row in range(H_p):
                if col[row] > 0:
                    found_block = True
                elif found_block and col[row] == 0:
                    prev_holes += 1
            filled_rows = np.where(col > 0)[0]
            height = H_p - filled_rows[0] if len(filled_rows) > 0 else 0
            prev_column_heights.append(height)

        prev_smoothness = sum(abs(prev_column_heights[i] - prev_column_heights[i+1]) for i in range(W_p-1))
        prev_max_height = max(prev_column_heights)

        # ------------------------
        # Holes delta: positive if fewer holes than previous
        delta_holes = prev_holes - holes
        # shift so that zero holes is optimal (+1)
        reward += (delta_holes + 1.0) * 1.0  # weight can be tuned

        # Smoothness delta: positive if smoother than previous
        delta_smooth = prev_smoothness - smoothness
        reward += delta_smooth * 1.0  # smaller effect

        # Max height delta: negative if the max height increased
        delta_height = prev_max_height - max_height
        reward += delta_height * 1.0  # reward lowering max height

    return reward

def save_training_graph(rewards_per_episode, epsilon_history):
    fig = plt.figure(figsize=(12,5))
    window = 50
    if len(rewards_per_episode) >= window:
        avg_rewards = np.convolve(rewards_per_episode, np.ones(window)/window, mode='valid')
    else:
        avg_rewards = rewards_per_episode
    plt.subplot(1, 2, 1)
    plt.plot(range(len(avg_rewards)), avg_rewards)
    plt.xlabel("Episode")
    plt.ylabel(f"Avg Reward (window={window})")
    plt.title("Reward Progression")
    plt.subplot(1, 2, 2)
    plt.plot(epsilon_history)
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay")
    plt.tight_layout()
    plt.savefig(GRAPH_FILE)
    plt.close(fig)

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
# Replay Memory & Networks
# ------------------------
dummy_state = np.array(env.unwrapped.board, dtype=np.float32)
state_dim = (1, *dummy_state.shape)
policy_net = DQN(state_dim).to(device)
target_net = DQN(state_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=lr)
loss_fn = nn.MSELoss()
memory = deque(maxlen=memory_size)

def remember(state, reward, next_state, done):
    memory.append((state, reward, next_state, done))

def replay():
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    states, rewards, next_states, dones = zip(*batch)
    states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)
    q_values = policy_net(states).squeeze()
    next_q_values = target_net(next_states).squeeze()
    target = rewards + gamma * next_q_values * (1 - dones)
    loss = loss_fn(q_values, target.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ------------------------
# Main Training Loop (stop at 20 clears)
# ------------------------

CLEARS_GOAL = 20
if __name__ == "__main__":
    best_reward = -float("inf")
    rewards_per_episode = []
    epsilon_history = []

    try:
        episode = 0
        reached_goal_clears = False

        while not reached_goal_clears:
            episode += 1
            obs, info = env.reset()
            terminated = False
            total_reward = 0
            prev_board = env.unwrapped.board.copy()
            cumulative_lines = 0  # Track lines cleared in this game

            while not terminated:
                tetro = env.unwrapped.active_tetromino
                tetro_idx = env.unwrapped.tetrominoes.index(tetro)

                placements = get_all_board_states(env, tetro_idx, tetro)
                if not placements:  # No valid moves
                    break

                # Convert boards to batch tensor
                boards_batch = np.array([p["board"] for p in placements], dtype=np.float32)
                boards_batch = torch.tensor(boards_batch[:, np.newaxis, :, :], dtype=torch.float32).to(device)

                with torch.no_grad():
                    q_values = policy_net(boards_batch).squeeze()

                if random.random() < epsilon:
                    chosen_idx = random.randint(0, len(placements)-1)
                else:
                    chosen_idx = torch.argmax(q_values).item()

                chosen = placements[chosen_idx]

                # Execute actions
                for a in chosen["actions"]:
                    if terminated:
                        break
                    obs, reward_step, terminated, truncated, info = env.step(a)
                    env.render()
                    cv2.waitKey(50)

                reward = shape_reward(env, prev_board, info)
                state = prev_board[np.newaxis, :, :].astype(np.float32)
                next_state = get_state_channels(env)

                remember(state, reward, next_state, terminated)

                replay()


                total_reward += reward
                prev_board = env.unwrapped.board.copy()

                # Update cumulative lines
                if info and "lines_cleared" in info:
                    cumulative_lines += info["lines_cleared"]

                if cumulative_lines >= CLEARS_GOAL:
                    print(f"🎯 Reached {CLEARS_GOAL} line clears in Episode {episode}!")
                    reached_goal_clears = True
                    break

            # Decay epsilon
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            if (episode) % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if (episode) % SAVE_EVERY == 0:
                checkpoint_path = f"{CHECKPOINT_DIR}/tetris_dqn_ep{episode}.pth"
                torch.save({
                    "policy_state_dict": policy_net.state_dict(),
                    "target_state_dict": target_net.state_dict(),
                    "epsilon": epsilon,
                    "episode": episode
                }, checkpoint_path)
                cleanup_checkpoints(CHECKPOINT_DIR, MAX_CHECKPOINTS)
                save_training_graph(rewards_per_episode, epsilon_history)

            # Save best reward
            if total_reward > best_reward:
                best_reward = total_reward
                torch.save({
                    "policy_state_dict": policy_net.state_dict(),
                    "target_state_dict": target_net.state_dict(),
                    "epsilon": epsilon,
                    "episode": episode
                }, "best_model.pth")
                print(f"⭐ Saved new best model at Episode {episode}")

            rewards_per_episode.append(total_reward)
            epsilon_history.append(epsilon)
            print(f"Episode {episode}: Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}, Lines: {cumulative_lines}")

    finally:
        torch.save({
            "policy_state_dict": policy_net.state_dict(),
            "target_state_dict": target_net.state_dict(),
            "epsilon": epsilon,
            "episode": episode
        }, f"{CHECKPOINT_DIR}/tetris_dqn_latest.pth")
        save_training_graph(rewards_per_episode, epsilon_history)
        env.close()
        print("✅ Latest model saved and graph generated")


# # ------------------------
# # Main Training Loop
# # ------------------------
# if __name__ == "__main__":
#     best_reward = -float("inf")
#     rewards_per_episode = []
#     epsilon_history = []

#     try:
#         for episode in range(num_episodes):
#             obs, info = env.reset()
#             terminated = False
#             total_reward = 0
#             prev_board = env.unwrapped.board.copy()
#             cumulative_lines = 0

#             while not terminated:
#                 # --- Get current tetromino ---
#                 tetro = env.unwrapped.active_tetromino
#                 tetro_idx = env.unwrapped.tetrominoes.index(tetro)

#                 # --- Enumerate all possible placements ---
#                 placements = get_all_board_states(env, tetro_idx, tetro)
#                 if not placements:  # No valid moves
#                     break

#                 # --- Convert boards to tensor for policy evaluation ---
#                 boards_batch = np.array([p["board"] for p in placements], dtype=np.float32)
#                 boards_batch = torch.tensor(boards_batch[:, np.newaxis, :, :], dtype=torch.float32).to(device)

#                 with torch.no_grad():
#                     q_values = policy_net(boards_batch).squeeze()

#                 # --- Epsilon-greedy selection ---
#                 if random.random() < epsilon:
#                     chosen_idx = random.randint(0, len(placements)-1)
#                 else:
#                     chosen_idx = torch.argmax(q_values).item()

#                 chosen = placements[chosen_idx]

#                 # --- Execute actions for the chosen placement ---
#                 for a in chosen["actions"]:
#                     if terminated:
#                         break
#                     obs, reward_step, terminated, truncated, info = env.step(a)
#                     env.render()
#                     cv2.waitKey(50)

#                 # --- Compute shaped reward ---
#                 reward = shape_reward(env, prev_board, info)

#                 # --- Store transition in memory ---
#                 state = prev_board[np.newaxis, :, :].astype(np.float32)
#                 next_state = get_state_channels(env)
#                 remember(state, reward, next_state, terminated)

#                 # --- Replay only if memory has enough transitions ---
#                 if len(memory) > batch_size * 5:
#                     replay()

#                 total_reward += reward
#                 prev_board = env.unwrapped.board.copy()

#                 # --- Update cumulative lines if needed ---
#                 if info and "lines_cleared" in info:
#                     cumulative_lines += info["lines_cleared"]

#             # --- Epsilon decay ---
#             epsilon = max(epsilon_min, epsilon * epsilon_decay)

#             # --- Update target network periodically ---
#             if (episode+1) % target_update_freq == 0:
#                 target_net.load_state_dict(policy_net.state_dict())

#             # --- Save checkpoints periodically ---
#             if (episode+1) % SAVE_EVERY == 0:
#                 checkpoint_path = f"{CHECKPOINT_DIR}/tetris_dqn_ep{episode+1}.pth"
#                 torch.save({
#                     "policy_state_dict": policy_net.state_dict(),
#                     "target_state_dict": target_net.state_dict(),
#                     "epsilon": epsilon,
#                     "episode": episode+1
#                 }, checkpoint_path)
#                 cleanup_checkpoints(CHECKPOINT_DIR, MAX_CHECKPOINTS)
#                 save_training_graph(rewards_per_episode, epsilon_history)

#             # --- Save best model ---
#             if total_reward > best_reward:
#                 best_reward = total_reward
#                 torch.save({
#                     "policy_state_dict": policy_net.state_dict(),
#                     "target_state_dict": target_net.state_dict(),
#                     "epsilon": epsilon,
#                     "episode": episode+1
#                 }, "best_model.pth")
#                 print(f"⭐ Saved new best model at Episode {episode+1}")

#             rewards_per_episode.append(total_reward)
#             epsilon_history.append(epsilon)
#             print(f"Episode {episode+1}: Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}, Lines: {cumulative_lines}")

#     finally:
#         torch.save({
#             "policy_state_dict": policy_net.state_dict(),
#             "target_state_dict": target_net.state_dict(),
#             "epsilon": epsilon,
#             "episode": episode
#         }, f"{CHECKPOINT_DIR}/tetris_dqn_latest.pth")
#         save_training_graph(rewards_per_episode, epsilon_history)
#         env.close()
#         print("✅ Latest model saved and graph generated")
