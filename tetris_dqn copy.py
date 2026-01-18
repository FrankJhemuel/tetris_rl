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
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU()
        )
        conv_out_size = 64 * input_shape[1] * input_shape[2]
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
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
    H, W = board.shape
    
    # Remove padding - extract playable area only
    playable_board = board[:H-4, 4:W-4]  # Remove bottom 4 rows and side 4 columns
    
    # Channel 1: Board state (0=empty, 1=occupied)
    board_channel = (playable_board > 0).astype(np.float32)
    
    # Channel 2: Piece ID channel (normalized piece type across entire board)
    # Tetromino IDs are 2-8, convert to 0-6 for normalization
    piece_id = env.unwrapped.active_tetromino.id - 2  
    piece_channel = np.full_like(board_channel, piece_id / 6.0)  # Normalize to [0, 1]
    
    # Stack channels: (2, H, W) - board state + piece info
    return np.stack([board_channel, piece_channel], axis=0)

def shape_reward(env, prev_board=None, info=None):
    env = env.unwrapped
    reward = 2.0  # Increased base reward for placing a piece (survival)

    board = env.board.copy()
    H, W = board.shape
    playable_board = board[:H-4, 4:W-4]
    H_p, W_p = playable_board.shape

    # ------------------------
    # Reward for lines cleared (DOMINANT)
    # ------------------------
    lines = info.get("lines_cleared", 0) if info else 0
    reward += lines * 30  # Increased line clearing reward to be more dominant

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
    # Reward based on improvement from previous board (SMALLER IMPACT)
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
        # REDUCED delta weights to prevent overwhelming negative rewards
        delta_holes = prev_holes - holes
        reward += (delta_holes + 0.5) * 2.0  # Reduced weight and smaller baseline

        delta_smooth = prev_smoothness - smoothness
        reward += delta_smooth * 0.1  # Reduced weight

        delta_height = prev_max_height - max_height
        reward += delta_height * 0.5  # Reduced weight

    return reward

def save_training_graph(rewards_per_episode, epsilon_history, lines_per_episode):
    fig = plt.figure(figsize=(18,5))
    window = 50
    
    # Rewards
    plt.subplot(1, 3, 1)
    if len(rewards_per_episode) >= window:
        avg_rewards = np.convolve(rewards_per_episode, np.ones(window)/window, mode='valid')
    else:
        avg_rewards = rewards_per_episode
    plt.plot(range(len(avg_rewards)), avg_rewards)
    plt.xlabel("Episode")
    plt.ylabel(f"Avg Reward (window={window})")
    plt.title("Reward Progression")
    
    # Epsilon
    plt.subplot(1, 3, 2)
    plt.plot(epsilon_history)
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay")
    
    # Lines cleared
    plt.subplot(1, 3, 3)
    if len(lines_per_episode) >= window:
        avg_lines = np.convolve(lines_per_episode, np.ones(window)/window, mode='valid')
    else:
        avg_lines = lines_per_episode
    plt.plot(range(len(avg_lines)), avg_lines)
    plt.xlabel("Episode")
    plt.ylabel(f"Avg Lines Cleared (window={window})")
    plt.title("Lines Cleared Progression")
    
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
    6: [0, 0, 0, -1],   # L
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
            
            # CRITICAL: Convert to the same 2-channel format as get_state_channels()
            H_full, W_full = new_board.shape
            playable_board = new_board[:H_full-4, 4:W_full-4]  # Remove padding
            
            # Channel 1: Board state (0=empty, 1=occupied)
            board_channel = (playable_board > 0).astype(np.float32)
            
            # Channel 2: Piece ID channel (same piece ID as current)
            piece_channel = np.full_like(board_channel, tetro_idx / 6.0)  # Normalize to [0, 1]
            
            # Stack channels to match get_state_channels format
            clean_board = np.stack([board_channel, piece_channel], axis=0)

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
                "board": clean_board,  # Now returns cleaned board format
                "actions": actions
            })

    return results

# ------------------------
# Replay Memory & Networks
# ------------------------
# Create a dummy state with the NEW cleaned dimensions to get the correct state_dim
dummy_env_reset = env.reset()
dummy_state = get_state_channels(env)  # This now returns the cleaned (20x10) board
state_dim = dummy_state.shape
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
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()

# ------------------------
# Main Training Loop (stop at 20 clears)
# ------------------------

CLEARS_GOAL = 60
if __name__ == "__main__":
    # Load checkpoint if it exists
    RESUME_CHECKPOINT = "best_model__.pth"  # or specify a checkpoint like "checkpoints/tetris_dqn_ep1500.pth"
    start_episode = 0
    if os.path.exists(RESUME_CHECKPOINT):
        print(f"Loading checkpoint: {RESUME_CHECKPOINT}")
        checkpoint = torch.load(RESUME_CHECKPOINT, map_location=device)
        policy_net.load_state_dict(checkpoint["policy_state_dict"])
        target_net.load_state_dict(checkpoint["target_state_dict"])
        epsilon = checkpoint.get("epsilon", epsilon)  # Resume from saved epsilon
        start_episode = checkpoint.get("episode", 0)
        print(f"Resumed from episode {start_episode}, epsilon={epsilon:.4f}")
    else:
        print("No checkpoint found, starting from scratch")
    
    best_reward = -float("inf")
    best_lines = 0
    rewards_per_episode = []
    epsilon_history = []
    lines_per_episode = []

    try:
        episode = 0
        reached_goal_clears = False

        while not reached_goal_clears:
            episode += 1
            obs, info = env.reset()
            terminated = False
            total_reward = 0
            prev_board = env.unwrapped.board.copy()
            cumulative_lines = 0

            step_count = 0
            while not terminated:
                tetro = env.unwrapped.active_tetromino
                tetro_idx = tetro.id - 2  # Convert tetromino ID (2-8) to index (0-6)

                placements = get_all_board_states(env, tetro_idx, tetro)
                if not placements:
                    break

                # Get all hypothetical future boards (now 2-channel format)
                hypothetical_boards = np.array([p["board"] for p in placements], dtype=np.float32)
                boards_batch = torch.tensor(hypothetical_boards, dtype=torch.float32).to(device)  # No need to add dimension, already has channels

                with torch.no_grad():
                    q_values = policy_net(boards_batch).squeeze()

                # Pure epsilon-greedy selection (standard DQN)
                if random.random() < epsilon:
                    # Pure random exploration - let the network learn without bias
                    chosen_idx = random.randint(0, len(placements)-1)
                else:
                    # Exploitation - use Q-network's learned strategy
                    chosen_idx = torch.argmax(q_values).item()

                chosen = placements[chosen_idx]
                
                # CRITICAL: Store the board state we CHOSE to transition to
                chosen_board_state = hypothetical_boards[chosen_idx]  # Already has correct shape (2, H, W)

                # Execute actions
                for a in chosen["actions"]:
                    if terminated:
                        break
                    obs, reward_step, terminated, truncated, info = env.step(a)
                    env.render()
                    cv2.waitKey(1)

                reward = shape_reward(env, prev_board, info)
                
                # Add death penalty to emphasize survival value
                if terminated:
                    reward -= 5.0  # Reduced penalty for game over
                
                next_state = get_state_channels(env)
                
                # Store: the board we chose, the reward we got, and the resulting next state
                remember(chosen_board_state, reward, next_state, terminated)
                replay()

                total_reward += reward
                prev_board = env.unwrapped.board.copy()
                step_count += 1

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
                save_training_graph(rewards_per_episode, epsilon_history, lines_per_episode)
                print(f"💾 Checkpoint saved at episode {episode}")

            # Save best model based on lines cleared
            if cumulative_lines > best_lines:
                best_lines = cumulative_lines
                torch.save({
                    "policy_state_dict": policy_net.state_dict(),
                    "target_state_dict": target_net.state_dict(),
                    "epsilon": epsilon,
                    "episode": episode
                }, "best_model.pth")
                print(f"⭐ Saved new best model at Episode {episode} (Lines: {cumulative_lines})")

            rewards_per_episode.append(total_reward)
            epsilon_history.append(epsilon)
            lines_per_episode.append(cumulative_lines)
            print(f"Episode {episode}: Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}, Lines: {cumulative_lines}, Steps: {step_count}")

    finally:
        torch.save({
            "policy_state_dict": policy_net.state_dict(),
            "target_state_dict": target_net.state_dict(),
            "epsilon": epsilon,
            "episode": episode
        }, f"{CHECKPOINT_DIR}/tetris_dqn_latest.pth")
        save_training_graph(rewards_per_episode, epsilon_history, lines_per_episode)
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

#             while not terminated:
#                 curr_tetro = env.unwrapped.active_tetromino
#                 placements = get_all_board_states(env, curr_tetro)

#                 # Convert boards to batch tensor
#                 boards_batch = np.array([p["board"] for p in placements], dtype=np.float32)
#                 boards_batch = torch.tensor(boards_batch[:, np.newaxis, :, :], dtype=torch.float32).to(device)

#                 with torch.no_grad():
#                     q_values = policy_net(boards_batch).squeeze()

#                 if random.random() < epsilon:
#                     chosen_idx = random.randint(0, len(placements)-1)
#                 else:
#                     chosen_idx = torch.argmax(q_values).item()

#                 chosen = placements[chosen_idx]

#                 # Execute actions
#                 for a in chosen["actions"]:
#                     if terminated:
#                         break
#                     obs, reward_step, terminated, truncated, info = env.step(a)
#                     env.render()
#                     cv2.waitKey(50)

#                 reward = shape_reward(env, prev_board, info)
#                 next_state = get_state_channels(env)
#                 remember(get_state_channels(env), reward, next_state, terminated)
#                 replay()
#                 # print(reward)
#                 total_reward += reward
#                 prev_board = env.unwrapped.board.copy()

#             epsilon = max(epsilon_min, epsilon * epsilon_decay)

#             if (episode+1) % target_update_freq == 0:
#                 target_net.load_state_dict(policy_net.state_dict())

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
#             print(f"Episode {episode+1}: Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

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
