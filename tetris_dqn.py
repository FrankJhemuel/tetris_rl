import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tetris_gymnasium.envs.tetris import Tetris
import yaml

# ------------------------
# Directories
# ------------------------
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
GRAPH_FILE = "tetris_training_graph.png"
SAVE_EVERY = 100
MAX_CHECKPOINTS = 10

# ------------------------
# Hyperparameters
# ------------------------
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

# ------------------------
# Device
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
            nn.Linear(512, 1)  # single output: board evaluation
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
    env = env.unwrapped
    board = env.board.astype(np.float32)
    H, W = board.shape
    state = np.zeros((3, H, W), dtype=np.float32)

    # Current tetromino
    state[0] = env.project_tetromino().astype(np.float32)

    # Next tetromino
    next_id = env.queue.get_queue()[0]
    if next_id is not None:
        next_tet = env.tetrominoes[next_id]
        next_board = np.zeros_like(board)
        next_board[:next_tet.matrix.shape[0], :next_tet.matrix.shape[1]] = next_tet.matrix
        state[1] = next_board

    # Hold tetromino
    hold_list = env.holder.get_tetrominoes()
    if hold_list:
        hold_tet = hold_list[0]
        hold_board = np.zeros_like(board)
        hold_board[:hold_tet.matrix.shape[0], :hold_tet.matrix.shape[1]] = hold_tet.matrix
        state[2] = hold_board

    return state

def shape_reward(env, prev_board=None):
    """
    Compute reward based on lines cleared, holes, and smoothness.
    """
    env = env.unwrapped
    reward = 0.05  # survival reward
    board = env.board.copy()
    H, W = board.shape
    playable_board = board[:H-4, 4:W-4]
    H_p, W_p = playable_board.shape

    # Lines cleared
    lines = env.lines_cleared
    reward += lines * 15

    # Bonus for placing piece
    if prev_board is not None and piece_placed(prev_board, env):
        reward += 0.2

    # Holes
    holes = 0
    for col in range(W_p):
        found_block = False
        for row in range(H_p):
            if playable_board[row, col] > 0:
                found_block = True
            elif found_block and playable_board[row, col] == 0:
                holes += 1

    # Smoothness
    column_heights = []
    for c in range(W_p):
        col = playable_board[:, c]
        filled_rows = np.where(col > 0)[0]
        height = H_p - filled_rows[0] if len(filled_rows) > 0 else 0
        column_heights.append(height)
    smoothness = sum(abs(column_heights[i] - column_heights[i+1]) for i in range(W_p-1))

    # Delta rewards
    if prev_board is not None and piece_placed(prev_board, env):
        prev_playable = prev_board[:H-4, 4:W-4]
        prev_holes = 0
        for col in range(W_p):
            found_block = False
            for row in range(H_p):
                if prev_playable[row, col] > 0:
                    found_block = True
                elif found_block and prev_playable[row, col] == 0:
                    prev_holes += 1

        prev_column_heights = []
        for c in range(W_p):
            col = prev_playable[:, c]
            filled_rows = np.where(col > 0)[0]
            height = H_p - filled_rows[0] if len(filled_rows) > 0 else 0
            prev_column_heights.append(height)
        prev_smoothness = sum(abs(prev_column_heights[i] - prev_column_heights[i+1]) for i in range(W_p-1))

        delta_holes = np.clip(prev_holes - holes, -2.0, 2.0)
        delta_smooth = np.clip(prev_smoothness - smoothness, -2.0, 2.0)
        reward += delta_holes + delta_smooth

    return reward

def save_moving_avg_graph(rewards_per_episode, window=50, filename="tetris_moving_avg.png"):
    rewards_array = np.array(rewards_per_episode)
    if len(rewards_array) >= window:
        moving_avg = np.convolve(rewards_array, np.ones(window)/window, mode='valid')
        plt.figure(figsize=(10,5))
        plt.plot(range(window-1, len(rewards_array)), moving_avg, color='blue')
        plt.xlabel("Episode")
        plt.ylabel(f"Moving Avg Reward (window={window})")
        plt.title("Tetris Reward Moving Average")
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

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
# Networks and Replay
# ------------------------
dummy_state = env.unwrapped.board.astype(np.float32)
state_dim = (1, *dummy_state.shape)  # 1 channel now
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
# Position Evaluation
# ------------------------
def simulate_swap(env):
    """Return a copy of the unwrapped environment with the active tetromino swapped."""
    temp_env = copy.deepcopy(env.unwrapped)
    if not temp_env.has_swapped:
        swapped = temp_env.holder.swap(temp_env.active_tetromino)
        if swapped is not None:
            temp_env.active_tetromino = swapped
            temp_env.has_swapped = True
    return temp_env

def get_all_final_positions(env):
    """Get all final positions (x, rotation, hold) for the current tetromino."""
    env_unwrapped = env.unwrapped
    positions = []

    hold_options = [False]
    if not env_unwrapped.has_swapped:
        hold_options.append(True)

    for hold in hold_options:
        temp_env = copy.deepcopy(env_unwrapped)

        if hold:
            if not temp_env.has_swapped:
                swapped = temp_env.holder.swap(temp_env.active_tetromino)
                if swapped is None:
                    continue
                temp_env.active_tetromino = swapped
                temp_env.has_swapped = True

        if temp_env.active_tetromino is None:
            continue

        for rotation in range(4):
            tet = temp_env.active_tetromino
            for _ in range(rotation):
                tet = temp_env.rotate(tet, clockwise=True)

            if tet is None:
                continue

            tet_width = tet.matrix.shape[1]
            for x in range(0, temp_env.width - tet_width + 1):
                y = temp_env.y
                while not temp_env.collision(tet, x, y + 1):
                    y += 1
                positions.append((x, rotation, hold))

    return positions

def evaluate_positions(env, policy_net):
    env_unwrapped = env.unwrapped
    positions = get_all_final_positions(env)
    best_score = -float("inf")
    best_move = None

    for x, rotation, hold in positions:
        temp_env = copy.deepcopy(env_unwrapped)
        if hold:
            swapped = temp_env.holder.swap(temp_env.active_tetromino)
            if swapped is not None:
                temp_env.active_tetromino = swapped
                temp_env.has_swapped = True

        tet = temp_env.active_tetromino
        if tet is None:
            continue
        for _ in range(rotation):
            tet = temp_env.rotate(tet, clockwise=True)
        temp_env.active_tetromino = tet
        temp_env.x = x

        # Hard drop
        temp_env.drop_active_tetromino()

        # Get board after drop
        board_obs = temp_env._get_obs()["board"]
        board_tensor = torch.tensor(temp_env.unwrapped.board.astype(np.float32), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) # shape = (1, 1, H, W)


        with torch.no_grad():
            score = policy_net(board_tensor).item()

        if score > best_score:
            best_score = score
            best_move = (x, rotation, hold)

    return best_move

def execute_position(env: Tetris, position):
    x_target, rotation_target, hold = position
    env_unwrapped = env.unwrapped

    if hold:
        env.swap()

    # Rotate to target
    current_rotation = 0
    while current_rotation != rotation_target:
        env.step(env_unwrapped.actions.rotate_clockwise)
        current_rotation = (current_rotation + 1) % 4

    # Move horizontally
    while env_unwrapped.x < x_target:
        env.step(env_unwrapped.actions.move_right)
    while env_unwrapped.x > x_target:
        env.step(env_unwrapped.actions.move_left)

    # Hard drop
    env.step(env_unwrapped.actions.hard_drop)


# ------------------------
# Training Loop
# ------------------------
if __name__ == "__main__":
    best_reward = -float("inf")
    rewards_per_episode = []
    epsilon_history = []

    try:
        for episode in range(num_episodes):
            obs, info = env.reset()
            total_reward = 0
            terminated = False
            prev_board = env.unwrapped.board.copy()

            while not terminated:
                # Evaluate all positions
                best_pos = evaluate_positions(env, policy_net)

                # Execute best position
                execute_position(env, best_pos)

                # Compute reward
                reward = shape_reward(env, prev_board)
                next_state = get_state_channels(env)
                remember(get_state_channels(env), reward, next_state, terminated)
                replay()

                total_reward += reward
                prev_board = env.unwrapped.board.copy()

                # Check termination
                terminated = env.unwrapped.done

            # Epsilon decay
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            # Update target network
            if (episode+1) % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # Checkpoints
            if (episode+1) % SAVE_EVERY == 0:
                checkpoint_path = f"{CHECKPOINT_DIR}/tetris_dqn_ep{episode+1}.pth"
                torch.save({
                    "policy_state_dict": policy_net.state_dict(),
                    "target_state_dict": target_net.state_dict(),
                    "epsilon": epsilon,
                    "episode": episode+1
                }, checkpoint_path)
                cleanup_checkpoints(CHECKPOINT_DIR, MAX_CHECKPOINTS)
                save_training_graph(rewards_per_episode, epsilon_history)
                save_moving_avg_graph(rewards_per_episode)

            if total_reward > best_reward:
                best_reward = total_reward
                torch.save({
                    "policy_state_dict": policy_net.state_dict(),
                    "target_state_dict": target_net.state_dict(),
                    "epsilon": epsilon,
                    "episode": episode+1
                }, "best_model.pth")
                print(f"⭐ Saved new best model at Episode {episode+1}")

            rewards_per_episode.append(total_reward)
            epsilon_history.append(epsilon)
            print(f"Episode {episode+1}: Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

    finally:
        torch.save({
            "policy_state_dict": policy_net.state_dict(),
            "target_state_dict": target_net.state_dict(),
            "epsilon": epsilon,
            "episode": episode
        }, f"{CHECKPOINT_DIR}/tetris_dqn_latest.pth")
        save_training_graph(rewards_per_episode, epsilon_history)
        save_moving_avg_graph(rewards_per_episode)
        env.close()
        print("✅ Latest model saved and graph generated")
