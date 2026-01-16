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

# ------------------------
# Directories
# ------------------------
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
GRAPH_FILE = "tetris_training_graph.png"
SAVE_EVERY = 100
MAX_CHECKPOINTS = 10

# ------------------------
# Device
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------
# DQN Network
# ------------------------
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
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
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ------------------------
# Hyperparameters
# ------------------------
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.9995
epsilon_min = 0.1
lr = 1e-4
batch_size = 32
memory_size = 50000
target_update_freq = 50
num_episodes = 10000

# ------------------------
# Environment
# ------------------------
env = gym.make("tetris_gymnasium/Tetris", render_mode=None)
obs, info = env.reset()

# ------------------------
# Helper functions
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

def get_state_channels(env: Tetris) -> np.ndarray:
    """Return a 3-channel state: current, next, hold tetromino"""
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

def shape_reward(obs, base_reward, info, env):
    """Reward shaping for Tetris using only constructed blocks."""
    env = env.unwrapped
    reward = base_reward
    # board = obs['board']   # static/frozen board only
    board = env.board.copy()  # now only frozen blocks
    # print(board)
    H, W = board.shape
    
    # Remove padding: bottom 4, left 4, right 4
    playable_board = board[:H-4, 4:W-4]
    H_p, W_p = playable_board.shape

    # Survive
    reward += 0.5

    # Lines cleared
    lines = info.get('lines_cleared', 0)
    reward += lines * 10

    # # Holes penalty
    holes = 0
    for col in range(W_p):
        found_block = False
        for row in range(H_p):
            if playable_board[row, col] > 0:
                found_block = True
            elif found_block and playable_board[row, col] == 0:
                holes += 1
    reward -= (holes / (H_p * W_p)) * 0.33

    # Height penalty (normalized)
    filled_rows = np.where(playable_board.sum(axis=1) > 0)[0]  # rows with blocks
    if len(filled_rows) > 0:
        # Distance from top of playable board to highest block
        max_height = H_p - filled_rows[0]
        reward -= (max_height / H_p) * 0.33  # normalize to [0,5]
    else:
        max_height = 0  # empty board → no penalty

    # Smoothness penalty (normalized)
    # Compute column heights
    column_heights = []
    for c in range(W_p):
        col = playable_board[:, c]
        filled_rows = np.where(col > 0)[0]
        if len(filled_rows) > 0:
            height = H_p - filled_rows[0]  # distance from top of playable board
        else:
            height = 0
        column_heights.append(height)

    # Smoothness = sum of absolute differences between adjacent columns
    smoothness = sum(abs(column_heights[i] - column_heights[i+1]) for i in range(W_p-1))

    # Normalize by width and max possible height difference (H_p)
    reward -= (smoothness / ((W_p - 1) * H_p)) * 0.33

    return reward

def get_valid_actions(env):
    env = env.unwrapped
    a = env.actions
    valid = []

    if not env.collision(env.active_tetromino, env.x - 1, env.y):
        valid.append(a.move_left)
    if not env.collision(env.active_tetromino, env.x + 1, env.y):
        valid.append(a.move_right)
    if not env.collision(env.active_tetromino, env.x, env.y + 1):
        valid.append(a.move_down)
    if not env.collision(env.rotate(env.active_tetromino, True), env.x, env.y):
        valid.append(a.rotate_clockwise)
    if not env.collision(env.rotate(env.active_tetromino, False), env.x, env.y):
        valid.append(a.rotate_counterclockwise)
    if not env.has_swapped:
        valid.append(a.swap)
    valid.append(a.hard_drop)
    valid.append(a.no_op)

    return valid

def save_training_graph(rewards_per_episode, epsilon_history):
    fig = plt.figure(figsize=(12,5))

    # Reward moving average
    window = 50
    avg_rewards = np.convolve(rewards_per_episode, np.ones(window)/window, mode='valid')
    plt.subplot(1, 2, 1)
    plt.plot(range(len(avg_rewards)), avg_rewards)
    plt.xlabel("Episode")
    plt.ylabel(f"Avg Reward (window={window})")
    plt.title("Reward Progression")

    # Epsilon decay
    plt.subplot(1, 2, 2)
    plt.plot(epsilon_history)
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay")

    plt.tight_layout()
    plt.savefig(GRAPH_FILE)
    plt.close(fig)

# ------------------------
# Networks and replay
# ------------------------
dummy_state = get_state_channels(env)
state_dim = dummy_state.shape
action_dim = env.action_space.n

policy_net = DQN(state_dim, action_dim).to(device)
target_net = DQN(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=lr)
loss_fn = nn.MSELoss()
memory = deque(maxlen=memory_size)

def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

def replay():
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.int64).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = target_net(next_states).max(1)[0]
    target = rewards + gamma * next_q_values * (1 - dones)

    loss = loss_fn(q_values, target.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ------------------------
# Training loop
# ------------------------
if __name__ == "__main__":
    best_reward = -float("inf")
    rewards_per_episode = []
    epsilon_history = []

    try:
        for episode in range(num_episodes):
            obs, info = env.reset()
            state = get_state_channels(env)
            total_reward = 0
            terminated = False

            lines_cleared = 0

            while not terminated:
                valid_actions = get_valid_actions(env)

                # Epsilon-greedy with action masking
                if random.random() < epsilon:
                    action = random.choice(valid_actions)
                else:
                    with torch.no_grad():
                        state_tensor = torch.tensor(state[None, :, :, :], dtype=torch.float32).to(device)
                        q_values = policy_net(state_tensor)[0].cpu().numpy()
                        masked_q = -np.inf * np.ones_like(q_values)
                        masked_q[valid_actions] = q_values[valid_actions]
                        action = masked_q.argmax()

                next_obs, reward, terminated, truncated, info = env.step(action)
                next_state = get_state_channels(env)
                reward = shape_reward(next_obs, reward, info, env)
                lines_cleared += info.get("lines_cleared", 0)

                remember(state, action, reward, next_state, terminated)
                state = next_state
                # print(reward)
                total_reward += reward

                replay()

            # Epsilon decay
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            # Update target network
            if (episode + 1) % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # Save checkpoint
            if (episode + 1) % SAVE_EVERY == 0:
                checkpoint_path = f"{CHECKPOINT_DIR}/tetris_dqn_ep{episode+1}.pth"
                torch.save({
                    "policy_state_dict": policy_net.state_dict(),
                    "target_state_dict": target_net.state_dict(),
                    "epsilon": epsilon,
                    "episode": episode + 1
                }, checkpoint_path)
                print(f"💾 Saved checkpoint: {checkpoint_path}")
                cleanup_checkpoints(CHECKPOINT_DIR, MAX_CHECKPOINTS)
                save_training_graph(rewards_per_episode, epsilon_history)

            # Save best model
            if total_reward > best_reward:
                best_reward = total_reward
                torch.save({
                    "policy_state_dict": policy_net.state_dict(),
                    "target_state_dict": target_net.state_dict(),
                    "epsilon": epsilon,
                    "episode": episode + 1
                }, "best_model.pth")
                print(f"⭐ Saved new best model at Episode {episode+1}")

            rewards_per_episode.append(total_reward)
            epsilon_history.append(epsilon)

            print(f"Episode {episode+1}: Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")
            if lines_cleared > 0:
                print(f"   Lines cleared this episode: {lines_cleared}")

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted")

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
