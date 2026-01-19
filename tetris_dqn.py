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
GRAPHS_DIR = "graphs"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)
GRAPH_FILE = f"{GRAPHS_DIR}/tetris_training_graph_features.png"
SAVE_EVERY = 200
MAX_CHECKPOINTS = 10

# Load reference hyperparameters
with open("hyperparams.yaml", "r") as f:
    config = yaml.safe_load(f)

gamma = config["gamma"]
epsilon = config["initial_epsilon"]
epsilon_min = config["final_epsilon"]
epsilon_decay = config["epsilon_decay"]
lr = config["lr"]
batch_size = config["batch_size"]
memory_size = config["memory_size"]
target_update_freq = config["target_update_freq"]
num_episodes = config["num_episodes"]
save_interval = config["save_interval"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

A = ActionsMapping()

# Fixed to 5 features: aggregate_height, complete_lines, holes, min_height, max_height
FEATURE_DIM = 5

print("🔧 Using 5-feature extraction (aggregate_height, complete_lines, holes, min_height, max_height)")

# ------------------------
# Feature-Only DQN Network (TetrisAI Style)
# ------------------------
class DQN(nn.Module):
    def __init__(self, feature_dim=5):
        super(DQN, self).__init__()
        
        # 5-feature network: feature_dim -> 64 -> 64 -> 1
        self.conv1 = nn.Sequential(nn.Linear(feature_dim, 64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Linear(64, 1))
        self._create_weights()
        print(f"🔧 Network architecture: {feature_dim} -> 64 -> 64 -> 1")

    def _create_weights(self):
        """Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, features):
        x = self.conv1(features)
        x = self.conv2(x) 
        x = self.conv3(x)
        return x

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

def extract_features(board_state):
    """
    Extract 5 features for Tetris board evaluation:
    1. Aggregate height (sum of all column heights)
    2. Complete lines (lines that can be cleared)
    3. Holes (empty cells below filled cells)
    4. Min height (shortest column)
    5. Max height (tallest column)
    
    Args:
        board_state: 2D numpy array representing the playable board
    
    Returns:
        dict with 5 feature values
    """
    H, W = board_state.shape
    
    # Calculate column heights
    column_heights = []
    for col in range(W):
        column = board_state[:, col]
        filled_rows = np.where(column > 0)[0]
        if len(filled_rows) > 0:
            height = H - filled_rows[0]  # Height from bottom
            column_heights.append(height)
        else:
            column_heights.append(0)
    
    # Feature 1: Aggregate height
    aggregate_height = sum(column_heights)
    
    # Feature 2: Complete lines
    complete_lines = 0
    for row in range(H):
        if all(board_state[row, col] > 0 for col in range(W)):
            complete_lines += 1
            
    # Feature 3: Holes
    holes = 0
    for col in range(W):
        column = board_state[:, col]
        filled_rows = np.where(column > 0)[0]
        if len(filled_rows) > 0:
            for row in range(filled_rows[0] + 1, H):
                if column[row] == 0:
                    holes += 1
    
    # Feature 4: Min height
    min_height = min(column_heights) if column_heights else 0
    
    # Feature 5: Max height
    max_height = max(column_heights) if column_heights else 0
    
    return {
        'aggregate_height': aggregate_height,
        'complete_lines': complete_lines, 
        'holes': holes,
        'min_height': min_height,
        'max_height': max_height
    }

def shape_reward(lines_cleared, terminated, last_clear_was_tetris=False):
    """
    Reward formula based on the Processing reference code:
    - Regular clears (1-3 lines): cleared * 100
    - Tetris (4 lines): cleared * 200
    - Back-to-back Tetris bonus: +400
    - Game over: return negative score
    
    Args:
        lines_cleared: Number of lines cleared this step
        terminated: Whether the game ended
        last_clear_was_tetris: Whether the previous clear was also a tetris (for bonus)
    
    Returns:
        Reward score for this step
    """
    if terminated:
        return -200  # Penalty for game over
    
    score = 0
    
    if lines_cleared >= 4:  # Tetris (4-line clear)
        score += lines_cleared * 200
        if last_clear_was_tetris:
            score += 400  # Back-to-back tetris bonus
    elif lines_cleared > 0:  # Regular clear (1-3 lines)
        score += lines_cleared * 100
    
    return score

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
# Replay Memory & Networks
# ------------------------
# Set seed like reference
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed(123)

# Use the configured feature dimension
feature_dim = FEATURE_DIM

# Print hyperparameters
print("\n🔧 Reference Implementation Hyperparameters:")
for key, value in config.items():
    print(f"   {key}: {value}")
print()

# Create networks
policy_net = DQN(feature_dim).to(device)
target_net = DQN(feature_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=lr)
loss_fn = nn.MSELoss()
memory = deque(maxlen=memory_size)

def remember(features, reward, next_features, done):
    memory.append((features, reward, next_features, done))

def replay():
    """Reference-style replay training (exact match to TetrisAI)"""
    if len(memory) < batch_size:
        return
    
    batch = random.sample(memory, batch_size)
    features, rewards, next_features, dones = zip(*batch)
    
    features = torch.tensor(np.array(features), dtype=torch.float32).to(device)
    next_features = torch.tensor(np.array(next_features), dtype=torch.float32).to(device)
    rewards = torch.tensor(np.array(rewards, dtype=np.float32), dtype=torch.float32).to(device)
    
    # Q-values for current states
    q_values = policy_net(features).squeeze()
    
    # Q-values for next states (using target network)
    policy_net.eval()
    with torch.no_grad():
        next_q_values = target_net(next_features).squeeze()
    policy_net.train()
    
    # Compute target Q-values (reference formula)
    y_batch = []
    for reward, done, next_q in zip(rewards, dones, next_q_values):
        if done:
            y_batch.append(reward)
        else:
            y_batch.append(reward + gamma * next_q)
    y_batch = torch.stack(y_batch).to(device)
    
    # Compute loss and update
    loss = loss_fn(q_values, y_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def get_current_state_features(env):
    """Get features for the current board state"""
    board = env.unwrapped.board.astype(np.float32)
    H, W = board.shape
    playable_board = board[:H-4, 4:W-4]  # Remove padding
    
    # Extract features
    features = extract_features(playable_board)
    feature_vector = np.array([
        features['aggregate_height'] / 200.0,
        features['complete_lines'] / 4.0,
        features['holes'] / 10.0,
        features['min_height'] / 20.0,
        features['max_height'] / 20.0
    ], dtype=np.float32)
    
    return feature_vector
# ------------------------
# Main Training Loop
# ------------------------

if __name__ == "__main__":
    # Load checkpoint if it exists
    RESUME_CHECKPOINT = "best_model_features__.pth"  # Feature-based model saved by reward
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
        print("No feature-based checkpoint found, starting from scratch")
    
    best_reward = -float("inf")
    best_lines = 0
    rewards_per_episode = []
    epsilon_history = []
    lines_per_episode = []

    try:
        episode = 0
        
        # Get initial state features
        obs, info = env.reset()
        current_state_features = get_current_state_features(env)
        
        print(f"\n🚀 Starting training for {num_episodes} episodes...")
        print(f"💾 Checkpoints will be saved every {save_interval} episodes\n")
        
        pieces_placed = 0
        episode_pieces = 0
        episode_reward_total = 0  # Track cumulative reward per episode
        last_clear_was_tetris = False  # Track back-to-back tetris bonus

        while episode < num_episodes:
            # Get all possible next states (like reference implementation)
            tetro = env.unwrapped.active_tetromino
            tetro_idx = tetro.id - 2
            placements = get_all_board_states(env, tetro_idx, tetro)
            
            if not placements:
                # Can't place piece - game over
                obs, info = env.reset()
                current_state_features = get_current_state_features(env)
                continue

            # Exponential epsilon decay
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            
            # Get all feature vectors and Q-values
            feature_vectors = np.array([p["features"] for p in placements], dtype=np.float32)
            features_batch = torch.tensor(feature_vectors, dtype=torch.float32).to(device)

            policy_net.eval()
            with torch.no_grad():
                q_values = policy_net(features_batch).squeeze()
            policy_net.train()

            # Epsilon-greedy action selection (reference style)
            u = random.random()
            if u <= epsilon:
                chosen_idx = random.randint(0, len(placements) - 1)
            else:
                chosen_idx = torch.argmax(q_values).item()

            chosen = placements[chosen_idx]
            next_state_features = feature_vectors[chosen_idx]
            
            # Track pieces placed
            pieces_placed += 1
            episode_pieces += 1
            
            # Execute the action sequence
            lines_cleared_this_step = 0
            terminated = False
            for a in chosen["actions"]:
                if terminated:
                    break
                obs, reward_step, terminated, truncated, info = env.step(a)
                lines_cleared_this_step = info.get("lines_cleared", 0)
                env.render()
                cv2.waitKey(1)
            
            # Calculate reward with back-to-back tetris tracking
            step_reward = shape_reward(lines_cleared_this_step, terminated, last_clear_was_tetris)
            episode_reward_total += step_reward
            
            # Update tetris tracking for next piece
            if lines_cleared_this_step >= 4:
                last_clear_was_tetris = True
            elif lines_cleared_this_step > 0:
                last_clear_was_tetris = False
            
            # Store transition in replay memory
            remember(current_state_features, step_reward, next_state_features, terminated)
            
            # If episode ended, train and start new episode
            if terminated:
                episode += 1
                
                # Train if we have enough samples
                if len(memory) >= batch_size:
                    replay()
                
                # Target network update
                if episode % target_update_freq == 0:
                    target_net.load_state_dict(policy_net.state_dict())
                
                # Checkpoint saving
                if episode % save_interval == 0:
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
                
                # Save best model based on total episode reward
                if episode_reward_total > best_reward:
                    best_reward = episode_reward_total
                    torch.save({
                        "policy_state_dict": policy_net.state_dict(),
                        "target_state_dict": target_net.state_dict(),
                        "epsilon": epsilon,
                        "episode": episode
                    }, "best_model_features.pth")
                    print(f"💎 Saved new best model at Episode {episode} (Reward: {episode_reward_total:.2f})")
                
                # Logging
                rewards_per_episode.append(episode_reward_total)
                epsilon_history.append(epsilon)
                lines_per_episode.append(info.get("lines_cleared", 0))  # Lines from env info
                
                # Per-episode diagnostics with Q-value monitoring
                if len(memory) >= batch_size:
                    sample_batch = random.sample(memory, min(100, len(memory)))
                    sample_features = torch.tensor(np.array([s[0] for s in sample_batch]), dtype=torch.float32).to(device)
                    with torch.no_grad():
                        sample_q = policy_net(sample_features).squeeze()
                    avg_q = sample_q.mean().item()
                    max_q = sample_q.max().item()
                    min_q = sample_q.min().item()
                    print(f"Episode {episode}: {episode_pieces} pieces, reward={episode_reward_total:.1f}, ε={epsilon:.3f} | Q: avg={avg_q:.1f}, max={max_q:.1f}, min={min_q:.1f}, mem={len(memory)}")
                else:
                    print(f"Episode {episode}: {episode_pieces} pieces, reward={episode_reward_total:.1f}, ε={epsilon:.3f}, mem={len(memory)}/{batch_size} (warmup)")
                
                # Reset episode tracking variables
                episode_pieces = 0
                episode_reward_total = 0
                last_clear_was_tetris = False  # Reset tetris tracking for new episode
                
                # Reset environment
                obs, info = env.reset()
                current_state_features = get_current_state_features(env)
            else:
                # Continue with next piece
                current_state_features = next_state_features

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