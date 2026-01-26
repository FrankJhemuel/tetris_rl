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

# Initialize ActionsMapping for use in functions
A = ActionsMapping()

# ------------------------
# Directories & Config
# ------------------------
CHECKPOINT_DIR = "checkpoints"
GRAPHS_DIR = "graphs"
BEST_MODELS_DIR = "best_models"  # Store history of best models
GRAPH_FILE = f"{GRAPHS_DIR}/tetris_training_graph_features.png"
SAVE_EVERY = 200
MAX_CHECKPOINTS = 10
MAX_BEST_MODELS = 10 

# Fixed to 5 features: smoothness, lines_cleared, holes, min_height, max_height
FEATURE_DIM = 5

# ------------------------
# DQN - Deep Q-Network
# ------------------------
class DQN(nn.Module):
    def __init__(self, feature_dim=5):
        super(DQN, self).__init__()

        # 5-feature network: feature_dim -> 64 -> 64 -> 1
        self.fc1 = nn.Sequential(nn.Linear(feature_dim, 64), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.fc3 = nn.Sequential(nn.Linear(64, 1))
        self._create_weights()

    def _create_weights(self):
        """Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, features):
        x = self.fc1(features)
        x = self.fc2(x) 
        x = self.fc3(x)
        return x

# ------------------------
# Agent Class
# ------------------------
class Agent:
    def __init__(self, feature_dim, lr, gamma, batch_size, memory_size, device):
        """
        Initialize the DQN Agent with networks, optimizer, and replay memory.
        
        Args:
            feature_dim: Dimension of the feature vector
            lr: Learning rate
            gamma: Discount factor
            batch_size: Batch size for training
            memory_size: Maximum size of replay memory
            device: torch device (cuda or cpu)
        """
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        
        # Create networks
        self.policy_net = DQN(feature_dim).to(device)
        self.target_net = DQN(feature_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        
        # Replay memory
        self.memory = deque(maxlen=memory_size)
    
    def remember(self, features, reward, next_features, done):
        """Store transition in replay memory"""
        self.memory.append((features, reward, next_features, done))

    
    def replay(self):
        """Train the network using a batch from replay memory"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        features, rewards, next_features, dones = zip(*batch)

        features = torch.tensor(np.array(features), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_features = torch.tensor(np.array(next_features), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Current state values
        state_values = self.policy_net(features).squeeze()

        # Compute target Q-values for next states using target network
        with torch.no_grad():
            max_next_q_values = self.target_net(next_features).squeeze()

        # Compute y = r + γ max_a' Q_target(s', a') for non-terminal states
        y_batch = rewards + self.gamma * max_next_q_values * (1 - dones)

        # Loss & update
        loss = self.loss_fn(state_values, y_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    
    def update_target_network(self):
        """Copy weights from policy network to target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, filepath, epsilon, episode, reward=None):
        """Save agent state to file"""
        state = {
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "epsilon": epsilon,
            "episode": episode
        }
        if reward is not None:
            state["reward"] = reward
        torch.save(state, filepath)
    
    def load(self, filepath):
        """Load agent state from file"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        return checkpoint

# ------------------------
# Tetris Environment Wrapper
# ------------------------
class TetrisEnv:
    """Wrapper class for Tetris environment with helper methods"""
    
    # Offsets to convert trimmed-x → env-x
    # Format: piece_index : [offset_per_rotation]
    X_OFFSETS = {
        0: [0, -2],             # I
        1: [0],                 # O
        2: [0, -1, 0, 0],       # T
        3: [0, -1],             # S
        4: [0, -1],             # Z
        5: [0, -1, 0, 0],       # J
        6: [0, -1, 0, 0],       # L
    }
    
    def __init__(self, render_mode=None):
        """Initialize the Tetris environment"""
        self.env = gym.make("tetris_gymnasium/Tetris", render_mode=render_mode)
        self.obs, self.info = self.env.reset()
    
    def reset(self):
        """Reset the environment"""
        self.obs, self.info = self.env.reset()
        return self.obs, self.info
    
    def step(self, action):
        """Take a step in the environment"""
        return self.env.step(action)
    
    def render(self):
        """Render the environment"""
        return self.env.render()
    
    def close(self):
        """Close the environment"""
        self.env.close()
    
    @staticmethod
    def extract_features(board_state, clear_lines=True):
        """
        Vectorized feature extraction for Tetris.

        Features:
        1. Smoothness (bumpiness)
        2. Lines cleared
        3. Holes
        4. Min height
        5. Max height

        Args:
            board_state: 2D numpy array representing the playable board
            clear_lines: Whether to simulate line clearing

        Returns:
            dict with features and optionally cleared board
        """
        board = board_state.copy()
        H, W = board.shape

        # ----------------------
        # 1️⃣ Lines cleared
        # ----------------------
        full_rows = np.all(board > 0, axis=1)
        lines_cleared = np.sum(full_rows)

        if clear_lines and lines_cleared > 0:
            board = np.vstack([np.zeros((lines_cleared, W), dtype=board.dtype),
                            board[~full_rows]])

        # ----------------------
        # 2️⃣ Column heights
        # ----------------------
        filled = board > 0
        # For each column, find the first filled cell from the top
        first_filled = np.argmax(filled, axis=0)
        has_filled = np.any(filled, axis=0)
        column_heights = np.where(has_filled, H - first_filled, 0)

        # ----------------------
        # 3️⃣ Smoothness (bumpiness)
        # ----------------------
        smoothness = np.sum(np.abs(np.diff(column_heights)))

        # ----------------------
        # 4️⃣ Holes
        # ----------------------
        # Cumulative sum from top; a hole is an empty cell below any filled cell
        holes = np.sum(np.logical_and(np.cumsum(filled, axis=0) > 0, ~filled))

        # ----------------------
        # 5️⃣ Min/Max height
        # ----------------------
        min_height = np.min(column_heights)
        max_height = np.max(column_heights)

        return {
            'smoothness': smoothness,
            'lines_cleared': lines_cleared,
            'holes': holes,
            'min_height': min_height,
            'max_height': max_height,
            'cleared_board': board
        }

    
    def get_current_state_features(self):
        """Get features for the current board state"""
        board = self.env.unwrapped.board.astype(np.float32)
        H, W = board.shape
        playable_board = board[:H-4, 4:W-4]  # Remove padding
        
        # Extract features (don't clear lines for current state)
        features = self.extract_features(playable_board, clear_lines=False)
        feature_vector = np.array([
            features['smoothness'] / 100.0,     # Bumpiness: 9 pairs * ~10 avg diff = 90 typical max
            features['lines_cleared'] / 4.0,    # 0-4 → 0-1
            features['holes'] / 10.0,           # 0-10 → 0-1
            features['min_height'] / 20.0,      # 0-20 → 0-1
            features['max_height'] / 20.0       # 0-20 → 0-1
        ], dtype=np.float32)
        
        return feature_vector
    
    @staticmethod
    def shape_reward(lines_cleared, terminated, max_height):
        """
        Reward formula focused on line clearing and survival:
        - Tetris (4 lines): +800 points (good reward, but not excessive)
        - Triple (3 lines): +300 points
        - Double (2 lines): +200 points
        - Single (1 line):  +100 points
        - No lines:         +10 points (survival reward - staying alive is good!)
        - Height penalties: Progressive penalties for dangerous heights
        - Game over:        -500 penalty (strong deterrent)
        
        The +10 survival reward encourages efficient piece placement and teaches
        the agent that successfully placing pieces (even without clearing) is valuable.
        
        Holes and bumpiness are NOT directly penalized here - the agent should
        learn they're bad because they lead to worse future states (fewer options,
        higher risk of death). This allows the agent to discover strategy naturally.
        
        Args:
            lines_cleared: Number of lines cleared this step (0-4)
            terminated: Whether the game ended
            max_height: Maximum column height on the board (0-20)
        
        Returns:
            Reward score for this step
        """
        # Heavy penalty for game over - this is the main teaching signal
        if terminated:
            return -500  # Increased from -300 to strongly discourage risky play
        
        # Base scoring system - reduced Tetris reward to balance risk/reward
        if lines_cleared >= 4:  # Tetris
            reward = 800  # Reduced from 1200 - still excellent but not worth dying for
        elif lines_cleared == 3:  # Triple
            reward = 300
        elif lines_cleared == 2:  # Double
            reward = 200
        elif lines_cleared == 1:  # Single
            reward = 100
        else:
            reward = 10  # Small survival reward - successfully placing a piece is good!
        
        # Height-based penalties: teach agent that high stacks are dangerous
        # These are necessary because height directly affects immediate survival risk
        if max_height >= 16:
            reward -= 500
        elif max_height >= 12:
            reward -= 20   
        elif max_height >= 10:
            reward -= 10   
        
        return reward
    
    @staticmethod
    def trim_piece(piece):
        """Trim whitespace from piece matrix"""
        rows = np.any(piece, axis=1)
        cols = np.any(piece, axis=0)
        return piece[np.ix_(rows, cols)]

    def get_unique_rotations(self, tetromino):
        """Get all unique rotations of a tetromino"""
        rotations = []
        seen = []
        mat = tetromino.matrix.copy()
        for _ in range(4):
            trimmed = self.trim_piece(mat)
            if not any(np.array_equal(trimmed, s) for s in seen):
                seen.append(trimmed)
                rotations.append(trimmed)
            mat = np.rot90(mat, -1)
        return rotations

    @staticmethod
    def piece_bounds(piece):
        """Get left and right bounds of piece"""
        cols = np.any(piece, axis=0)
        left = np.argmax(cols)
        right = len(cols) - 1 - np.argmax(cols[::-1])
        return left, right

    @staticmethod
    def check_collision(board, piece, x, y):
        """Check if piece collides with board at position (x, y)"""
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

    @staticmethod
    def place_piece(board, piece, x, y):
        """Place piece on board at position (x, y)"""
        new_board = board.copy()
        h, w = piece.shape
        for py in range(h):
            for px in range(w):
                if piece[py, px]:
                    new_board[y + py, x + px] = 1
        return new_board

    def compute_action_sequence(self, tetro_idx, tetro, target_rot_id, target_x):
        """Compute action sequence to place piece at target position and rotation"""
        actions = []

        # The piece always spawns at rotation 0, so target_rot_id directly tells us how many clockwise rotations to perform
        n_rot = target_rot_id
        actions += [A.rotate_counterclockwise] * n_rot

        # --- Apply per-piece per-rotation offset ---
        offset = self.X_OFFSETS[tetro_idx][target_rot_id]
        corrected_target_x = target_x + offset

        start_x = self.env.unwrapped.x
        dx = corrected_target_x - start_x

        if dx > 0:
            actions += [A.move_right] * dx
        elif dx < 0:
            actions += [A.move_left] * (-dx)

        actions.append(A.hard_drop)
        return actions

    def get_all_board_states(self, tetro_idx, tetro):
        """Get all possible board states for current piece"""
        board = self.env.unwrapped.board.copy()
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
        
        rotations = self.get_unique_rotations(tetro)
        results = []

        pad = 4
        playable_x_min = pad
        playable_x_max = W - pad

        for rot_id, piece in enumerate(rotations):
            h, w = piece.shape
            left, right = self.piece_bounds(piece)

            # x is the LEFTMOST occupied column in board coords
            min_x = playable_x_min - left
            max_x = playable_x_max - (right + 1)

            for x in range(min_x, max_x + 1):
                y = 0
                while not self.check_collision(board, piece, x, y):
                    y += 1
                y -= 1
                if y < 0:
                    continue

                new_board = self.place_piece(board, piece, x, y)
                
                # Extract playable area for feature calculation
                H_full, W_full = new_board.shape
                playable_board = new_board[:H_full-4, 4:W_full-4]  # Remove padding
                
                # Extract features with line clearing simulation
                # This counts lines BEFORE clearing, then simulates clearing for other features
                features = self.extract_features(playable_board, clear_lines=True)
                
                # Use the cleared board for future state evaluation
                playable_board = features['cleared_board']
                
                # Build feature vector
                feature_vector = np.array([
                    features['smoothness'] / 100.0,         # Bumpiness: 9 pairs * ~10 avg diff = 90 typical max
                    features['lines_cleared'] / 4.0,       # 0-4 → 0-1
                    features['holes'] / 10.0,              # 0-10 → 0-1
                    features['min_height'] / 20.0,         # 0-20 → 0-1
                    features['max_height'] / 20.0          # 0-20 → 0-1
                ], dtype=np.float32)
                
                actions = self.compute_action_sequence(
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

def save_training_graph(rewards_per_episode, epsilon_history, lines_per_episode):
    fig = plt.figure(figsize=(18,5))
    window = 50
    
    # Rewards
    plt.subplot(1, 3, 1)
    if len(rewards_per_episode) >= window:
        smoothed_rewards = np.convolve(rewards_per_episode, np.ones(window)/window, mode='valid')
    else:
        smoothed_rewards = rewards_per_episode
    plt.plot(range(len(smoothed_rewards)), smoothed_rewards)
    plt.xlabel("Episode")
    plt.ylabel(f"Total Reward (window={window})")
    plt.title("Total Reward Progression")
    
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

def evaluate_agent(agent, env, n_episodes=5, render=False):
    """
    Evaluate the agent over n_episodes and return the mean reward.
    
    Args:
        agent: DQN agent
        env: TetrisEnv instance
        n_episodes: Number of episodes to average
        render: If True, render the game
    
    Returns:
        mean_reward: Average total reward over n_episodes
        mean_lines: Average lines cleared
    """
    total_rewards = []
    total_lines = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_lines = 0
        terminated = False

        while not terminated:
            # Get all possible placements
            tetro = env.env.unwrapped.active_tetromino
            tetro_idx = tetro.id - 2
            placements = env.get_all_board_states(tetro_idx, tetro)
            if not placements:
                break

            # Compute Q-values
            feature_vectors = np.array([p["features"] for p in placements], dtype=np.float32)
            features_batch = torch.tensor(feature_vectors, dtype=torch.float32).to(agent.device)
            agent.policy_net.eval()
            with torch.no_grad():
                state_values = agent.policy_net(features_batch).squeeze()
            agent.policy_net.train()

            # Pick best action
            chosen_idx = torch.argmax(state_values).item()
            chosen = placements[chosen_idx]

            # Execute action sequence
            lines_cleared_this_step = 0
            for a in chosen["actions"]:
                obs, reward_step, terminated, truncated, info = env.step(a)
                lines_cleared_this_step = info.get("lines_cleared", 0)
                if terminated:
                    break
                if render:
                    env.render()

            # Reward calculation (same as during training)
            max_height = chosen["features"][4] * 20.0
            step_reward = env.shape_reward(lines_cleared_this_step, terminated, max_height)
            episode_reward += step_reward
            episode_lines += lines_cleared_this_step

        total_rewards.append(episode_reward)
        total_lines.append(episode_lines)

    mean_reward = np.mean(total_rewards)
    mean_lines = np.mean(total_lines)
    return mean_reward, mean_lines

# ------------------------
# Main Training Loop
# ------------------------

if __name__ == "__main__":
    # Create directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    os.makedirs(BEST_MODELS_DIR, exist_ok=True)
    
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
    
    print("🔧 Using 5-feature extraction (smoothness, lines_cleared, holes, min_height, max_height)")
    
    # Create environment wrapper
    tetris_env = TetrisEnv(render_mode=None)
    
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

    # Create agent
    agent = Agent(
        feature_dim=feature_dim,
        lr=lr,
        gamma=gamma,
        batch_size=batch_size,
        memory_size=memory_size,
        device=device
    )

    # Load checkpoint if it exists
    RESUME_CHECKPOINT = "best_model_features__.pth"  # Feature-based model saved by reward
    start_episode = 0
    if os.path.exists(RESUME_CHECKPOINT):
        print(f"Loading checkpoint: {RESUME_CHECKPOINT}")
        checkpoint = agent.load(RESUME_CHECKPOINT)
        epsilon = checkpoint.get("epsilon", epsilon)  # Resume from saved epsilon
        start_episode = checkpoint.get("episode", 0)
        print(f"Resumed from episode {start_episode}, epsilon={epsilon:.4f}")
    else:
        print("No feature-based checkpoint found, starting from scratch")
    
    best_reward = -float("inf")
    best_mean = -float("inf")
    best_lines = 0
    rewards_per_episode = []
    epsilon_history = []
    lines_per_episode = []
    

    try:
        episode = 0

        # Get initial state features
        obs, info = tetris_env.reset()
        current_state_features = tetris_env.get_current_state_features()

        print(f"\n🚀 Starting training for {num_episodes} episodes...")
        print(f"💾 Checkpoints will be saved every {save_interval} episodes\n")

        pieces_placed = 0
        episode_pieces = 0
        episode_reward_total = 0
        episode_lines_total = 0
        
        

        while episode < num_episodes:
            # Get all possible next states
            tetro = tetris_env.env.unwrapped.active_tetromino
            tetro_idx = tetro.id - 2
            placements = tetris_env.get_all_board_states(tetro_idx, tetro)

            if not placements:
                # Can't place piece - game over
                obs, info = tetris_env.reset()
                current_state_features = tetris_env.get_current_state_features()
                continue

            # Get all feature vectors and Q-values
            feature_vectors = np.array([p["features"] for p in placements], dtype=np.float32)
            features_batch = torch.tensor(feature_vectors, dtype=torch.float32).to(device)

            agent.policy_net.eval()
            with torch.no_grad():
                state_values = agent.policy_net(features_batch).squeeze()
            agent.policy_net.train()

            # Epsilon-greedy action selection
            if random.random() <= epsilon:
                chosen_idx = random.randint(0, len(placements) - 1)
            else:
                chosen_idx = torch.argmax(state_values).item()

            chosen = placements[chosen_idx]

            # Track pieces placed
            pieces_placed += 1
            episode_pieces += 1

            # Execute the action sequence
            lines_cleared_this_step = 0
            terminated = False
            for a in chosen["actions"]:
                if terminated:
                    break
                obs, reward_step, terminated, truncated, info = tetris_env.step(a)
                lines_cleared_this_step = info.get("lines_cleared", 0)

            # Denormalize max_height for reward
            max_height = feature_vectors[chosen_idx][4] * 20.0

            # Calculate reward
            step_reward = tetris_env.shape_reward(lines_cleared_this_step, terminated, max_height)
            episode_reward_total += step_reward
            episode_lines_total += lines_cleared_this_step

            # Get next state features
            if not terminated:
                next_state_features = tetris_env.get_current_state_features()
            else:
                next_state_features = current_state_features  # just keep current; TD target ignores done

            # Store transition in replay memory
            agent.remember(current_state_features, step_reward, next_state_features, terminated)

            # Update current state
            current_state_features = next_state_features
        
            # End of episode processing
            if terminated:
                episode += 1

                # Exponential epsilon decay
                epsilon = max(epsilon_min, epsilon * epsilon_decay)

                # Train if enough samples
                if len(agent.memory) >= batch_size:
                    agent.replay()
                
                # Target network update
                if episode % target_update_freq == 0:
                    agent.update_target_network()

                # Checkpoint saving
                if episode % save_interval == 0:
                    checkpoint_path = f"{CHECKPOINT_DIR}/tetris_dqn_ep{episode}.pth"
                    agent.save(checkpoint_path, epsilon, episode)
                    cleanup_checkpoints(CHECKPOINT_DIR, MAX_CHECKPOINTS)
                    save_training_graph(rewards_per_episode, epsilon_history, lines_per_episode)
                    print(f"💾 Checkpoint saved at episode {episode}")
                
                # Save best model based on total episode reward
                if episode_reward_total > best_reward:
                    mean_reward, mean_lines = evaluate_agent(agent, tetris_env, n_episodes=3)

                    if mean_reward > best_mean:
                        best_mean = mean_reward
                        best_reward = episode_reward_total
                        
                        # Save to best_models history folder
                        best_model_history_path = f"{BEST_MODELS_DIR}/best_model_ep{episode}_reward{episode_reward_total:.0f}.pth"
                        agent.save(best_model_history_path, epsilon, episode, episode_reward_total)
                        
                        # Clean up old best models (keep only last MAX_BEST_MODELS)
                        cleanup_checkpoints(BEST_MODELS_DIR, MAX_BEST_MODELS)
                        
                        # Also save as the main best_model.pth (most accessible for play_model)
                        agent.save("best_model.pth", epsilon, episode, episode_reward_total)
                        print(f"💎 Saved new best model at Episode {episode} (Reward: {episode_reward_total:.2f})")
                        print(f"   📜 History saved to: {best_model_history_path}")
                
                # Logging
                rewards_per_episode.append(episode_reward_total)
                epsilon_history.append(epsilon)
                lines_per_episode.append(episode_lines_total)

                # Q-value diagnostics
                if len(agent.memory) >= batch_size:
                    sample_batch = random.sample(agent.memory, min(100, len(agent.memory)))
                    sample_features = torch.tensor(np.array([s[0] for s in sample_batch]), dtype=torch.float32).to(device)
                    with torch.no_grad():
                        sample_q = agent.policy_net(sample_features).squeeze()
                    avg_q = sample_q.mean().item()
                    max_q = sample_q.max().item()
                    min_q = sample_q.min().item()
                    print(f"Episode {episode}: {episode_pieces} pieces, reward={episode_reward_total:.1f}, ε={epsilon:.3f} | Q: avg={avg_q:.1f}, max={max_q:.1f}, min={min_q:.1f}, mem={len(agent.memory)}")
                else:
                    print(f"Episode {episode}: {episode_pieces} pieces, reward={episode_reward_total:.1f}, ε={epsilon:.3f}, mem={len(agent.memory)}/{batch_size} (warmup)")

                # Reset episode tracking
                episode_pieces = 0
                episode_reward_total = 0
                episode_lines_total = 0

                obs, info = tetris_env.reset()
                current_state_features = tetris_env.get_current_state_features()

    finally:
        agent.save(f"{CHECKPOINT_DIR}/tetris_dqn_latest.pth", epsilon, episode)
        save_training_graph(rewards_per_episode, epsilon_history, lines_per_episode)
        tetris_env.close()
        print("✅ Latest model saved and graph generated")