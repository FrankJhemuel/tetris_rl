import gymnasium as gym
import torch
import cv2
import numpy as np
import random
import argparse
import os
import glob

from tetris_gymnasium.envs.tetris import Tetris
from tetris_gymnasium.mappings.actions import ActionsMapping
from tetris_dqn import DQN, get_state_channels, get_all_board_states  # helpers from your training script

A = ActionsMapping()

# ------------------------
# Command Line Arguments
# ------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Play Tetris with trained DQN model')
    parser.add_argument('--model', type=str, choices=['best', 'latest'], default='best',
                       help='Choose model: "best" (best_model.pth) or "latest" (latest checkpoint)')
    parser.add_argument('--checkpoint', type=str, 
                       help='Specific checkpoint file (e.g., checkpoints/tetris_dqn_ep1000.pth)')
    return parser.parse_args()

def get_model_path(args):
    if args.checkpoint:
        # Use specific checkpoint file
        if os.path.exists(args.checkpoint):
            return args.checkpoint
        else:
            print(f"❌ Checkpoint file not found: {args.checkpoint}")
            exit(1)
    
    elif args.model == 'best':
        # Use best model
        if os.path.exists("best_model.pth"):
            return "best_model.pth"
        else:
            print("❌ best_model.pth not found!")
            exit(1)
    
    elif args.model == 'latest':
        # Find latest checkpoint in checkpoints directory
        checkpoint_files = glob.glob("checkpoints/tetris_dqn_*.pth")
        if checkpoint_files:
            # Sort by modification time and get the latest
            latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
            return latest_checkpoint
        else:
            print("❌ No checkpoints found in checkpoints/ directory!")
            exit(1)

args = parse_args()
model_path = get_model_path(args)

# ------------------------
# Device
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------
# Environment
# ------------------------
env = gym.make("tetris_gymnasium/Tetris", render_mode="rgb_array")
obs, info = env.reset()
state = get_state_channels(env)
state_dim = state.shape

# ------------------------
# Load trained model
# ------------------------
policy_net = DQN(state_dim).to(device)
print(f"📂 Loading model from: {model_path}")
checkpoint = torch.load(model_path, map_location=device)

try:
    policy_net.load_state_dict(checkpoint["policy_state_dict"])
    policy_net.eval()
    
    # Print model information
    episode = checkpoint.get("episode", "Unknown")
    epsilon = checkpoint.get("epsilon", "Unknown")
    print(f"📋 Loaded model from episode: {episode}")
    if epsilon != "Unknown":
        print(f"📋 Model's epsilon at save time: {epsilon:.4f}")
    print(f"📋 Model loaded successfully!")
    print("-" * 40)
    
except RuntimeError as e:
    if "size mismatch" in str(e):
        print("❌ Model architecture mismatch!")
        print("💡 The saved model was trained with a different input format.")
        print("💡 Please train a new model with the current architecture or use a compatible checkpoint.")
        print(f"💡 Current model expects: {state_dim}")
        exit(1)
    else:
        raise e

# ------------------------
# Play loop
# ------------------------
terminated = False
total_reward = 0
total_lines = 0

while not terminated:
    tetro = env.unwrapped.active_tetromino
    tetro_idx = tetro.id - 2  # Convert tetromino ID (2-8) to index (0-6)

    # 1️⃣ Generate all possible placements for current piece
    placements = get_all_board_states(env, tetro_idx, tetro)

    # 2️⃣ Convert boards to batch tensor for DQN (now 2-channel format)
    boards_batch = np.array([p["board"] for p in placements], dtype=np.float32)
    boards_batch = torch.tensor(boards_batch, dtype=torch.float32).to(device)  # No need to add dimension, already has 2 channels

    # 3️⃣ Predict Q-values
    with torch.no_grad():
        q_values = policy_net(boards_batch).squeeze()

    # 4️⃣ Choose the best placement (greedy)
    chosen_idx = torch.argmax(q_values).item()
    chosen = placements[chosen_idx]

    # 5️⃣ Execute the full action sequence for this placement
    for a in chosen["actions"]:
        if terminated:
            break
        obs, reward_step, terminated, truncated, info = env.step(a)
        state = get_state_channels(env)
        total_reward += reward_step
        total_lines += info.get("lines_cleared", 0)

        # Render
        frame = env.render()
        if frame is not None:
            frame = cv2.resize(frame, (frame.shape[1]*4, frame.shape[0]*4), interpolation=cv2.INTER_NEAREST)
            cv2.putText(
                frame,
                f"Lines: {total_lines}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            cv2.imshow("Tetris DQN Play", frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                terminated = True
                break

print("Game Over!")
print("Total Reward:", total_reward)
print("Total Lines Cleared:", total_lines)

env.close()
cv2.destroyAllWindows()
