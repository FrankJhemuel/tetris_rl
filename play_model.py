import gymnasium as gym
import torch
import cv2
import numpy as np
import random
import argparse
import os
import glob
import yaml

from tetris_gymnasium.envs.tetris import Tetris
from tetris_gymnasium.mappings.actions import ActionsMapping
from tetris_dqn import DQN, get_all_board_states, extract_features

A = ActionsMapping()

# Fixed to 5-feature approach
feature_dim = 5
print("🔧 Using 5-feature extraction (aggregate_height, complete_lines, holes, min_height, max_height)")

# ------------------------
# Command Line Arguments
# ------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Play Tetris with trained DQN model')
    parser.add_argument('--model', type=str, choices=['best', 'latest', 'features', 'reference'], default='features',
                       help='Choose model: "best" (best_model.pth), "latest" (latest checkpoint), "features" (best_model_features.pth), or "reference" (best_model_reference.pth)')
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
    
    elif args.model == 'features':
        # Use best feature-based model
        if os.path.exists("best_model_features.pth"):
            return "best_model_features.pth"
        else:
            print("❌ best_model_features.pth not found!")
            exit(1)
    
    elif args.model == 'reference':
        # Use best reference model
        if os.path.exists("best_model_reference.pth"):
            return "best_model_reference.pth"
        else:
            print("❌ best_model_reference.pth not found!")
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

# ------------------------
# Load trained model
# ------------------------
policy_net = DQN(feature_dim).to(device)
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
        print(f"💡 Current model expects: {feature_dim} features")
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

    # 2️⃣ Extract features for each placement
    features_batch = []
    for placement in placements:
        # The placement already contains the feature vector from training
        features_batch.append(placement["features"])
    
    features_batch = torch.tensor(np.array(features_batch), dtype=torch.float32).to(device)

    # 3️⃣ Predict Q-values
    with torch.no_grad():
        q_values = policy_net(features_batch).squeeze()

    # 4️⃣ Choose the best placement (greedy)
    chosen_idx = torch.argmax(q_values).item()
    chosen = placements[chosen_idx]
    
    # 🔍 Debug: Show what the model is thinking
    print(f"Piece {tetro_idx}: {len(placements)} placements, Q-values: max={q_values.max():.2f}, min={q_values.min():.2f}")
    print(f"Chosen features: agg_height={chosen['features'][0]:.3f}, lines={chosen['features'][1]:.3f}, holes={chosen['features'][2]:.3f}, min_h={chosen['features'][3]:.3f}, max_h={chosen['features'][4]:.3f}")
    
    # Show top 3 choices
    top_3 = torch.topk(q_values, min(3, len(q_values)))
    for i, (q_val, idx) in enumerate(zip(top_3.values, top_3.indices)):
        place = placements[idx]
        print(f"  #{i+1}: Q={q_val:.2f} x={place['x']} rot={place['rotation']} features={place['features']}")

    # 5️⃣ Execute the full action sequence for this placement
    for a in chosen["actions"]:
        if terminated:
            break
        obs, reward_step, terminated, truncated, info = env.step(a)
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
