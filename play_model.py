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
        if os.path.exists("best_model.pth"):
            return "best_model.pth"
        else:
            print("❌ best_model.pth not found!")
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
# High Score Management
# ------------------------
HIGHSCORE_FILE = "highscore.txt"

def load_highscore():
    """Load high score from file"""
    if os.path.exists(HIGHSCORE_FILE):
        try:
            with open(HIGHSCORE_FILE, 'r') as f:
                data = f.read().strip().split(',')
                return {
                    'score': int(data[0]),
                    'lines': int(data[1]),
                    'tetrises': int(data[2]) if len(data) > 2 else 0
                }
        except:
            return {'score': 0, 'lines': 0, 'tetrises': 0}
    return {'score': 0, 'lines': 0, 'tetrises': 0}

def save_highscore(score, lines, tetrises):
    """Save high score to file"""
    with open(HIGHSCORE_FILE, 'w') as f:
        f.write(f"{score},{lines},{tetrises}")

highscore = load_highscore()
print(f"\n🏆 Current High Score: {highscore['score']} (Lines: {highscore['lines']}, Tetrises: {highscore['tetrises']})\n")

# ------------------------
# Play loop
# ------------------------
game_count = 0

while True:  # Infinite loop - play until user quits
    game_count += 1
    obs, info = env.reset()
    terminated = False
    total_reward = 0
    total_lines = 0
    total_score = 0
    pieces_placed = 0
    line_clears = {1: 0, 2: 0, 3: 0, 4: 0}  # Track single, double, triple, tetris
    last_clear_was_tetris = False
    
    print(f"\n{'='*50}")
    print(f"🎮 GAME #{game_count} START")
    print(f"{'='*50}\n")

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
        lines_cleared_this_step = 0
        for a in chosen["actions"]:
            if terminated:
                break
            obs, reward_step, terminated, truncated, info = env.step(a)
            total_reward += reward_step
            lines_cleared_this_step = info.get("lines_cleared", 0)
            
            # Render
            frame = env.render()
            if frame is not None:
                frame = cv2.resize(frame, (frame.shape[1]*4, frame.shape[0]*4), interpolation=cv2.INTER_NEAREST)
                
                # Display stats on frame
                y_pos = 30
                cv2.putText(frame, f"Score: {total_score}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                y_pos += 30
                cv2.putText(frame, f"Lines: {total_lines}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                y_pos += 30
                cv2.putText(frame, f"Pieces: {pieces_placed}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                y_pos += 30
                cv2.putText(frame, f"1x: {line_clears[1]}  2x: {line_clears[2]}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                y_pos += 25
                cv2.putText(frame, f"3x: {line_clears[3]}  4x: {line_clears[4]}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                y_pos += 30
                # High score display with gold color
                cv2.putText(frame, f"Highscore: {highscore['score']}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2, cv2.LINE_AA)
                
                cv2.imshow("Tetris DQN Play", frame)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    terminated = True
                    break
        
        # Update statistics after piece placement
        pieces_placed += 1
        total_lines += lines_cleared_this_step
        
        # Calculate score based on the same reward formula as training
        if lines_cleared_this_step >= 4:  # Tetris
            step_score = lines_cleared_this_step * 200
            if last_clear_was_tetris:
                step_score += 400  # Back-to-back tetris bonus
            last_clear_was_tetris = True
            line_clears[4] += 1
        elif lines_cleared_this_step > 0:  # Regular clear
            step_score = lines_cleared_this_step * 100
            last_clear_was_tetris = False
            line_clears[lines_cleared_this_step] += 1
        else:
            step_score = 0
        
        total_score += step_score

    # Game Over - Show Results
    print("\n" + "="*50)
    print("💀 GAME OVER!")
    print("="*50)
    print(f"Final Score:        {total_score}")
    print(f"Total Lines:        {total_lines}")
    print(f"Pieces Placed:      {pieces_placed}")
    print(f"Total Reward:       {total_reward}")
    print("\nLine Clears Breakdown:")
    print(f"  Singles (1x):     {line_clears[1]}")
    print(f"  Doubles (2x):     {line_clears[2]}")
    print(f"  Triples (3x):     {line_clears[3]}")
    print(f"  Tetrises (4x):    {line_clears[4]} 🎯")
    
    # Check for high score
    is_highscore = total_score > highscore['score']
    if is_highscore:
        print("\n" + "🎉" * 25)
        print("🏆 NEW HIGH SCORE! 🏆")
        print("🎉" * 25)
        print(f"Previous: {highscore['score']} → New: {total_score}")
        highscore = {'score': total_score, 'lines': total_lines, 'tetrises': line_clears[4]}
        save_highscore(total_score, total_lines, line_clears[4])
    else:
        print(f"\nHigh Score: {highscore['score']} (Lines: {highscore['lines']}, Tetrises: {highscore['tetrises']})")
    
    print("="*50)
    print("Press Ctrl+C to quit or wait for next game...")
    print("="*50)
    
    # Wait a bit before restarting
    cv2.waitKey(2000)  # 2 second pause
