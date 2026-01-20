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
from tetris_dqn import VNN, Agent, TetrisEnv

A = ActionsMapping()

# Fixed to 5-feature approach
feature_dim = 5
print("🔧 Using 5-feature extraction (smoothness, complete_lines, holes, min_height, max_height)")

# ------------------------
# Command Line Arguments
# ------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Play Tetris with trained VNN model')
    parser.add_argument('--model', type=str, choices=['best', 'latest', 'checkpoint', 'history', 'type', 'list'], default='best',
                       help='Choose model: "best" (best_model.pth - latest best), "latest" (latest checkpoint), "checkpoint" (specific checkpoint), "history" (from best_models folder), "type" (from best_model_types folder), "list" (list available models)')
    parser.add_argument('--path', type=str, 
                       help='Specific model file path (e.g., checkpoints/tetris_dqn_ep1000.pth or best_model_types/survival_focused.pth)')
    parser.add_argument('--episode', type=int,
                       help='Specific episode number to load from best_models history (e.g., --model history --episode 500)')
    parser.add_argument('--name', type=str,
                       help='Model name from best_model_types (e.g., --model type --name survival_focused)')
    return parser.parse_args()

def list_available_models():
    """List all available models"""
    print("\n📋 Available Models:\n")
    
    # Best model (main)
    if os.path.exists("best_model.pth"):
        stat = os.stat("best_model.pth")
        print(f"✅ best_model.pth (latest best model)")
    
    # Best model types (different training methods)
    type_models = glob.glob("best_model_types/*.pth")
    if type_models:
        print(f"\n🎯 Best Model Types ({len(type_models)} models - different training methods):")
        for model in sorted(type_models):
            model_name = os.path.basename(model).replace('.pth', '')
            # Check for corresponding training script
            script_path = model.replace('.pth', '.py')
            if os.path.exists(script_path):
                print(f"   - {model_name} (with training script)")
            else:
                print(f"   - {model_name}")
    
    # Best models history
    best_models = sorted(glob.glob("best_models/best_model_*.pth"))
    if best_models:
        print(f"\n📜 Best Models History ({len(best_models)} models):")
        for model in best_models[-10:]:  # Show last 10
            print(f"   - {model}")
        if len(best_models) > 10:
            print(f"   ... and {len(best_models) - 10} more")
    
    # Checkpoints
    checkpoints = sorted(glob.glob("checkpoints/tetris_dqn_*.pth"))
    if checkpoints:
        print(f"\n💾 Checkpoints ({len(checkpoints)} files):")
        for ckpt in checkpoints[-5:]:  # Show last 5
            print(f"   - {ckpt}")
        if len(checkpoints) > 5:
            print(f"   ... and {len(checkpoints) - 5} more")
    
    print("\n💡 Usage examples:")
    print("   python play_model.py --model best              # Use latest best model")
    print("   python play_model.py --model latest            # Use latest checkpoint")
    print("   python play_model.py --model type --name survival_focused  # Use specific training type")
    print("   python play_model.py --model history --episode 500  # Use best model from episode 500")
    print("   python play_model.py --model checkpoint --path checkpoints/tetris_dqn_ep1000.pth")
    print("   python play_model.py --model list              # Show this list\n")
    exit(0)

def get_model_path(args):
    if args.model == 'list':
        list_available_models()
    
    if args.path:
        # Use specific file path
        if os.path.exists(args.path):
            return args.path
        else:
            print(f"❌ Model file not found: {args.path}")
            exit(1)
    
    elif args.model == 'best':
        # Use latest best model
        if os.path.exists("best_model.pth"):
            return "best_model.pth"
        else:
            print("❌ best_model.pth not found!")
            exit(1)
    
    elif args.model == 'history':
        # Use model from best_models history
        if args.episode:
            # Find model with specific episode number
            pattern = f"best_models/best_model_ep{args.episode}_*.pth"
            matches = glob.glob(pattern)
            if matches:
                return matches[0]
            else:
                print(f"❌ No best model found for episode {args.episode}")
                print(f"   Searched for: {pattern}")
                exit(1)
        else:
            # Find the latest best model from history
            best_models = glob.glob("best_models/best_model_*.pth")
            if best_models:
                latest = max(best_models, key=os.path.getmtime)
                return latest
            else:
                print("❌ No models found in best_models/ directory!")
                exit(1)
    
    elif args.model == 'type':
        # Use model from best_model_types (different training methods)
        if args.name:
            # Look for model with specific name
            model_path = f"best_model_types/{args.name}.pth"
            if os.path.exists(model_path):
                # Check if there's a corresponding training script
                script_path = f"best_model_types/{args.name}.py"
                if os.path.exists(script_path):
                    print(f"📄 Training script available: {script_path}")
                return model_path
            else:
                print(f"❌ Model not found: {model_path}")
                print(f"   Available models in best_model_types/:")
                type_models = glob.glob("best_model_types/*.pth")
                if type_models:
                    for m in sorted(type_models):
                        print(f"      - {os.path.basename(m).replace('.pth', '')}")
                else:
                    print(f"      (no models found)")
                exit(1)
        else:
            print("❌ --name required when using --model type")
            print("   Example: python play_model.py --model type --name survival_focused")
            print("\n   Available models:")
            type_models = glob.glob("best_model_types/*.pth")
            if type_models:
                for m in sorted(type_models):
                    print(f"      - {os.path.basename(m).replace('.pth', '')}")
            else:
                print(f"      (no models found)")
            exit(1)
    
    elif args.model == 'checkpoint':
        if args.path:
            if os.path.exists(args.path):
                return args.path
            else:
                print(f"❌ Checkpoint file not found: {args.path}")
                exit(1)
        else:
            print("❌ --path required when using --model checkpoint")
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
    
    else:
        print(f"❌ Unknown model option: {args.model}")
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
tetris_env = TetrisEnv(render_mode="rgb_array")

# ------------------------
# Load trained model
# ------------------------
policy_net = VNN(feature_dim).to(device)
print(f"\n📂 Loading model from: {model_path}")
checkpoint = torch.load(model_path, map_location=device, weights_only=False)

try:
    policy_net.load_state_dict(checkpoint["policy_state_dict"])
    policy_net.eval()
    
    # Print model information
    episode = checkpoint.get("episode", "Unknown")
    epsilon = checkpoint.get("epsilon", "Unknown")
    reward = checkpoint.get("reward", "Unknown")
    
    print(f"✅ Model loaded successfully!")
    print(f"📊 Model Info:")
    print(f"   - Episode: {episode}")
    if reward != "Unknown":
        print(f"   - Best Reward: {reward:.2f}")
    if epsilon != "Unknown":
        print(f"   - Epsilon at save: {epsilon:.4f}")
    print("-" * 50)
    
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

try:
    while True:  # Infinite loop - play until user quits
        game_count += 1
        obs, info = tetris_env.reset()
        terminated = False
        total_reward = 0
        total_lines = 0
        total_score = 0
        pieces_placed = 0
        line_clears = {1: 0, 2: 0, 3: 0, 4: 0}  # Track single, double, triple, tetris
        
        print(f"\n{'='*50}")
        print(f"🎮 GAME #{game_count} START")
        print(f"{'='*50}\n")

        while not terminated:
            tetro = tetris_env.env.unwrapped.active_tetromino
            tetro_idx = tetro.id - 2  # Convert tetromino ID (2-8) to index (0-6)

            # 1️⃣ Generate all possible placements for current piece
            placements = tetris_env.get_all_board_states(tetro_idx, tetro)

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
                obs, reward_step, terminated, truncated, info = tetris_env.step(a)
                total_reward += reward_step
                lines_cleared_this_step = info.get("lines_cleared", 0)
                
                # Render
                frame = tetris_env.render()
                if frame is not None:
                    frame = cv2.resize(frame, (frame.shape[1]*4, frame.shape[0]*4), interpolation=cv2.INTER_NEAREST)
                    
                    # Display stats on frame (lower right corner)
                    frame_height = frame.shape[0]
                    frame_width = frame.shape[1]
                    
                    # Start from bottom and work upwards
                    y_pos = frame_height - 30
                    
                    # High score display with gold color
                    text = f"Highscore: {highscore['score']}"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
                    cv2.putText(frame, text, (frame_width - text_size[0] - 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 215, 255), 3, cv2.LINE_AA)
                    y_pos -= 45
                    
                    text = f"3x: {line_clears[3]}  4x: {line_clears[4]}"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                    cv2.putText(frame, text, (frame_width - text_size[0] - 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                    y_pos -= 40
                    
                    text = f"1x: {line_clears[1]}  2x: {line_clears[2]}"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                    cv2.putText(frame, text, (frame_width - text_size[0] - 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                    y_pos -= 45
                    
                    text = f"Pieces: {pieces_placed}"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
                    cv2.putText(frame, text, (frame_width - text_size[0] - 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
                    y_pos -= 45
                    
                    text = f"Lines: {total_lines}"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
                    cv2.putText(frame, text, (frame_width - text_size[0] - 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
                    y_pos -= 45
                    
                    text = f"Score: {total_score}"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
                    cv2.putText(frame, text, (frame_width - text_size[0] - 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
                    
                    cv2.imshow("Tetris VNN Play", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        terminated = True
                        break
            
            # Update statistics after piece placement
            pieces_placed += 1
            total_lines += lines_cleared_this_step
            
            # Calculate score based on the same reward formula as training
            if lines_cleared_this_step >= 4:  # Tetris
                step_score = 1200  # Fixed tetris reward
                line_clears[4] += 1
            elif lines_cleared_this_step > 0:  # Regular clear (1-3 lines)
                step_score = lines_cleared_this_step * 100
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

except KeyboardInterrupt:
    print("\n\n" + "="*50)
    print("⚠️  Interrupted by user (Ctrl+C)")
    print("="*50)
    
    # Check if there's a current game score to save
    if 'total_score' in locals() and total_score > highscore['score']:
        print(f"\n🎉 Saving new high score: {total_score}")
        print(f"   Previous high score was: {highscore['score']}")
        highscore = {'score': total_score, 'lines': total_lines, 'tetrises': line_clears[4]}
        save_highscore(total_score, total_lines, line_clears[4])
        print("✅ High score saved!")
    else:
        print(f"\n💾 High score remains: {highscore['score']}")
    
    print("\n👋 Thanks for playing! Goodbye!")
    print("="*50)

finally:
    # Clean up
    tetris_env.close()
    cv2.destroyAllWindows()
