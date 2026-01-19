"""
Visual T-piece placement monitor with pause on suspicious placements
Press SPACE to continue, Q to quit
"""
import sys
import os
# Add parent directory to path for imports when in debug folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
import torch
import cv2
from tetris_dqn import (
    DQN, get_all_board_states, extract_features, device
)
from tetris_gymnasium.mappings.actions import ActionsMapping

A = ActionsMapping()

def visualize_board_console(board, title="Board"):
    """Print board"""
    H, W = board.shape
    print(f"\n{title}:")
    print("  " + "".join([str(i) for i in range(W)]))
    for row_idx, row in enumerate(board):
        row_str = "".join(["█" if cell > 0 else "·" for cell in row])
        print(f"{row_idx:2d} {row_str}")

# Load model (4 features, reference network)
policy_net = DQN(4, reference_network=True).to(device)
checkpoint = torch.load("best_model_features.pth", map_location=device)
policy_net.load_state_dict(checkpoint["policy_state_dict"])
policy_net.eval()

print("🔍 T-Piece Placement Monitor")
print("=" * 80)
print("Instructions:")
print("  - Game will PAUSE when T piece is placed")
print("  - Check if visual board matches reported holes")
print("  - Press SPACE to continue, Q to quit")
print("=" * 80)

env = gym.make("tetris_gymnasium/Tetris", render_mode="human")
obs, info = env.reset()

piece_count = 0
t_count = 0

try:
    while True:
        tetro = env.unwrapped.active_tetromino
        tetro_id = tetro.id
        tetro_idx = tetro_id - 2
        
        piece_count += 1
        is_t_piece = (tetro_idx == 0)
        
        if is_t_piece:
            t_count += 1
            print(f"\n{'='*80}")
            print(f"🔷 T PIECE #{t_count} (Piece #{piece_count})")
            print(f"{'='*80}")
            
            # Get board before placement
            board_before = env.unwrapped.board.copy()
            H, W = board_before.shape
            playable_before = board_before[:H-4, 4:W-4]
        
        # Get all placements
        placements = get_all_board_states(env, tetro_idx, tetro)
        for i in range(len(placements)):
            print(placements[i]['features'])
            print(placements[i]['board'])
            print(placements[i]['actions'])
            print(f"index: {i}")
            
        if not placements:
            obs, info = env.reset()
            continue
        
        # Get Q-values
        feature_vectors = np.array([p["features"] for p in placements], dtype=np.float32)
        features_batch = torch.tensor(feature_vectors, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            q_values = policy_net(features_batch).squeeze()
        
        chosen_idx = torch.argmax(q_values).item()
        chosen = placements[chosen_idx]
        
        if is_t_piece:
            features = chosen['features']
            print(f"\n� DEBUG: Chosen placement index {chosen_idx} out of {len(placements)} options")
            print(f"   Raw features array: {features}")
            print(f"\n�📊 CHOSEN PLACEMENT (from {len(placements)} options):")
            print(f"   Q={q_values[chosen_idx].item():.2f}")
            print(f"   x={chosen['x']}, rot={chosen['rotation']}")
            print(f"   Features from get_all_board_states():")
            print(f"     - aggregate_height: {features[0]:.3f} (×200 = {features[0]*200:.1f})")
            print(f"     - complete_lines:   {features[1]:.3f} (×4 = {features[1]*4:.1f})")
            print(f"     - holes:            {features[2]:.3f} (×10 = {features[2]*10:.1f})")
            print(f"     - bumpiness:        {features[3]:.3f} (×50 = {features[3]*50:.1f})")
            print(f"\n   ⚠️  These are features from the SIMULATED board (with T piece already placed)")
        
        # Execute placement
        for action in chosen["actions"]:
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            cv2.waitKey(50)
        
        if is_t_piece:
            # Get board after placement
            board_after = env.unwrapped.board.copy()
            H, W = board_after.shape
            playable_after = board_after[:H-4, 4:W-4]
            
            visualize_board_console(playable_after, "BOARD AFTER T PLACEMENT")
            
            # Use extract_features to count holes (same as training)
            actual_features_dict = extract_features(playable_after)
            actual_holes = actual_features_dict['holes']
            
            reported_holes = features[2] * 10
            
            print(f"\n📊 FEATURES:")
            print(f"   Simulated: {features}")
            print(f"   Actual:    aggregate_height={actual_features_dict['aggregate_height']}, "
                  f"lines={actual_features_dict['complete_lines']}, "
                  f"holes={actual_holes}, "
                  f"min_height={actual_features_dict['min_height']}, "
                  f"max_height={actual_features_dict['max_height']}")
            
            print(f"\n🔍 VERIFICATION:")
            print(f"   SIMULATED holes (from get_all_board_states): {reported_holes:.1f}")
            print(f"   ACTUAL holes (from real board):              {actual_holes}")
            
            if abs(actual_holes - reported_holes) > 0.5:
                print(f"\n❌ MISMATCH DETECTED! Difference: {abs(actual_holes - reported_holes):.1f}")
                print(f"   Our get_all_board_states() predicted {reported_holes:.1f} holes")
                print(f"   But actual board after placement has {actual_holes} holes!")
                print(f"   This means the simulation is WRONG!")
            else:
                print(f"   ✅ Counts match! Simulation is working correctly")
            
            print(f"\n👀 Visually inspect the board above:")
            print(f"   - Holes are empty cells (·) with filled cells (█) above them")
            print(f"   - Count them manually to verify")
            print(f"\n   Press SPACE to continue, Q to quit...")
            
            # Wait for keypress
            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord(' '):  # Space
                    break
                elif key == ord('q'):  # Q
                    raise KeyboardInterrupt
        else:
            cv2.waitKey(50)
        
        if terminated:
            print(f"\n💀 Game Over! Played {piece_count} pieces, {t_count} T pieces")
            print("Starting new game...")
            obs, info = env.reset()
            piece_count = 0
            t_count = 0

except KeyboardInterrupt:
    print("\n\n⏸️  Stopped by user")

env.close()
