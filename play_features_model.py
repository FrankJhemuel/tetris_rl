#!/usr/bin/env python3

import cv2
import gymnasium as gym
import numpy as np
import os
import torch
import torch.nn as nn
from tetris_dqn import DQN, get_all_board_states, extract_tetris_features, get_current_state_features
from tetris_gymnasium.mappings.actions import ActionsMapping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
A = ActionsMapping()

# Load the feature-based model
def play_with_features_model(model_path="best_model_features.pth", episodes=5):
    """Play Tetris using the feature-based DQN model"""
    
    # Create environment
    env = gym.make("tetris_gymnasium/Tetris", render_mode="rgb_array")
    
    # Load model
    feature_dim = 5
    
    # Try to load checkpoint first to determine network architecture
    checkpoint = None
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
    
    # Determine network architecture from checkpoint or default to simple
    use_simple = True  # Default to simple network
    if checkpoint and "use_simple_network" in checkpoint:
        use_simple = checkpoint["use_simple_network"]
    elif checkpoint:
        # For backwards compatibility, check model parameter count
        state_dict = checkpoint["policy_state_dict"]
        param_count = sum([torch.numel(p) for p in state_dict.values()])
        use_simple = param_count < 1000  # Simple network has ~400 params, complex has ~13K
    
    model = DQN(feature_dim, simple_network=use_simple).to(device)
    
    if checkpoint:
        model.load_state_dict(checkpoint["policy_state_dict"])
        
        # Print model information like in play_model.py
        episode = checkpoint.get("episode", "Unknown")
        epsilon = checkpoint.get("epsilon", "Unknown")
        architecture = "Simple (1 hidden layer)" if use_simple else "Complex (2 hidden layers)"
        print(f"✅ Loaded feature-based model: {model_path}")
        print(f"📋 Model architecture: {architecture}")
        print(f"📋 Model from episode: {episode}")
        if epsilon != "Unknown":
            print(f"📋 Model's epsilon at save time: {epsilon:.4f}")
        print("-" * 50)
    else:
        print("⚠️  No model found, using random policy")
    
    model.eval()
    
    for episode in range(episodes):
        obs, info = env.reset()
        terminated = False
        total_reward = 0
        lines_cleared = 0
        steps = 0
        
        print(f"\n🎮 Episode {episode + 1}")
        print("Features: [Roughness, Holes, Max_Height, Min_Height, Lines_Cleared]")
        
        while not terminated:
            tetro = env.unwrapped.active_tetromino
            tetro_idx = tetro.id - 2
            
            # Get current board features for display
            current_features = get_current_state_features(env)
            print(f"Step {steps:2d} | Current features: [{current_features[0]:.2f}, {current_features[1]:.2f}, {current_features[2]:.2f}, {current_features[3]:.2f}, {current_features[4]:.2f}]")
            
            # Get all possible placements
            placements = get_all_board_states(env, tetro_idx, tetro)
            if not placements:
                break
            
            # Evaluate all placements with the model
            feature_vectors = np.array([p["features"] for p in placements], dtype=np.float32)
            features_batch = torch.tensor(feature_vectors, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                if model_path and os.path.exists(model_path):
                    q_values = model(features_batch).squeeze()
                    chosen_idx = torch.argmax(q_values).item()
                    best_q_value = q_values[chosen_idx].item()
                else:
                    # Random policy if no model
                    chosen_idx = np.random.randint(0, len(placements))
                    best_q_value = 0.0
            
            chosen = placements[chosen_idx]
            chosen_features = feature_vectors[chosen_idx]
            
            print(f"         | Chosen placement: [{chosen_features[0]:.2f}, {chosen_features[1]:.2f}, {chosen_features[2]:.2f}, {chosen_features[3]:.2f}, {chosen_features[4]:.2f}] Q={best_q_value:.2f}")
            
            # Execute the chosen actions
            for action in chosen["actions"]:
                if terminated:
                    break
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Render with 4x size
                frame = env.render()
                if frame is not None:
                    frame = cv2.resize(frame, (frame.shape[1]*4, frame.shape[0]*4), interpolation=cv2.INTER_NEAREST)
                    cv2.putText(
                        frame,
                        f"Lines: {lines_cleared} | Steps: {steps} | Episode: {episode + 1}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )
                    cv2.imshow("Feature-based Tetris AI", frame)
                    if cv2.waitKey(100) & 0xFF == ord('q'):
                        terminated = True
                        break
            
            if info and "lines_cleared" in info:
                lines_this_step = info["lines_cleared"]
                lines_cleared += lines_this_step
                if lines_this_step > 0:
                    print(f"         | 🎉 Cleared {lines_this_step} line(s)! Total: {lines_cleared}")
            
            total_reward += reward
            steps += 1
            
        print(f"Episode {episode + 1} finished: {steps} steps, {lines_cleared} lines cleared, {total_reward:.1f} reward")
    
    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import os
    play_with_features_model()
