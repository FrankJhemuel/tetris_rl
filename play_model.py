import gymnasium as gym
import torch
import cv2
import numpy as np
import random

from tetris_gymnasium.envs.tetris import Tetris
from tetris_gymnasium.mappings.actions import ActionsMapping
from tetris_dqn import DQN, get_state_channels, get_all_board_states  # helpers from your training script

A = ActionsMapping()

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
checkpoint = torch.load("best_model.pth", map_location=device)
policy_net.load_state_dict(checkpoint["policy_state_dict"])
policy_net.eval()

# ------------------------
# Play loop
# ------------------------
terminated = False
total_reward = 0
total_lines = 0

while not terminated:
    tetro = env.unwrapped.active_tetromino
    tetro_idx = env.unwrapped.tetrominoes.index(tetro)

    # 1️⃣ Generate all possible placements for current piece
    placements = get_all_board_states(env, tetro_idx, tetro)

    # 2️⃣ Convert boards to batch tensor for DQN
    boards_batch = np.array([p["board"] for p in placements], dtype=np.float32)
    boards_batch = torch.tensor(boards_batch[:, np.newaxis, :, :], dtype=torch.float32).to(device)

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
