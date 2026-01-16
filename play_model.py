import gymnasium as gym
import torch
import cv2
import numpy as np

from tetris_gymnasium.envs.tetris import Tetris
from tetris_dqn import DQN, get_state_channels  # Import the DQN class from your training script

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
action_dim = env.action_space.n

# ------------------------
# Load trained model
# ------------------------
policy_net = DQN(state_dim, action_dim).to(device)
checkpoint = torch.load("best_model.pth", map_location=device)
policy_net.load_state_dict(checkpoint["policy_state_dict"])
policy_net.eval()

# ------------------------
# Play
# ------------------------
terminated = False
total_reward = 0
total_lines = 0  # Track lines cleared

while not terminated:
    # Prepare state tensor (add batch dim)
    state_tensor = torch.tensor(state[None, :, :, :], dtype=torch.float32).to(device)

    # Choose action (greedy)
    with torch.no_grad():
        action = policy_net(state_tensor).argmax().item()

    # Step environment
    next_obs, reward, terminated, truncated, info = env.step(action)

    # Update state
    state = get_state_channels(env)
    total_reward += reward

    # Update lines cleared
    total_lines += info.get("lines_cleared", 0)

    # Render
    frame = env.render()
    if frame is not None:
        # Resize frame
        frame = cv2.resize(frame, (frame.shape[1]*4, frame.shape[0]*4), interpolation=cv2.INTER_NEAREST)

        # Display total lines
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
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

print("Game Over!")
print("Total Reward:", total_reward)
print("Total Lines Cleared:", total_lines)

env.close()
cv2.destroyAllWindows()
