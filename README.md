# Tetris-RL

A Deep Q-Network (DQN) reinforcement learning project for playing Tetris using the `tetris-gymnasium` environment.


## Note

`tetris_dqn.py` is currently the same as `best_model_types/hybrid.py`.

## Folder Structure

- `tetris_dqn.py` — Main training script for the DQN agent.
- `play_model.py` — Script to play Tetris using a trained model.
- `hyperparams.yaml` — Hyperparameters for training.
- `requirements.txt` — Python dependencies.
- `best_model_types/` — Contains different model types, their weights (`.pth`), code, and visualizations.
    - `hybrid.py`, `survival_focused.py` — Model definitions.
    - `hybrid.pth`, `survival_focused.pth` — Trained model weights.
    - `hybrid.png`, `survival_focused.png` — Training graphs visualizations.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Training

To train a new DQN agent:
```bash
python tetris_dqn.py
```
- Training parameters can be adjusted in `hyperparams.yaml`.

## Playing with a Trained Model

To play Tetris using a trained model:
```bash
python play_model.py --model type --name hybrid
```
- Use `--model` to select the model source (`type`, `best`, `latest`, etc.).
- Use `--name` to specify the model type (e.g., `hybrid`, `survival_focused`).
- For more options, run:
  ```bash
  python play_model.py --help
  ```

## Model Types

- `best_model_types/hybrid.py` and `survival_focused.py` contain different reward strategies.
- Corresponding `.pth` files are the trained weights.
- `.png` files are training visualizations.


## CUDA / GPU Support

This project can utilize CUDA (GPU acceleration) if you have a compatible NVIDIA GPU and the appropriate PyTorch version installed. By default, the code will use the GPU if available, otherwise it will fall back to CPU.

**To enable CUDA:**

- Install the CUDA-enabled version of PyTorch. See the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) for the correct command for your system. For example:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```
- Device selection (`cuda` or `cpu`) is already set in the code. When you run the scripts, you will see a print statement indicating whether CUDA (GPU) or CPU is being used.
- Training and inference will automatically use the GPU if available.

You can check if CUDA is available in your environment with:
```python
import torch
print(torch.cuda.is_available())
```

## Notes

- Make sure you have the `tetris-gymnasium` package installed and working.
- The project uses PyTorch for neural networks.
- For custom experiments, modify `tetris_dqn.py` or the model files in `best_model_types/`.
