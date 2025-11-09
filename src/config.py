"""
Centralized configuration file.
Contains all hyperparameters, file paths, and model settings

"""

import numpy as np
import torch
from pathlib import Path
from typing import Tuple

# File Paths
PROJECT_ROOT: Path = Path(__file__).parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_FILE_PATH: Path = DATA_DIR / "xy_data.csv"

# Data Parameters
# Parameters for the 't' vector
T_MIN: float = 6.0
T_MAX: float = 60.0
T_POINTS: int = 100

# Model Parameter Bounds
# We use numpy for pi, but convert to float for consistency
THETA_BOUNDS: Tuple[float, float] = (1e-9, 50.0 * np.pi / 180.0)
M_BOUNDS: Tuple[float, float] = (-0.05, 0.05)
X_BOUNDS: Tuple[float, float] = (1e-9, 100.0)

# Training Hyperparameters
N_EPOCHS: int = 8000  # Iterations per optimization run
N_RUNS: int = 100        # Number of runs to find local minima
INITIAL_LR: float = 1e-2  # Starting learning rate for Adam
SCHEDULER_PATIENCE: int = 200 # Epochs to wait before reducing LR
SCHEDULER_FACTOR: float = 0.1 # Factor to reduce LR by (e.g., 0.1)

# Reproducibility
SEED: int = 42

# Device
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"