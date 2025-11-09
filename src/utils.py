"""
Utility functions for logging and reproducibility.
"""

import logging
import os
import random
import sys

import numpy as np
import torch

from . import config

# Get a logger for this module, though setup_logging configures the root
logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """
    Configures the root logger for the project.
    
    Logs will be sent to stdout with a clean, consistent format.
    """
    # Use a slightly more aligned format
    log_format = "%(asctime)s [%(levelname)-8s] [%(name)s] %(message)s"
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # Suppress verbose logging from common libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    
    logger.info("Root logger configured.")


def set_seed(seed: int = config.SEED) -> None:
    """
    Sets the random seed for all relevant libraries to ensure
    reproducible results.
    
    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Seeding for CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you're using multi-GPU
    
    # These two are important for full reproducibility with CUDA
    # Note: This can impact performance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Global seed set to {seed}")