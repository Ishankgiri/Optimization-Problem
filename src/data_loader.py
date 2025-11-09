"""
Handles loading and preprocessing of the data.
This module separates data I/O and transformation logic from the
model definition and training loops, following a good OOP practice.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch

from . import config

# Initialize logger for this module
logger = logging.getLogger(__name__)


def load_data(file_path: Path) -> Optional[pd.DataFrame]:
    """
    Loads the CSV data from the specified file path.
    
    Args:
        file_path (Path): The path to the xy_data.csv file.
    
    Returns:
        Optional[pd.DataFrame]: A DataFrame with 'x' and 'y' columns,
                              
    """
    if not file_path.exists():
        logger.error(f"Data file not found at {file_path}.")
        logger.error("Please create the 'data/' directory and place 'xy_data.csv' inside.")
        return None
        
    try:
        data = pd.read_csv(file_path)
        if 'x' not in data.columns or 'y' not in data.columns:
            logger.error(f"File {file_path} must contain 'x' and 'y' columns.")
            return None
            
        logger.info(f"Successfully loaded data from {file_path} ({len(data)} rows)")
        return data
        
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading {file_path}: {e}")
        return None


def preprocess_data(
    df: pd.DataFrame, 
    device: str = config.DEVICE
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts DataFrame data to PyTorch tensors and generates the 't' vector.
    
    Args:
        df (pd.DataFrame): The input DataFrame (from load_data).
        device (str): The device to move the tensors to ('cpu' or 'cuda').
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - x_data (tensor): The 'x' column data.
            - y_data (tensor): The 'y' column data.
            - t_vec (tensor): The generated 't' vector.
    """
    # Convert data columns to tensors
    x_data = torch.tensor(
        df['x'].values, dtype=torch.float32
    ).to(device)
    
    y_data = torch.tensor(
        df['y'].values, dtype=torch.float32
    ).to(device)
    
    # Create the 't' vector used for generating the predicted curve
    t_vec = torch.linspace(
        config.T_MIN, config.T_MAX, config.T_POINTS, dtype=torch.float32
    ).to(device)
    
    logger.info(f"Data preprocessed and moved to device: {device}")
    return x_data, y_data, t_vec