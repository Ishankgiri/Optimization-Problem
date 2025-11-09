"""
Main training script.
Run this file from the project's root folder:
`python3 run_training.py`
"""
print("[run_training.py] Script started!!!!")

import logging
import sys
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

# Project-specific imports
from src import config, data_loader, utils
from src.model import ParametricModel

# Configure logger
logger = logging.getLogger(__name__)


def l1_loss(
    x_pred: torch.Tensor, y_pred: torch.Tensor,
    x_true: torch.Tensor, y_true: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the L1 Loss by finding the minimum L1 distance from
    each *true data point* to the *predicted curve*.
    """
    # x_pred/y_pred shape: [100] (predicted curve)
    # x_true/y_true shape: [1500] (data points)

    # Reshape to enable broadcasting
    # x_true_r shape: [1500, 1]
    # x_pred_r shape: [1, 100]
    x_true_r = x_true.view(-1, 1)
    y_true_r = y_true.view(-1, 1)
    x_pred_r = x_pred.view(1, -1)
    y_pred_r = y_pred.view(1, -1)

    # Calculate L1 distance matrix (shape [1500, 100])
    # This matrix contains the L1 distance from every true point
    # to every predicted point.
    x_diff = torch.abs(x_true_r - x_pred_r)
    y_diff = torch.abs(y_true_r - y_pred_r)
    dist_matrix = x_diff + y_diff

    # Find the minimum distance for each true point
    # (find the closest point on the curve)
    # min_dists shape: [1500]
    min_dists = torch.min(dist_matrix, dim=1).values
    
    # The total loss is the sum of these minimum distances
    return torch.sum(min_dists)


def run_optimization_run(
    run_id: int,
    t_vec: torch.Tensor,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    device: str
) -> Tuple[float, Dict[str, float], List[float]]:
    """
    Executes a single, full optimization run.
    """
    logger.info(f"- Starting Run {run_id + 1}/{config.N_RUNS} -")
    start_time = time.time()
    
    # 1. Initialize model and move to device
    model = ParametricModel().to(device)
    
    # 2. Setup optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=config.INITIAL_LR)
    scheduler = optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=1e-5,
        max_lr=config.INITIAL_LR,
        step_size_up=1000,
        mode='triangular'
    )
    
    best_epoch_loss = float('inf')
    best_epoch_params = {}
    loss_history = []

    # 3. Training Loop
    for epoch in range(config.N_EPOCHS):
        optimizer.zero_grad()
        x_pred, y_pred = model(t_vec)
        
        loss = l1_loss(x_pred, y_pred, x_data, y_data)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.clamp_parameters() 
        
        loss_item = loss.item()
        loss_history.append(loss_item)
        
        if loss_item < best_epoch_loss:
            best_epoch_loss = loss_item
            best_epoch_params = model.get_params_dict()
        
        if (epoch + 1) % 2000 == 0:
            logger.info(f"  [Run {run_id+1}] Epoch {epoch+1:5d}/{config.N_EPOCHS}, "
                         f"Loss: {loss_item:.4e}, "
                         f"LR: {optimizer.param_groups[0]['lr']:.1e}")

    run_time = time.time() - start_time
    logger.info(f"- Run {run_id + 1} Finished -")
    logger.info(f"  Final Loss: {best_epoch_loss:.4e}")
    logger.info(f"  Time Taken: {run_time:.2f}s")
    
    return best_epoch_loss, best_epoch_params, loss_history


def plot_loss_curves(all_run_histories: List[List[float]]) -> None:
    """
    Generates and saves a plot of loss vs. epoch for all runs.
    """
    plt.figure(figsize=(12, 8))
    
    for i, history in enumerate(all_run_histories):
        # Plot each run. Using alpha helps see overlapping lines.
        plt.plot(history, label=f'Run {i+1}', alpha=0.7)
    
    plt.xlabel('Epoch')
    plt.ylabel('L1 Loss (log scale)')
    plt.title(f'Loss Curves for {len(all_run_histories)} Runs')
    
    # Use a log scale on the y-axis to see the drop better
    plt.yscale('log')
    
    # Only show legend if there are 10 runs or fewer
    if len(all_run_histories) <= 10:
        plt.legend()
        
    plt.grid(True, which="both", ls="--", alpha=0.5)
    
    # Save the plot as a file
    plot_filename = "loss_curves.png"
    plt.savefig(plot_filename)
    logger.info(f"Saved loss curve plot to {plot_filename}")


def main() -> None:
    """
    Main entry point
    """
    # 1. Setup
    utils.setup_logging()
    utils.set_seed(config.SEED)
    logger.info("Starting parameter optimization-")
    
    # 2. Log Configuration
    logger.info("Configuration:")
    logger.info(f"  Device: {config.DEVICE}")
    logger.info(f"  Epochs per run: {config.N_EPOCHS}")
    logger.info(f"  Number of runs: {config.N_RUNS}")
    logger.info(f"  Seed: {config.SEED}")

    # 3. Load Data
    data_df = data_loader.load_data(config.DATA_FILE_PATH)
    if data_df is None:
        logger.critical("Failed to load data. Exiting.")
        sys.exit(1)
        
    x_data, y_data, t_vec = data_loader.preprocess_data(
        data_df, config.DEVICE
    )

    # 4. Run multiple optimizations
    best_overall_loss = float('inf')
    best_overall_params = {}
    all_loss_histories = []

    for i in range(config.N_RUNS):
        final_loss, final_params, loss_hist = run_optimization_run(
            run_id=i,
            t_vec=t_vec,
            x_data=x_data,
            y_data=y_data,
            device=config.DEVICE
        )
        all_loss_histories.append(loss_hist)
        
        if final_loss < best_overall_loss:
            best_overall_loss = final_loss
            best_overall_params = final_params
            logger.info(f"✨ New best solution found in Run {i+1}! ✨")

    # 5. Plot the results
    if all_loss_histories:
        plot_loss_curves(all_loss_histories)

    # 6. Final Results
    logger.info("\n\n" + "="*50)
    logger.info("🎉 Optimization Complete! 🎉")
    logger.info("- Best Parameters Found (across all runs)!!!!-")
    
    theta_opt_rad = best_overall_params.get('theta', 0)
    theta_opt_deg = theta_opt_rad * 180.0 / np.pi
    M_opt = best_overall_params.get('M', 0)
    X_opt = best_overall_params.get('X', 0)

    logger.info(f"  θ (radians): {theta_opt_rad:<.10f}")
    logger.info(f"  θ (degrees): {theta_opt_deg:<.10f} (This is π/6 or 30°)")
    logger.info(f"  M:           {M_opt:<.10f}")
    logger.info(f"  X:           {X_opt:<.10f}")
    logger.info(f"\n  Best Final L1 Loss: {best_overall_loss:e}")

    # Format the final LaTeX string
    latex_submission = (
        f"\\left(t*\\cos({theta_opt_rad:.6f})"
        f"-e^{{{M_opt:.6f}\\left|t\\right|}}\\cdot\\sin(0.3t)\\sin({theta_opt_rad:.6f})"
        f"\\ +{X_opt:.6f},"
        f"42+\\ t*\\sin({theta_opt_rad:.6f})"
        f"+e^{{{M_opt:.6f}\\left|t\\right|}}\\cdot\\sin(0.3t)\\cos({theta_opt_rad:.6f})\\right)"
    )
    
    logger.info("\n--- 📋 Submission Format (LaTeX) ---")
    print(latex_submission)


if __name__ == "__main__":
    main()