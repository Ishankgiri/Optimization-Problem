"""
Defines the Parametric Model as a torch.nn.Module.
This file contains the model parameters (theta, M, X) and the
forward pass, which implements the equations.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn

from . import config


class ParametricModel(nn.Module):
    
    def __init__(self):
        """
        Parameters are initialized randomly within their specified bounds.
        """
        super().__init__()
        
        # Store bounds for later use in clamping
        self.bounds = {
            'theta': config.THETA_BOUNDS,
            'M': config.M_BOUNDS,
            'X': config.X_BOUNDS
        }
        
        # Helper for DRY (Don't Repeat Yourself) initialization
        def _init_param(bounds: Tuple[float, float]) -> torch.Tensor:
            """Initializes a random scalar tensor within the given bounds."""
            return torch.rand(1) * (bounds[1] - bounds[0]) + bounds[0]

        # Initialize parameters
        init_theta = _init_param(self.bounds['theta'])
        init_M = _init_param(self.bounds['M'])
        init_X = _init_param(self.bounds['X'])
        
        # Register them as nn.Parameter to make them trainable
        self.theta = nn.Parameter(init_theta)
        self.M = nn.Parameter(init_M)
        self.X = nn.Parameter(init_X)
        
    def forward(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward pass of the model.
        Calculates x_pred and y_pred based on the current parameters.
        
        Args:
            t (torch.Tensor): The input 't' vector.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - x_pred (tensor): The predicted 'x' values.
                - y_pred (tensor): The predicted 'y' values.
        """
        # Calculate the common exponential/sine term
        exp_sin_term = torch.exp(self.M * torch.abs(t)) * torch.sin(0.3 * t)
        
        # Calculate x_pred
        x_pred = (
            t * torch.cos(self.theta) 
            - exp_sin_term * torch.sin(self.theta) 
            + self.X
        )
        
        # Calculate y_pred
        y_pred = (
            42.0 + t * torch.sin(self.theta) 
            + exp_sin_term * torch.cos(self.theta)
        )
        
        return x_pred, y_pred

    @torch.no_grad()
    def clamp_parameters(self) -> None:
        """
        Enforces the parameter bounds in place.
        called after each optimizer step to ensure parameters
        stay within their predefined ranges.
        """
        self.theta.clamp_(self.bounds['theta'][0], self.bounds['theta'][1])
        self.M.clamp_(self.bounds['M'][0], self.bounds['M'][1])
        self.X.clamp_(self.bounds['X'][0], self.bounds['X'][1])

    def get_params_dict(self) -> Dict[str, float]:
        """
        Returns the current parameter values as a Python dictionary.
        
        Returns:
            Dict[str, float]: A dictionary of {param_name: param_value}.
        """
        return {
            'theta': self.theta.item(),
            'M': self.M.item(),
            'X': self.X.item()
        }