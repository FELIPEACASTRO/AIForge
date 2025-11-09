"""Custom Loss Functions for CSIRO Biomass Prediction"""

from .custom_losses import (
    HuberLoss,
    QuantileLoss,
    R2Loss,
    MultiTaskLoss,
    MMDLoss,
    get_loss_function
)

__all__ = [
    'HuberLoss',
    'QuantileLoss',
    'R2Loss',
    'MultiTaskLoss',
    'MMDLoss',
    'get_loss_function'
]
