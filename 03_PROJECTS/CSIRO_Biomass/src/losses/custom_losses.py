"""
Custom Loss Functions for CSIRO Biomass Prediction

This module implements robust loss functions optimized for regression tasks
with outliers, specifically designed for biomass prediction.

Author: AIForge Team
Date: 2025-01-08
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HuberLoss(nn.Module):
    """
    Huber Loss (Smooth L1 Loss)
    
    Combines MSE for small errors and MAE for large errors.
    More robust to outliers than pure MSE.
    
    Args:
        delta (float): Threshold at which to change between L1 and L2 loss.
                      Default: 1.0
        reduction (str): Specifies the reduction to apply to the output.
                        Options: 'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(self, delta=1.0, reduction='mean'):
        super().__init__()
        self.delta = delta
        self.reduction = reduction
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Tensor of shape (batch_size, num_targets)
            targets: Tensor of shape (batch_size, num_targets)
            
        Returns:
            loss: Scalar tensor
        """
        errors = predictions - targets
        abs_errors = torch.abs(errors)
        
        # Huber loss formula
        quadratic = torch.min(abs_errors, torch.tensor(self.delta))
        linear = abs_errors - quadratic
        loss = 0.5 * quadratic ** 2 + self.delta * linear
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class QuantileLoss(nn.Module):
    """
    Quantile Loss (Pinball Loss)
    
    Predicts a specific quantile of the target distribution.
    Extremely robust to outliers.
    
    Args:
        quantile (float): The quantile to predict. Default: 0.5 (median)
        reduction (str): Specifies the reduction to apply to the output.
                        Options: 'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(self, quantile=0.5, reduction='mean'):
        super().__init__()
        self.quantile = quantile
        self.reduction = reduction
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Tensor of shape (batch_size, num_targets)
            targets: Tensor of shape (batch_size, num_targets)
            
        Returns:
            loss: Scalar tensor
        """
        errors = targets - predictions
        loss = torch.max(
            (self.quantile - 1) * errors,
            self.quantile * errors
        )
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class R2Loss(nn.Module):
    """
    R2 Loss (Coefficient of Determination Loss)
    
    Directly optimizes the R2 metric.
    R2 = 1 - (SS_res / SS_tot)
    
    Args:
        reduction (str): Specifies the reduction to apply to the output.
                        Options: 'none' | 'mean' | 'sum'. Default: 'mean'
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Tensor of shape (batch_size, num_targets)
            targets: Tensor of shape (batch_size, num_targets)
            
        Returns:
            loss: Scalar tensor (1 - R2, so lower is better)
        """
        # Calculate SS_res (sum of squared residuals)
        ss_res = torch.sum((targets - predictions) ** 2, dim=0)
        
        # Calculate SS_tot (total sum of squares)
        ss_tot = torch.sum((targets - targets.mean(dim=0, keepdim=True)) ** 2, dim=0)
        
        # Calculate R2
        r2 = 1 - (ss_res / (ss_tot + 1e-8))  # Add epsilon to avoid division by zero
        
        # Loss is 1 - R2 (so we minimize)
        loss = 1 - r2
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class MultiTaskLoss(nn.Module):
    """
    Multi-Task Loss with Learned Weights
    
    Combines multiple task losses with learnable weights.
    Uses uncertainty weighting (Kendall et al., 2018).
    
    Args:
        num_tasks (int): Number of tasks. Default: 5 (for CSIRO Biomass)
        base_loss (nn.Module): Base loss function for each task. Default: HuberLoss
    """
    def __init__(self, num_tasks=5, base_loss=None):
        super().__init__()
        self.num_tasks = num_tasks
        self.base_loss = base_loss if base_loss is not None else HuberLoss()
        
        # Learnable log variance for each task
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Tensor of shape (batch_size, num_tasks)
            targets: Tensor of shape (batch_size, num_tasks)
            
        Returns:
            loss: Scalar tensor
        """
        total_loss = 0
        
        for i in range(self.num_tasks):
            # Calculate loss for each task
            task_loss = self.base_loss(
                predictions[:, i:i+1],
                targets[:, i:i+1]
            )
            
            # Uncertainty weighting
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * task_loss + self.log_vars[i]
        
        return total_loss


class MMDLoss(nn.Module):
    """
    Maximum Mean Discrepancy (MMD) Loss for Domain Adaptation
    
    Measures the distance between two distributions using kernel methods.
    Used to align source (train) and target (test) domains.
    
    Args:
        kernel (str): Kernel type. Options: 'gaussian' | 'linear'. Default: 'gaussian'
        bandwidth (float): Bandwidth for Gaussian kernel. Default: 1.0
    """
    def __init__(self, kernel='gaussian', bandwidth=1.0):
        super().__init__()
        self.kernel = kernel
        self.bandwidth = bandwidth
        
    def gaussian_kernel(self, x, y):
        """Gaussian (RBF) kernel"""
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        
        x = x.unsqueeze(1)  # (x_size, 1, dim)
        y = y.unsqueeze(0)  # (1, y_size, dim)
        
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
        return torch.exp(-kernel_input / self.bandwidth)
    
    def linear_kernel(self, x, y):
        """Linear kernel"""
        return torch.mm(x, y.t())
    
    def forward(self, source_features, target_features):
        """
        Args:
            source_features: Features from source domain (train set)
            target_features: Features from target domain (test set)
            
        Returns:
            mmd_loss: Scalar tensor
        """
        if self.kernel == 'gaussian':
            kernel = self.gaussian_kernel
        else:
            kernel = self.linear_kernel
        
        # Calculate kernel matrices
        xx = kernel(source_features, source_features)
        yy = kernel(target_features, target_features)
        xy = kernel(source_features, target_features)
        
        # MMD^2 = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]
        mmd_loss = xx.mean() + yy.mean() - 2 * xy.mean()
        
        return mmd_loss


def get_loss_function(loss_name='huber', **kwargs):
    """
    Factory function to get loss function by name.
    
    Args:
        loss_name (str): Name of the loss function.
                        Options: 'mse' | 'mae' | 'huber' | 'quantile' | 'r2' | 'multitask'
        **kwargs: Additional arguments for the loss function.
        
    Returns:
        loss_fn: Loss function instance
    """
    loss_functions = {
        'mse': nn.MSELoss,
        'mae': nn.L1Loss,
        'huber': HuberLoss,
        'quantile': QuantileLoss,
        'r2': R2Loss,
        'multitask': MultiTaskLoss,
        'mmd': MMDLoss
    }
    
    if loss_name.lower() not in loss_functions:
        raise ValueError(f"Unknown loss function: {loss_name}. "
                        f"Available: {list(loss_functions.keys())}")
    
    return loss_functions[loss_name.lower()](**kwargs)


# Example usage
if __name__ == "__main__":
    # Test loss functions
    batch_size = 32
    num_targets = 5
    
    predictions = torch.randn(batch_size, num_targets)
    targets = torch.randn(batch_size, num_targets)
    
    # Test Huber Loss
    huber = HuberLoss(delta=1.0)
    loss = huber(predictions, targets)
    print(f"Huber Loss: {loss.item():.4f}")
    
    # Test Quantile Loss
    quantile = QuantileLoss(quantile=0.5)
    loss = quantile(predictions, targets)
    print(f"Quantile Loss: {loss.item():.4f}")
    
    # Test R2 Loss
    r2 = R2Loss()
    loss = r2(predictions, targets)
    print(f"R2 Loss: {loss.item():.4f}")
    
    # Test Multi-Task Loss
    mtl = MultiTaskLoss(num_tasks=5)
    loss = mtl(predictions, targets)
    print(f"Multi-Task Loss: {loss.item():.4f}")
    
    # Test MMD Loss
    source_features = torch.randn(32, 128)
    target_features = torch.randn(32, 128)
    mmd = MMDLoss()
    loss = mmd(source_features, target_features)
    print(f"MMD Loss: {loss.item():.4f}")
