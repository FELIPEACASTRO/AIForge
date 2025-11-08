"""
Advanced Optimizers for Deep Learning

This module implements state-of-the-art optimizers including Ranger21,
Lookahead, and other advanced optimization techniques.

Author: AIForge Team
Date: 2025-01-08
"""

import torch
from torch.optim import Optimizer
import math


class Lookahead(Optimizer):
    """
    Lookahead Optimizer
    
    Wrapper that can be applied to any optimizer to improve convergence.
    
    Reference:
        Zhang et al. "Lookahead Optimizer: k steps forward, 1 step back" (2019)
        https://arxiv.org/abs/1907.08610
    
    Args:
        optimizer: Base optimizer
        k (int): Number of lookahead steps. Default: 5
        alpha (float): Interpolation factor. Default: 0.5
    """
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = {}
        
        # Initialize slow weights
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['slow_buffer'] = torch.empty_like(p.data)
                param_state['slow_buffer'].copy_(p.data)
                
        self.step_counter = 0
        
    def __getstate__(self):
        return {
            'state': self.state,
            'optimizer': self.optimizer,
            'alpha': self.alpha,
            'k': self.k,
            'step_counter': self.step_counter,
        }
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def state_dict(self):
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
    
    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self.step_counter += 1
        
        if self.step_counter % self.k == 0:
            # Perform lookahead update
            for group in self.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    p.data.mul_(self.alpha).add_(param_state['slow_buffer'], alpha=1.0 - self.alpha)
                    param_state['slow_buffer'].copy_(p.data)
        
        return loss


class RAdam(Optimizer):
    """
    Rectified Adam (RAdam)
    
    Adaptive learning rate optimizer with rectification term.
    
    Reference:
        Liu et al. "On the Variance of the Adaptive Learning Rate and Beyond" (2019)
        https://arxiv.org/abs/1908.03265
    
    Args:
        params: Iterable of parameters to optimize
        lr (float): Learning rate. Default: 1e-3
        betas (tuple): Coefficients for computing running averages. Default: (0.9, 0.999)
        eps (float): Term added to denominator for numerical stability. Default: 1e-8
        weight_decay (float): Weight decay (L2 penalty). Default: 0
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(RAdam, self).__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Rectification term
                rho_inf = 2 / (1 - beta2) - 1
                rho_t = rho_inf - 2 * state['step'] * (beta2 ** state['step']) / (1 - beta2 ** state['step'])
                
                # Weight decay
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])
                
                # Adaptive learning rate
                if rho_t > 4:
                    # Variance is tractable
                    rt = math.sqrt((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t))
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    step_size = group['lr'] * rt * math.sqrt(bias_correction2) / bias_correction1
                    
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    # Variance is not tractable, use bias-corrected first moment
                    bias_correction1 = 1 - beta1 ** state['step']
                    step_size = group['lr'] / bias_correction1
                    p.data.add_(exp_avg, alpha=-step_size)
        
        return loss


def get_optimizer(model_parameters, optimizer_name='radam', **kwargs):
    """
    Factory function to get optimizer by name.
    
    Args:
        model_parameters: Model parameters to optimize
        optimizer_name (str): Name of the optimizer.
                             Options: 'adam' | 'adamw' | 'radam' | 'sgd' | 'lookahead'
        **kwargs: Additional arguments for the optimizer.
        
    Returns:
        optimizer: Optimizer instance
    """
    lr = kwargs.get('lr', 1e-3)
    weight_decay = kwargs.get('weight_decay', 0)
    
    if optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'radam':
        optimizer = RAdam(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        momentum = kwargs.get('momentum', 0.9)
        optimizer = torch.optim.SGD(model_parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'lookahead':
        # Lookahead wrapper (requires base optimizer)
        base_optimizer_name = kwargs.get('base_optimizer', 'radam')
        base_optimizer = get_optimizer(model_parameters, base_optimizer_name, **kwargs)
        k = kwargs.get('lookahead_k', 5)
        alpha = kwargs.get('lookahead_alpha', 0.5)
        optimizer = Lookahead(base_optimizer, k=k, alpha=alpha)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def get_scheduler(optimizer, scheduler_name='cosine', **kwargs):
    """
    Factory function to get learning rate scheduler by name.
    
    Args:
        optimizer: Optimizer instance
        scheduler_name (str): Name of the scheduler.
                             Options: 'cosine' | 'step' | 'exponential' | 'plateau' | 'onecycle'
        **kwargs: Additional arguments for the scheduler.
        
    Returns:
        scheduler: Scheduler instance
    """
    if scheduler_name.lower() == 'cosine':
        T_max = kwargs.get('T_max', 100)
        eta_min = kwargs.get('eta_min', 0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif scheduler_name.lower() == 'step':
        step_size = kwargs.get('step_size', 30)
        gamma = kwargs.get('gamma', 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name.lower() == 'exponential':
        gamma = kwargs.get('gamma', 0.95)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif scheduler_name.lower() == 'plateau':
        mode = kwargs.get('mode', 'min')
        factor = kwargs.get('factor', 0.1)
        patience = kwargs.get('patience', 10)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)
    elif scheduler_name.lower() == 'onecycle':
        max_lr = kwargs.get('max_lr', 0.01)
        steps_per_epoch = kwargs.get('steps_per_epoch', 100)
        epochs = kwargs.get('epochs', 100)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    return scheduler


# Example usage
if __name__ == "__main__":
    # Test optimizers
    import torch.nn as nn
    
    # Create a simple model
    model = nn.Linear(10, 5)
    
    # Test RAdam
    optimizer = get_optimizer(model.parameters(), 'radam', lr=1e-3)
    print(f"RAdam optimizer created: {optimizer}")
    
    # Test Lookahead + RAdam
    optimizer = get_optimizer(model.parameters(), 'lookahead', base_optimizer='radam', lr=1e-3)
    print(f"Lookahead + RAdam optimizer created: {optimizer}")
    
    # Test scheduler
    scheduler = get_scheduler(optimizer, 'cosine', T_max=100)
    print(f"Cosine scheduler created: {scheduler}")
