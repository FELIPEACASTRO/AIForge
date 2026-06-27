"""
Ensemble and Stacking for CSIRO Biomass Prediction

This script implements advanced ensemble techniques:
- Simple averaging
- Weighted averaging (optimized weights)
- Stacking with CatBoost meta-learner
- Model diversity analysis

Author: AIForge Team
Date: 2025-01-08
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Simple Ensemble
# ============================================================================

def simple_average_ensemble(predictions_list: List[np.ndarray]) -> np.ndarray:
    """
    Simple average of predictions from multiple models
    
    Args:
        predictions_list: List of prediction arrays, each of shape (n_samples, n_targets)
        
    Returns:
        ensemble_predictions: Averaged predictions
    """
    return np.mean(predictions_list, axis=0)


def weighted_average_ensemble(
    predictions_list: List[np.ndarray],
    weights: List[float]
) -> np.ndarray:
    """
    Weighted average of predictions
    
    Args:
        predictions_list: List of prediction arrays
        weights: List of weights for each model (should sum to 1.0)
        
    Returns:
        ensemble_predictions: Weighted averaged predictions
    """
    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize
    
    ensemble_predictions = np.zeros_like(predictions_list[0])
    for pred, weight in zip(predictions_list, weights):
        ensemble_predictions += weight * pred
    
    return ensemble_predictions


def optimize_ensemble_weights(
    predictions_list: List[np.ndarray],
    targets: np.ndarray,
    method: str = 'grid_search'
) -> List[float]:
    """
    Optimize ensemble weights to maximize R2 score
    
    Args:
        predictions_list: List of prediction arrays from different models
        targets: Ground truth targets
        method: Optimization method ('grid_search' or 'scipy')
        
    Returns:
        optimal_weights: Optimized weights for each model
    """
    n_models = len(predictions_list)
    
    if method == 'grid_search':
        # Grid search over possible weight combinations
        best_r2 = -np.inf
        best_weights = None
        
        # Generate weight combinations
        from itertools import product
        weight_options = [i/10 for i in range(11)]  # 0.0, 0.1, ..., 1.0
        
        for weights in product(weight_options, repeat=n_models):
            if abs(sum(weights) - 1.0) > 0.01:  # Skip if doesn't sum to 1
                continue
            
            # Calculate ensemble predictions
            ensemble_pred = weighted_average_ensemble(predictions_list, weights)
            
            # Calculate R2
            r2 = r2_score(targets, ensemble_pred)
            
            if r2 > best_r2:
                best_r2 = r2
                best_weights = weights
        
        return list(best_weights)
    
    elif method == 'scipy':
        # Scipy optimization
        from scipy.optimize import minimize
        
        def objective(weights):
            ensemble_pred = weighted_average_ensemble(predictions_list, weights)
            r2 = r2_score(targets, ensemble_pred)
            return -r2  # Minimize negative R2
        
        # Constraints: weights sum to 1, all weights >= 0
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_models)]
        
        # Initial guess: equal weights
        x0 = np.ones(n_models) / n_models
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        return list(result.x)
    
    else:
        raise ValueError(f"Unknown method: {method}")


# ============================================================================
# Stacking with CatBoost
# ============================================================================

class StackingEnsemble:
    """
    Stacking Ensemble with CatBoost Meta-Learner
    
    Level 0: Multiple base models (DINOv2, EfficientNet, ConvNeXt, etc.)
    Level 1: CatBoost meta-learner that learns optimal combination
    
    Args:
        n_folds: Number of folds for out-of-fold predictions
        random_state: Random seed
    """
    def __init__(self, n_folds: int = 5, random_state: int = 42):
        self.n_folds = n_folds
        self.random_state = random_state
        self.meta_models = []  # One meta-model per target
        
    def fit(
        self,
        base_predictions_list: List[np.ndarray],
        targets: np.ndarray,
        verbose: bool = True
    ):
        """
        Train stacking ensemble
        
        Args:
            base_predictions_list: List of predictions from base models
                                  Each array has shape (n_samples, n_targets)
            targets: Ground truth targets (n_samples, n_targets)
            verbose: Whether to print progress
        """
        try:
            from catboost import CatBoostRegressor
        except ImportError:
            print("Warning: CatBoost not installed. Install with: pip install catboost")
            return
        
        # Stack predictions horizontally
        # Shape: (n_samples, n_models * n_targets)
        X_meta = np.hstack(base_predictions_list)
        
        n_targets = targets.shape[1]
        
        # Train one meta-model per target
        for target_idx in range(n_targets):
            if verbose:
                print(f"\nTraining meta-model for target {target_idx+1}/{n_targets}...")
            
            y_target = targets[:, target_idx]
            
            # CatBoost meta-learner
            meta_model = CatBoostRegressor(
                iterations=1000,
                learning_rate=0.05,
                depth=6,
                loss_function='RMSE',
                random_state=self.random_state,
                verbose=False
            )
            
            meta_model.fit(X_meta, y_target)
            self.meta_models.append(meta_model)
            
            # Evaluate
            meta_pred = meta_model.predict(X_meta)
            r2 = r2_score(y_target, meta_pred)
            mae = mean_absolute_error(y_target, meta_pred)
            
            if verbose:
                print(f"  R2: {r2:.4f}, MAE: {mae:.4f}")
    
    def predict(self, base_predictions_list: List[np.ndarray]) -> np.ndarray:
        """
        Generate stacked predictions
        
        Args:
            base_predictions_list: List of predictions from base models
            
        Returns:
            stacked_predictions: Final predictions (n_samples, n_targets)
        """
        # Stack predictions
        X_meta = np.hstack(base_predictions_list)
        
        # Predict with each meta-model
        stacked_predictions = []
        for meta_model in self.meta_models:
            pred = meta_model.predict(X_meta)
            stacked_predictions.append(pred)
        
        # Stack predictions vertically
        return np.column_stack(stacked_predictions)
    
    def get_feature_importance(self) -> Dict[str, List[float]]:
        """
        Get feature importance from meta-models
        
        Returns:
            importance_dict: Dictionary mapping target names to importance scores
        """
        target_names = ['Fresh_Weight', 'Dry_Weight', 'Height', 'Canopy_Size_1', 'Canopy_Size_2']
        
        importance_dict = {}
        for i, meta_model in enumerate(self.meta_models):
            importance = meta_model.get_feature_importance()
            importance_dict[target_names[i]] = importance.tolist()
        
        return importance_dict


# ============================================================================
# Model Diversity Analysis
# ============================================================================

def calculate_model_diversity(predictions_list: List[np.ndarray]) -> Dict[str, float]:
    """
    Calculate diversity metrics for ensemble models
    
    Higher diversity often leads to better ensemble performance.
    
    Args:
        predictions_list: List of prediction arrays from different models
        
    Returns:
        diversity_metrics: Dictionary of diversity metrics
    """
    n_models = len(predictions_list)
    n_samples, n_targets = predictions_list[0].shape
    
    # Calculate pairwise correlations
    correlations = []
    for i in range(n_models):
        for j in range(i+1, n_models):
            for target_idx in range(n_targets):
                corr = np.corrcoef(
                    predictions_list[i][:, target_idx],
                    predictions_list[j][:, target_idx]
                )[0, 1]
                correlations.append(corr)
    
    avg_correlation = np.mean(correlations)
    
    # Calculate disagreement (average pairwise MAE)
    disagreements = []
    for i in range(n_models):
        for j in range(i+1, n_models):
            mae = mean_absolute_error(predictions_list[i], predictions_list[j])
            disagreements.append(mae)
    
    avg_disagreement = np.mean(disagreements)
    
    return {
        'avg_correlation': avg_correlation,
        'avg_disagreement': avg_disagreement,
        'n_models': n_models
    }


# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """Example of how to use ensemble and stacking"""
    
    # Simulate predictions from 3 different models
    n_samples = 1000
    n_targets = 5
    
    predictions_list = [
        np.random.randn(n_samples, n_targets) * 10 + 50,  # Model 1
        np.random.randn(n_samples, n_targets) * 10 + 50,  # Model 2
        np.random.randn(n_samples, n_targets) * 10 + 50,  # Model 3
    ]
    
    targets = np.random.randn(n_samples, n_targets) * 10 + 50
    
    print("="*80)
    print("Ensemble and Stacking Example")
    print("="*80)
    
    # 1. Simple average
    print("\n1. Simple Average Ensemble")
    simple_pred = simple_average_ensemble(predictions_list)
    r2 = r2_score(targets, simple_pred)
    print(f"   R2: {r2:.4f}")
    
    # 2. Optimized weights
    print("\n2. Weighted Average Ensemble (Optimized)")
    optimal_weights = optimize_ensemble_weights(predictions_list, targets, method='scipy')
    print(f"   Optimal weights: {[f'{w:.3f}' for w in optimal_weights]}")
    weighted_pred = weighted_average_ensemble(predictions_list, optimal_weights)
    r2 = r2_score(targets, weighted_pred)
    print(f"   R2: {r2:.4f}")
    
    # 3. Stacking
    print("\n3. Stacking with CatBoost")
    stacker = StackingEnsemble(n_folds=5)
    stacker.fit(predictions_list, targets, verbose=True)
    if len(stacker.meta_models) > 0:
        stacked_pred = stacker.predict(predictions_list)
        r2 = r2_score(targets, stacked_pred)
        print(f"   Stacking R2: {r2:.4f}")
    else:
        print("   Stacking skipped (CatBoost not installed)")
    
    # 4. Diversity analysis
    print("\n4. Model Diversity Analysis")
    diversity = calculate_model_diversity(predictions_list)
    print(f"   Average correlation: {diversity['avg_correlation']:.4f}")
    print(f"   Average disagreement: {diversity['avg_disagreement']:.4f}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    example_usage()
