"""
Domain Adaptation Techniques for CSIRO Biomass Prediction

This module implements domain adaptation methods to handle the domain shift
between training locations and test locations.

Critical Problem:
- Training data: NSW, VIC, QLD, SA locations
- Test data: NEW LOCATIONS (domain shift)
- Solution: Domain adaptation to generalize to unseen locations

Author: AIForge Team
Date: 2025-01-08
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class DomainAdaptationModel(nn.Module):
    """
    Domain Adaptation Model Wrapper
    
    Wraps a base model with domain adaptation capabilities:
    - Feature extractor (shared between domains)
    - Task predictor (biomass prediction)
    - Domain discriminator (adversarial training)
    
    Args:
        base_model: Base feature extraction model (e.g., DINOv2, EfficientNet)
        feature_dim: Dimension of extracted features
        num_targets: Number of prediction targets (5 for CSIRO)
        use_gradient_reversal: Whether to use gradient reversal layer
    """
    def __init__(
        self,
        base_model: nn.Module,
        feature_dim: int = 768,
        num_targets: int = 5,
        use_gradient_reversal: bool = True
    ):
        super().__init__()
        
        # Feature extractor (shared)
        self.feature_extractor = base_model
        
        # Task predictor (biomass regression)
        self.task_predictor = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_targets)
        )
        
        # Domain discriminator (binary classification: source vs target)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.use_gradient_reversal = use_gradient_reversal
        
    def forward(
        self,
        x: torch.Tensor,
        alpha: float = 1.0,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with domain adaptation.
        
        Args:
            x: Input images (batch_size, 3, H, W)
            alpha: Gradient reversal strength (0 to 1)
            return_features: Whether to return intermediate features
            
        Returns:
            outputs: Dictionary containing:
                - 'predictions': Biomass predictions (batch_size, num_targets)
                - 'domain_preds': Domain predictions (batch_size, 1)
                - 'features': Extracted features (optional)
        """
        # Extract features
        features = self.feature_extractor(x)
        
        # Task prediction
        predictions = self.task_predictor(features)
        
        # Domain prediction with gradient reversal
        if self.use_gradient_reversal:
            reversed_features = GradientReversalLayer.apply(features, alpha)
            domain_preds = self.domain_discriminator(reversed_features)
        else:
            domain_preds = self.domain_discriminator(features)
        
        outputs = {
            'predictions': predictions,
            'domain_preds': domain_preds
        }
        
        if return_features:
            outputs['features'] = features
        
        return outputs


class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL)
    
    Forward pass: identity function
    Backward pass: reverses gradients (multiplies by -alpha)
    
    This encourages domain-invariant features by making the feature
    extractor unable to distinguish between source and target domains.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class DomainAdaptationLoss(nn.Module):
    """
    Combined Loss for Domain Adaptation
    
    Total Loss = Task Loss + λ * Domain Loss + β * MMD Loss
    
    Args:
        task_loss_fn: Loss function for task (e.g., HuberLoss)
        domain_loss_weight: Weight for domain adversarial loss (λ)
        mmd_loss_weight: Weight for MMD loss (β)
        use_mmd: Whether to use MMD loss
    """
    def __init__(
        self,
        task_loss_fn: nn.Module,
        domain_loss_weight: float = 0.1,
        mmd_loss_weight: float = 0.01,
        use_mmd: bool = True
    ):
        super().__init__()
        self.task_loss_fn = task_loss_fn
        self.domain_loss_fn = nn.BCELoss()
        self.domain_loss_weight = domain_loss_weight
        self.mmd_loss_weight = mmd_loss_weight
        self.use_mmd = use_mmd
        
        if use_mmd:
            from ..losses.custom_losses import MMDLoss
            self.mmd_loss_fn = MMDLoss(kernel='gaussian', bandwidth=1.0)
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        domain_labels: torch.Tensor,
        source_features: Optional[torch.Tensor] = None,
        target_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate combined domain adaptation loss.
        
        Args:
            outputs: Model outputs dictionary
            targets: Ground truth biomass values
            domain_labels: Domain labels (0 = source, 1 = target)
            source_features: Features from source domain (for MMD)
            target_features: Features from target domain (for MMD)
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        # Task loss (only on source domain with labels)
        task_loss = self.task_loss_fn(outputs['predictions'], targets)
        
        # Domain adversarial loss
        domain_loss = self.domain_loss_fn(
            outputs['domain_preds'],
            domain_labels.unsqueeze(1).float()
        )
        
        # Total loss
        total_loss = task_loss + self.domain_loss_weight * domain_loss
        
        loss_dict = {
            'task_loss': task_loss.item(),
            'domain_loss': domain_loss.item()
        }
        
        # MMD loss (if available)
        if self.use_mmd and source_features is not None and target_features is not None:
            mmd_loss = self.mmd_loss_fn(source_features, target_features)
            total_loss += self.mmd_loss_weight * mmd_loss
            loss_dict['mmd_loss'] = mmd_loss.item()
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict


class PseudoLabelingStrategy:
    """
    Pseudo-Labeling for Target Domain
    
    Generates pseudo-labels for unlabeled target domain data
    to enable semi-supervised learning.
    
    Args:
        confidence_threshold: Minimum confidence to accept pseudo-label
        warmup_epochs: Number of epochs before starting pseudo-labeling
    """
    def __init__(
        self,
        confidence_threshold: float = 0.9,
        warmup_epochs: int = 10
    ):
        self.confidence_threshold = confidence_threshold
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
    
    def update_epoch(self, epoch: int):
        """Update current epoch"""
        self.current_epoch = epoch
    
    def should_use_pseudo_labels(self) -> bool:
        """Check if pseudo-labeling should be used"""
        return self.current_epoch >= self.warmup_epochs
    
    def generate_pseudo_labels(
        self,
        model: nn.Module,
        target_loader: torch.utils.data.DataLoader,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate pseudo-labels for target domain.
        
        Args:
            model: Trained model
            target_loader: DataLoader for target domain
            device: Device to use
            
        Returns:
            pseudo_images: Images with high-confidence predictions
            pseudo_labels: Pseudo-labels for these images
            confidences: Confidence scores
        """
        model.eval()
        
        pseudo_images_list = []
        pseudo_labels_list = []
        confidences_list = []
        
        with torch.no_grad():
            for images, _ in target_loader:
                images = images.to(device)
                
                # Get predictions
                outputs = model(images)
                predictions = outputs['predictions']
                
                # Calculate confidence (inverse of prediction variance)
                # For regression, we use negative variance as confidence
                confidence = -predictions.var(dim=1)
                
                # Select high-confidence samples
                high_conf_mask = confidence > self.confidence_threshold
                
                if high_conf_mask.any():
                    pseudo_images_list.append(images[high_conf_mask])
                    pseudo_labels_list.append(predictions[high_conf_mask])
                    confidences_list.append(confidence[high_conf_mask])
        
        if len(pseudo_images_list) > 0:
            pseudo_images = torch.cat(pseudo_images_list, dim=0)
            pseudo_labels = torch.cat(pseudo_labels_list, dim=0)
            confidences = torch.cat(confidences_list, dim=0)
        else:
            # Return empty tensors if no high-confidence samples
            pseudo_images = torch.empty(0, 3, 224, 224).to(device)
            pseudo_labels = torch.empty(0, 5).to(device)
            confidences = torch.empty(0).to(device)
        
        return pseudo_images, pseudo_labels, confidences


def create_domain_adaptation_model(
    model_name: str = 'dinov2_base',
    num_targets: int = 5,
    pretrained: bool = True,
    use_gradient_reversal: bool = True
) -> DomainAdaptationModel:
    """
    Factory function to create domain adaptation model.
    
    Args:
        model_name: Name of the base model
        num_targets: Number of prediction targets
        pretrained: Whether to use pretrained weights
        use_gradient_reversal: Whether to use gradient reversal
        
    Returns:
        model: DomainAdaptationModel instance
    """
    # Import base model
    if 'dinov2' in model_name.lower():
        import torch
        base_model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=pretrained)
        feature_dim = 768  # DINOv2-Base
    elif 'efficientnet' in model_name.lower():
        import timm
        base_model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        feature_dim = base_model.num_features
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Create domain adaptation model
    model = DomainAdaptationModel(
        base_model=base_model,
        feature_dim=feature_dim,
        num_targets=num_targets,
        use_gradient_reversal=use_gradient_reversal
    )
    
    return model


# Example usage
if __name__ == "__main__":
    # Test domain adaptation model
    model = create_domain_adaptation_model(
        model_name='dinov2_base',
        num_targets=5,
        pretrained=False
    )
    
    # Test forward pass
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    outputs = model(images, alpha=0.5, return_features=True)
    
    print("Domain Adaptation Model Test:")
    print(f"Predictions shape: {outputs['predictions'].shape}")
    print(f"Domain predictions shape: {outputs['domain_preds'].shape}")
    print(f"Features shape: {outputs['features'].shape}")
    
    # Test loss
    from ..losses.custom_losses import HuberLoss
    task_loss_fn = HuberLoss()
    da_loss = DomainAdaptationLoss(task_loss_fn)
    
    targets = torch.randn(batch_size, 5)
    domain_labels = torch.tensor([0, 0, 1, 1])  # 0=source, 1=target
    
    total_loss, loss_dict = da_loss(outputs, targets, domain_labels)
    print(f"\nLoss Test:")
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Loss breakdown: {loss_dict}")
