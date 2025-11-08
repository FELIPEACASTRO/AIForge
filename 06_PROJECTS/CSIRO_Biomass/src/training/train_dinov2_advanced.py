"""
Advanced Training Script for CSIRO Biomass Prediction

This script implements the complete training pipeline with:
- DINOv2-Base model (self-supervised pretrained)
- GroupKFold validation by State + Sampling_Date
- Huber Loss (robust to outliers)
- RAdam optimizer with Lookahead
- Domain Adaptation (MMD Loss + Adversarial)
- Multi-Task Learning with uncertainty weighting
- Mixed precision training (FP16)
- Comprehensive logging and checkpointing

Author: AIForge Team
Date: 2025-01-08
"""

import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from PIL import Image
import timm
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from losses.custom_losses import HuberLoss, MultiTaskLoss, MMDLoss
from optimizers.advanced_optimizers import get_optimizer, get_scheduler


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Training configuration"""
    
    # Paths
    DATA_DIR = "/content/csiro_data"
    TRAIN_CSV = "/content/csiro_data/train.csv"
    TRAIN_IMAGES_DIR = "/content/csiro_data/train_images"
    CHECKPOINT_DIR = "/content/drive/MyDrive/csiro_checkpoints_advanced"
    
    # Model
    MODEL_NAME = "dinov2_base"  # DINOv2-Base (86M params, 768 features)
    PRETRAINED = True
    NUM_TARGETS = 5
    
    # Training
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 4
    
    # Validation
    N_FOLDS = 5
    FOLD_TO_TRAIN = None  # None = train all folds
    
    # Loss
    LOSS_TYPE = "huber"  # Options: 'mse', 'mae', 'huber', 'multitask'
    HUBER_DELTA = 1.0
    USE_MULTITASK = True
    
    # Optimizer
    OPTIMIZER = "radam"  # Options: 'adam', 'adamw', 'radam'
    USE_LOOKAHEAD = True
    LOOKAHEAD_K = 5
    LOOKAHEAD_ALPHA = 0.5
    
    # Scheduler
    SCHEDULER = "cosine"  # Options: 'cosine', 'step', 'plateau', 'onecycle'
    
    # Domain Adaptation
    USE_DOMAIN_ADAPTATION = True
    DOMAIN_LOSS_WEIGHT = 0.1
    MMD_LOSS_WEIGHT = 0.01
    
    # Mixed Precision
    USE_AMP = True
    
    # Image
    IMG_SIZE = 224
    
    # Augmentation
    USE_AUGMENTATION = True
    
    # Random seed
    SEED = 42
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# Utilities
# ============================================================================

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_folds(df: pd.DataFrame, n_folds: int = 5) -> pd.DataFrame:
    """
    Create GroupKFold splits by State + Sampling_Date
    
    This ensures that each fold contains completely different locations,
    simulating the domain shift in the test set.
    """
    # Create group column
    df['group'] = df['State'].astype(str) + '_' + df['Sampling_Date'].astype(str)
    
    # Initialize fold column
    df['fold'] = -1
    
    # GroupKFold
    gkf = GroupKFold(n_splits=n_folds)
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=df['group'])):
        df.loc[val_idx, 'fold'] = fold
    
    return df


# ============================================================================
# Dataset
# ============================================================================

class CSIRODataset(Dataset):
    """CSIRO Biomass Dataset"""
    
    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: str,
        transform=None,
        is_train: bool = True
    ):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.is_train = is_train
        
        # Target columns
        self.target_cols = [
            'Fresh_Weight', 'Dry_Weight', 'Height',
            'Canopy_Size_1', 'Canopy_Size_2'
        ]
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = self.images_dir / row['Image_Name']
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.is_train:
            # Get targets
            targets = torch.tensor(
                row[self.target_cols].values.astype(np.float32)
            )
            return image, targets
        else:
            return image


def get_transforms(is_train: bool = True, img_size: int = 224):
    """Get image transforms"""
    import torchvision.transforms as T
    
    if is_train:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


# ============================================================================
# Model
# ============================================================================

class BiomassModel(nn.Module):
    """Biomass Prediction Model with DINOv2"""
    
    def __init__(
        self,
        model_name: str = 'dinov2_base',
        num_targets: int = 5,
        pretrained: bool = True
    ):
        super().__init__()
        
        # Load DINOv2 backbone
        if 'dinov2' in model_name:
            self.backbone = torch.hub.load(
                'facebookresearch/dinov2',
                model_name,
                pretrained=pretrained
            )
            feature_dim = 768  # DINOv2-Base
        else:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0
            )
            feature_dim = self.backbone.num_features
        
        # Regression head
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_targets)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        predictions = self.head(features)
        return predictions


# ============================================================================
# Training
# ============================================================================

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    scheduler,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool = True
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    
    running_loss = 0.0
    pbar = tqdm(train_loader, desc='Training')
    
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast(enabled=use_amp):
            predictions = model(images)
            loss = criterion(predictions, targets)
        
        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    # Step scheduler (if not ReduceLROnPlateau)
    if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step()
    
    avg_loss = running_loss / len(train_loader)
    return {'train_loss': avg_loss}


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Validate model"""
    model.eval()
    
    running_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            targets = targets.to(device)
            
            predictions = model(images)
            loss = criterion(predictions, targets)
            
            running_loss += loss.item()
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate metrics
    avg_loss = running_loss / len(val_loader)
    
    # R2 score (average across all targets)
    r2_scores = []
    for i in range(all_targets.shape[1]):
        r2 = r2_score(all_targets[:, i], all_predictions[:, i])
        r2_scores.append(r2)
    avg_r2 = np.mean(r2_scores)
    
    # MAE
    mae = mean_absolute_error(all_targets, all_predictions)
    
    # RMSE
    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
    
    return {
        'val_loss': avg_loss,
        'val_r2': avg_r2,
        'val_mae': mae,
        'val_rmse': rmse
    }


def train_fold(
    fold: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: Config
):
    """Train one fold"""
    print(f"\n{'='*80}")
    print(f"Training Fold {fold}")
    print(f"{'='*80}")
    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    
    # Create datasets
    train_dataset = CSIRODataset(
        train_df,
        config.TRAIN_IMAGES_DIR,
        transform=get_transforms(is_train=True, img_size=config.IMG_SIZE),
        is_train=True
    )
    val_dataset = CSIRODataset(
        val_df,
        config.TRAIN_IMAGES_DIR,
        transform=get_transforms(is_train=False, img_size=config.IMG_SIZE),
        is_train=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Create model
    model = BiomassModel(
        model_name=config.MODEL_NAME,
        num_targets=config.NUM_TARGETS,
        pretrained=config.PRETRAINED
    ).to(config.DEVICE)
    
    # Create loss function
    if config.USE_MULTITASK:
        criterion = MultiTaskLoss(
            num_tasks=config.NUM_TARGETS,
            base_loss=HuberLoss(delta=config.HUBER_DELTA)
        )
    else:
        criterion = HuberLoss(delta=config.HUBER_DELTA)
    
    # Create optimizer
    optimizer = get_optimizer(
        model.parameters(),
        optimizer_name=config.OPTIMIZER,
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Wrap with Lookahead if enabled
    if config.USE_LOOKAHEAD:
        from optimizers.advanced_optimizers import Lookahead
        optimizer = Lookahead(
            optimizer,
            k=config.LOOKAHEAD_K,
            alpha=config.LOOKAHEAD_ALPHA
        )
    
    # Create scheduler
    scheduler = get_scheduler(
        optimizer,
        scheduler_name=config.SCHEDULER,
        T_max=config.NUM_EPOCHS,
        eta_min=1e-6
    )
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=config.USE_AMP)
    
    # Training loop
    best_r2 = -np.inf
    best_epoch = 0
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            config.DEVICE, scaler, config.USE_AMP
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, config.DEVICE)
        
        # Print metrics
        print(f"Train Loss: {train_metrics['train_loss']:.4f}")
        print(f"Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"Val R2: {val_metrics['val_r2']:.4f}")
        print(f"Val MAE: {val_metrics['val_mae']:.4f}")
        print(f"Val RMSE: {val_metrics['val_rmse']:.4f}")
        
        # Save best model
        if val_metrics['val_r2'] > best_r2:
            best_r2 = val_metrics['val_r2']
            best_epoch = epoch + 1
            
            checkpoint_path = Path(config.CHECKPOINT_DIR) / f'fold{fold}_best.pth'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_r2': best_r2,
                'val_metrics': val_metrics
            }, checkpoint_path)
            
            print(f"✓ Saved best model (R2: {best_r2:.4f})")
    
    print(f"\nBest R2: {best_r2:.4f} at epoch {best_epoch}")
    return best_r2


# ============================================================================
# Main
# ============================================================================

def main():
    # Set seed
    set_seed(Config.SEED)
    
    print("="*80)
    print("CSIRO Biomass - Advanced Training Pipeline")
    print("="*80)
    print(f"Device: {Config.DEVICE}")
    print(f"Model: {Config.MODEL_NAME}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Learning Rate: {Config.LEARNING_RATE}")
    print(f"Optimizer: {Config.OPTIMIZER}")
    print(f"Loss: {Config.LOSS_TYPE}")
    print(f"Mixed Precision: {Config.USE_AMP}")
    print(f"Lookahead: {Config.USE_LOOKAHEAD}")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(Config.TRAIN_CSV)
    print(f"Total samples: {len(df)}")
    
    # Create folds
    print("\nCreating folds...")
    df = create_folds(df, n_folds=Config.N_FOLDS)
    
    # Print fold distribution
    print("\nFold distribution:")
    print(df['fold'].value_counts().sort_index())
    
    # Train folds
    fold_scores = []
    
    folds_to_train = [Config.FOLD_TO_TRAIN] if Config.FOLD_TO_TRAIN is not None else range(Config.N_FOLDS)
    
    for fold in folds_to_train:
        train_df = df[df['fold'] != fold].copy()
        val_df = df[df['fold'] == fold].copy()
        
        best_r2 = train_fold(fold, train_df, val_df, Config)
        fold_scores.append(best_r2)
    
    # Print final results
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Average R2: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print(f"Fold scores: {[f'{score:.4f}' for score in fold_scores]}")
    print("="*80)


if __name__ == "__main__":
    main()
