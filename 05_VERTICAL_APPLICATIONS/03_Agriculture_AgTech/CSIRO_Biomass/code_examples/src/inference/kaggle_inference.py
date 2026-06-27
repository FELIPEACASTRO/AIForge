"""
Kaggle Inference Script for CSIRO Biomass Prediction

This script generates predictions for the test set using trained models.
Supports:
- Single model inference
- Ensemble inference (average multiple folds)
- TTA (Test-Time Augmentation)
- Proper submission format

Usage in Kaggle Notebook:
    python kaggle_inference.py --checkpoint_dir /kaggle/input/csiro-checkpoints --output submission.csv

Author: AIForge Team
Date: 2025-01-08
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T


# ============================================================================
# Configuration
# ============================================================================

class InferenceConfig:
    """Inference configuration"""
    
    # Paths (will be set by command line args)
    TEST_CSV = "/kaggle/input/csiro-biomass/test.csv"
    TEST_IMAGES_DIR = "/kaggle/input/csiro-biomass/test_images"
    CHECKPOINT_DIR = "/kaggle/input/csiro-checkpoints"
    OUTPUT_PATH = "submission.csv"
    
    # Model
    MODEL_NAME = "dinov2_base"
    NUM_TARGETS = 5
    
    # Inference
    BATCH_SIZE = 32
    NUM_WORKERS = 2
    
    # Ensemble
    USE_ENSEMBLE = True  # Average predictions from all folds
    NUM_FOLDS = 5
    
    # TTA (Test-Time Augmentation)
    USE_TTA = True
    TTA_TRANSFORMS = 4  # Number of augmented versions
    
    # Image
    IMG_SIZE = 224
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# Model
# ============================================================================

class BiomassModel(nn.Module):
    """Biomass Prediction Model with DINOv2"""
    
    def __init__(
        self,
        model_name: str = 'dinov2_base',
        num_targets: int = 5,
        pretrained: bool = False
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
            import timm
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
# Dataset
# ============================================================================

class CSIROTestDataset(Dataset):
    """CSIRO Test Dataset"""
    
    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: str,
        transform=None
    ):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        img_path = self.images_dir / row['Image_Name']
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, row['Image_Name']


def get_test_transforms(img_size: int = 224):
    """Get test transforms (no augmentation)"""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_tta_transforms(img_size: int = 224):
    """Get TTA transforms"""
    transforms = []
    
    # Original
    transforms.append(T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    
    # Horizontal flip
    transforms.append(T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=1.0),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    
    # Vertical flip
    transforms.append(T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomVerticalFlip(p=1.0),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    
    # Both flips
    transforms.append(T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=1.0),
        T.RandomVerticalFlip(p=1.0),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    
    return transforms


# ============================================================================
# Inference
# ============================================================================

def load_model(checkpoint_path: str, config: InferenceConfig) -> nn.Module:
    """Load trained model from checkpoint"""
    model = BiomassModel(
        model_name=config.MODEL_NAME,
        num_targets=config.NUM_TARGETS,
        pretrained=False
    ).to(config.DEVICE)
    
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def predict_single_model(
    model: nn.Module,
    test_loader: DataLoader,
    config: InferenceConfig,
    use_tta: bool = False
) -> np.ndarray:
    """Generate predictions using a single model"""
    model.eval()
    
    all_predictions = []
    
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc='Predicting'):
            images = images.to(config.DEVICE)
            
            if use_tta:
                # TTA: average predictions from multiple augmentations
                tta_predictions = []
                
                for tta_transform in get_tta_transforms(config.IMG_SIZE):
                    # Apply TTA transform
                    tta_images = torch.stack([
                        tta_transform(T.ToPILImage()(img))
                        for img in images
                    ]).to(config.DEVICE)
                    
                    predictions = model(tta_images)
                    tta_predictions.append(predictions.cpu().numpy())
                
                # Average TTA predictions
                predictions = np.mean(tta_predictions, axis=0)
            else:
                predictions = model(images)
                predictions = predictions.cpu().numpy()
            
            all_predictions.append(predictions)
    
    return np.concatenate(all_predictions, axis=0)


def predict_ensemble(
    checkpoint_dir: str,
    test_loader: DataLoader,
    config: InferenceConfig
) -> np.ndarray:
    """Generate predictions using ensemble of models"""
    fold_predictions = []
    
    for fold in range(config.NUM_FOLDS):
        checkpoint_path = Path(checkpoint_dir) / f'fold{fold}_best.pth'
        
        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint for fold {fold} not found. Skipping...")
            continue
        
        print(f"\nLoading fold {fold} model...")
        model = load_model(str(checkpoint_path), config)
        
        print(f"Generating predictions for fold {fold}...")
        predictions = predict_single_model(model, test_loader, config, use_tta=config.USE_TTA)
        fold_predictions.append(predictions)
        
        # Free memory
        del model
        torch.cuda.empty_cache()
    
    if len(fold_predictions) == 0:
        raise ValueError("No valid checkpoints found!")
    
    # Average predictions from all folds
    ensemble_predictions = np.mean(fold_predictions, axis=0)
    
    print(f"\nEnsemble: Averaged predictions from {len(fold_predictions)} folds")
    
    return ensemble_predictions


def create_submission(
    predictions: np.ndarray,
    test_df: pd.DataFrame,
    output_path: str
):
    """Create submission file in correct format"""
    target_cols = [
        'Fresh_Weight', 'Dry_Weight', 'Height',
        'Canopy_Size_1', 'Canopy_Size_2'
    ]
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'Image_Name': test_df['Image_Name'].values
    })
    
    for i, col in enumerate(target_cols):
        submission[col] = predictions[:, i]
    
    # Save to CSV
    submission.to_csv(output_path, index=False)
    print(f"\n✓ Submission saved to: {output_path}")
    print(f"  Shape: {submission.shape}")
    print(f"  Columns: {list(submission.columns)}")
    print(f"\nFirst few predictions:")
    print(submission.head())


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='CSIRO Biomass Inference')
    parser.add_argument('--test_csv', type=str, default=InferenceConfig.TEST_CSV)
    parser.add_argument('--test_images_dir', type=str, default=InferenceConfig.TEST_IMAGES_DIR)
    parser.add_argument('--checkpoint_dir', type=str, default=InferenceConfig.CHECKPOINT_DIR)
    parser.add_argument('--output', type=str, default=InferenceConfig.OUTPUT_PATH)
    parser.add_argument('--batch_size', type=int, default=InferenceConfig.BATCH_SIZE)
    parser.add_argument('--use_ensemble', action='store_true', default=InferenceConfig.USE_ENSEMBLE)
    parser.add_argument('--use_tta', action='store_true', default=InferenceConfig.USE_TTA)
    parser.add_argument('--num_folds', type=int, default=InferenceConfig.NUM_FOLDS)
    
    args = parser.parse_args()
    
    # Update config
    config = InferenceConfig()
    config.TEST_CSV = args.test_csv
    config.TEST_IMAGES_DIR = args.test_images_dir
    config.CHECKPOINT_DIR = args.checkpoint_dir
    config.OUTPUT_PATH = args.output
    config.BATCH_SIZE = args.batch_size
    config.USE_ENSEMBLE = args.use_ensemble
    config.USE_TTA = args.use_tta
    config.NUM_FOLDS = args.num_folds
    
    print("="*80)
    print("CSIRO Biomass - Kaggle Inference")
    print("="*80)
    print(f"Device: {config.DEVICE}")
    print(f"Test CSV: {config.TEST_CSV}")
    print(f"Test Images: {config.TEST_IMAGES_DIR}")
    print(f"Checkpoint Dir: {config.CHECKPOINT_DIR}")
    print(f"Output: {config.OUTPUT_PATH}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Use Ensemble: {config.USE_ENSEMBLE}")
    print(f"Use TTA: {config.USE_TTA}")
    print("="*80)
    
    # Load test data
    print("\nLoading test data...")
    test_df = pd.read_csv(config.TEST_CSV)
    print(f"Test samples: {len(test_df)}")
    
    # Create test dataset
    test_dataset = CSIROTestDataset(
        test_df,
        config.TEST_IMAGES_DIR,
        transform=get_test_transforms(config.IMG_SIZE)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Generate predictions
    if config.USE_ENSEMBLE:
        predictions = predict_ensemble(config.CHECKPOINT_DIR, test_loader, config)
    else:
        # Single model (fold 0)
        checkpoint_path = Path(config.CHECKPOINT_DIR) / 'fold0_best.pth'
        print(f"\nLoading model from: {checkpoint_path}")
        model = load_model(str(checkpoint_path), config)
        predictions = predict_single_model(model, test_loader, config, use_tta=config.USE_TTA)
    
    # Create submission
    create_submission(predictions, test_df, config.OUTPUT_PATH)
    
    print("\n" + "="*80)
    print("Inference Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
