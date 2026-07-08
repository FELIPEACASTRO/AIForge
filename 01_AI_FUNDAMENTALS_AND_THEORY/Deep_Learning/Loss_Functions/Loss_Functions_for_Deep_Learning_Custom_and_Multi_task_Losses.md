# Loss Functions for Deep Learning: Custom and Multi-task Losses

## Description

**Custom Losses** are user-defined mathematical expressions that quantify the discrepancy between the predicted output of a Deep Learning model and the ground-truth value, allowing the optimization process to be tuned for specific objectives that standard loss functions do not address. Their unique value proposition lies in the ability to **incorporate domain knowledge and business objectives directly into the training process**, enabling the optimization of non-differentiable metrics (such as F1-Score or IoU) and the handling of specific data conditions (such as class imbalance or asymmetric penalties). **Multi-task Losses** are the weighted sum of the individual losses of each task in a Multi-task Learning (MTL) model. Their unique value proposition is the **ability to leverage knowledge shared across related tasks to improve the performance of all tasks**, with the main challenge being **loss balancing** to prevent negative transfer and ensure balanced training.

## Statistics

**Performance Gain in MTL:** Studies show that loss balancing (e.g., GradNorm, Uncertainty Weighting) can lead to **improvements of 5% to 15%** in average task performance (on metrics such as mIoU, mAP, or accuracy) compared to a simple sum of losses. **Loss Balancing Metrics:** In MTL, metrics such as the **per-task gradient norm** and **homoscedastic uncertainty** are used internally to guide the optimization process, acting as dynamic control metrics. **Crucial Distinction:** The **Loss Function** (used for optimization, must be differentiable) is distinct from the **Metric** (used for evaluation, may be non-differentiable).

## Features

**Custom Losses:** Incorporation of Constraints (physical, geometric), Business Optimization (translating error cost into differentiable terms), and Distribution Manipulation (e.g., Focal Loss to focus on hard examples). **Multi-task Losses:** Uncertainty-based Balancing (Uncertainty Weighting, which learns weights dynamically based on the uncertainty of each task), Gradient Normalization (GradNorm, which adjusts weights so that gradients have similar magnitudes), and Gradient Surgery (PCGrad, which modifies gradients to avoid destructive conflicts between tasks).

## Use Cases

**Custom Losses:** **Computer Vision** (Using Dice Loss or IoU Loss to optimize pixel overlap in image segmentation), **Finance/Insurance** (Asymmetric losses that penalize risk underestimation, or False Negatives, much more than overestimation), and **Natural Language Processing (NLP)** (Losses that incorporate diversity or coherence metrics in text generation). **Multi-task Losses:** **Autonomous Driving** (Training a single model to simultaneously perform object detection, semantic segmentation, and depth estimation), **Medical Image Analysis** (A model that segments a tumor and classifies the stage of the disease from the same image), and **NLP** (A model that performs part-of-speech tagging and named entity recognition simultaneously).

## Integration

Integration is done by defining a function or class that inherits from the framework's loss class (e.g., `nn.Module` in PyTorch or `tf.keras.losses.Loss` in TensorFlow).

**Example 1: Custom Loss (Dice Loss in PyTorch)**
```python
import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice
```

**Example 2: Multi-task Loss (Uncertainty Weighting in PyTorch)**
```python
import torch
import torch.nn as nn

class UncertaintyLoss(nn.Module):
    def __init__(self, num_tasks):
        super(UncertaintyLoss, self).__init__()
        self.log_sigmas = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_sigmas[i])
            total_loss += precision * loss + 0.5 * self.log_sigmas[i]
        return total_loss
```

## URL

https://arxiv.org/abs/1705.07113, https://arxiv.org/abs/1711.02257, https://arxiv.org/abs/2001.06782
