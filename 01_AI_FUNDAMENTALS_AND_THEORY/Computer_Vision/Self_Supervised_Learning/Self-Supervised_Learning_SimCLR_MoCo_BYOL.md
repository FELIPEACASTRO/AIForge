# Self-Supervised Learning: SimCLR, MoCo, BYOL

## Description

SimCLR, MoCo, and BYOL are Self-Supervised Learning (SSL) methods that revolutionized the way Computer Vision models learn visual representations without the need for human labels. They fall into the category of Contrastive Learning (SimCLR, MoCo) or similarity-based methods (BYOL), with the common goal of learning a representation space where the transformations (augmented views) of the same image (positive pairs) are grouped together, while the representations of different images (negative pairs, except in BYOL) are separated. The success of these models lies in their ability to pre-train robust encoders on large unlabeled datasets, which can then be efficiently fine-tuned for downstream tasks with limited labeled data.

## Statistics

**Performance (Top-1 Accuracy on ImageNet, Linear Fine-Tuning with ResNet-50):**
*   **SimCLR (v2):** 71.1% (with batch of 4096)
*   **MoCo (v2):** 71.1% (with batch of 256 and queue of 65536)
*   **BYOL:** 74.3% (no negatives, with batch normalization)

**Training Requirements:**
| Method | Batch Size | Use of Negatives | Mechanism to Avoid Collapse |
| :--- | :--- | :--- | :--- |
| **SimCLR** | Very Large (e.g., 4096) | Yes (other samples in the batch) | Large batch and strong augmentations |
| **MoCo** | Small to Medium (e.g., 256) | Yes (feature queue) | Momentum Encoder and Feature Queue |
| **BYOL** | Small to Medium (e.g., 256) | No | Momentum Target Network and Predictor |

**Note:** BYOL proved to be more robust to different data augmentations than SimCLR, and its version with group normalization (GN) + weight standardization (WS) achieved 74.3% Top-1 accuracy, surpassing the version with Batch Normalization (BN) (73.9%).

## Features

**SimCLR (Simple Framework for Contrastive Learning of Visual Representations):**
*   **Architecture:** Encoder (ResNet) + Projection Head (non-linear MLP).
*   **Mechanism:** Maximizes the agreement between two augmented views of the same image (positive pair) and minimizes the agreement with all other images in the batch (negative pairs).
*   **Key Requirement:** Requires a **very large batch size** (e.g., 4096) to provide a sufficient number of high-quality negative samples.
*   **Loss Function:** NT-Xent (Normalized Temperature-Scaled Cross-Entropy Loss).

**MoCo (Momentum Contrast):**
*   **Architecture:** Query Encoder and Momentum Key Encoder + Feature Queue.
*   **Mechanism:** Treats Contrastive Learning as a dictionary lookup. The Key Encoder is updated by a smooth Exponential Moving Average (EMA) of the Query Encoder, and the Feature Queue stores representations of past keys, enabling a **large and consistent dictionary of negatives** without the need for a large batch.
*   **Key Requirement:** Allows a smaller batch size and is more memory-efficient than SimCLR.

**BYOL (Bootstrap Your Own Latent):**
*   **Architecture:** Online Network (Student) and Momentum Target Network (Teacher) + Predictor.
*   **Mechanism:** Trains the Online Network to predict the representation of the Target Network from a different augmented view of the same image. **It does not use negative pairs**. The Target Network is an EMA of the Online Network, providing a stable target for learning.
*   **Unique Value Proposition:** Avoids collapse (trivial solution) without using negatives, which makes it more robust to different data augmentations.

## Use Cases

Pre-training with SimCLR, MoCo, and BYOL is widely used to generate robust representations in domains with scarce labeled data.

*   **General Computer Vision:** Pre-training of encoders for classification, object detection, and semantic segmentation tasks on datasets such as ImageNet, COCO, and Pascal VOC.
*   **Medical Imaging:** Training on large volumes of unlabeled medical images (e.g., X-rays, MRIs) to learn relevant features, followed by fine-tuning on specific tasks such as tumor detection or disease classification.
*   **Video Analysis:** Adaptation of MoCo and BYOL to learn video representations, where the contrast is made between different frames of the same video (positive pairs) or between different videos (negatives).
*   **Recommendation Systems:** Use of learned representations to encode items (e.g., products, movies) or user interactions, improving the quality of recommendations on platforms such as Spotify and Pinterest.
*   **Robotics and Autonomous Vehicles:** Learning visual representations for scene understanding and real-time object detection, where collecting labeled data is expensive and dangerous.

## Integration

The integration of these models generally involves two stages: pre-training and fine-tuning.

**Pre-training (Example with PyTorch - Conceptual):**
```python
import torch
import torch.nn as nn
from torchvision import models

# 1. Define the Encoder (e.g., ResNet-50)
encoder = models.resnet50(pretrained=False)
# Remove the final classification layer
encoder.fc = nn.Identity() 

# 2. Define the Specific Architecture (SimCLR, MoCo, or BYOL)
# For SimCLR: Add Projection Head (MLP)
projection_head = nn.Sequential(
    nn.Linear(2048, 2048),
    nn.ReLU(),
    nn.Linear(2048, 128) # Projection space
)

# For MoCo/BYOL: Define the Momentum Update logic
# Example of momentum update for MoCo/BYOL
@torch.no_grad()
def update_momentum_encoder(online_net, target_net, m=0.999):
    for param_q, param_k in zip(online_net.parameters(), target_net.parameters()):
        param_k.data = param_k.data * m + param_q.data * (1. - m)

# 3. Train with the appropriate Loss Function (NT-Xent for SimCLR/MoCo, MSE for BYOL)
# The actual training code involves generating augmented views (t1, t2) and computing the loss.
```

**Fine-Tuning:**
After pre-training, the `encoder` (ResNet without the projection head) is used as a feature extractor and a new classification layer is added for the downstream task.

```python
# Load the pre-trained encoder
pretrained_encoder = load_pretrained_weights(encoder_path)

# Add a new classification layer
classifier = nn.Sequential(
    pretrained_encoder,
    nn.Linear(2048, num_classes) # num_classes is the number of classes in the downstream task
)

# Train the classifier on the downstream task (e.g., CIFAR-10)
# ... (standard supervised training code)
```

## URL

SimCLR: https://arxiv.org/abs/2002.05709 | MoCo: https://arxiv.org/abs/1911.05722 | BYOL: https://arxiv.org/abs/2006.07733