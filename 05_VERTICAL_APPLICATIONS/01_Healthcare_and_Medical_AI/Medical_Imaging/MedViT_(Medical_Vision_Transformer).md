# MedViT (Medical Vision Transformer)

## Description

MedViT (Medical Vision Transformer) is a robust hybrid model that combines the local feature extraction capability of Convolutional Neural Networks (CNNs) with the global connectivity of Vision Transformers (ViTs). It was designed specifically for generalized medical image classification, addressing concerns about the fragility of deep diagnostic models against adversarial attacks. The model uses an efficient convolution-based attention mechanism to mitigate the high quadratic complexity of the standard ViT self-attention mechanism. In addition, it incorporates a shape augmentation technique to learn smoother decision boundaries, which increases its robustness and generalization capability across diverse medical datasets. An enhanced version, MedViT-V2, integrates Kolmogorov-Arnold Networks (KAN) for architectural improvements and a new corruption benchmark.

## Statistics

- **Publication:** *Computers in Biology and Medicine*, 2023.
- **Citations:** 220 stars on GitHub (as of 2025).
- **Performance (ImageNet-1K Pre-training):**
    - MedViT\_small: Acc@1 of 83.70%
    - MedViT\_base: Acc@1 of 83.92%
    - MedViT\_large: Acc@1 of 83.96%
- **Advantage:** Demonstrates high robustness and generalization compared to baseline ResNets in terms of the *trade-off* between Accuracy/AUC and the number of Parameters across all MedMNIST-2D datasets.
- **Complexity:** Lower computational complexity compared to standard ViTs due to the convolution-based attention mechanism.

## Features

- **Hybrid CNN-Transformer Model:** Combines the locality of CNNs with the global connectivity of ViTs.
- **Efficient Attention Mechanism:** Uses an efficient convolution operation for the self-attention mechanism, reducing quadratic complexity.
- **Robustness to Adversarial Attacks:** Designed to learn smoother decision boundaries through shape augmentation, providing high robustness.
- **Generalization on Medical Images:** Demonstrates high generalization capability across a broad collection of MedMNIST-2D datasets.
- **MedViT-V2:** Enhanced version that integrates Kolmogorov-Arnold Networks (KAN) for architectural improvements.

## Use Cases

- **Generalized Medical Image Classification:** Automated diagnosis of diseases across diverse 2D datasets (for example, MedMNIST-2D).
- **Image Analysis with Robustness:** Applications where reliability against adversarial attacks is critical, such as in clinical diagnostic systems.
- **Visualization and Interpretability:** Use of techniques such as Grad-CAM for visual inspection and diagnostic interpretability on medical datasets.
- **Development of Hybrid Architectures:** Serves as a foundation for developing models that combine the best of CNNs and Transformers.

## Integration

The model is implemented in PyTorch. The official repository provides the source code and an instructions *notebook* (`Instructions.ipynb`) for training and evaluation.

**Usage Example (High-Level Structure):**

```python
import torch
from MedViT import MedViT_small # Assuming the MedViT_small class is in the file MedViT.py

# 1. Load the model
# The MedViT_small model pre-trained on ImageNet-1K
model = MedViT_small(pretrained=True) 
model.eval()

# 2. Prepare the input image (example)
# The image must be pre-processed (resized to 224x224, normalized)
# Example input tensor (batch_size=1, channels=3, height=224, width=224)
dummy_input = torch.randn(1, 3, 224, 224) 

# 3. Perform inference
with torch.no_grad():
    output = model(dummy_input)

# 4. Process the output (depends on the classification task)
# print(output.shape) 
# print(torch.argmax(output, dim=1))

# The repository also offers a guide for custom datasets.
# For training, follow the instructions in the 'Instructions.ipynb' or 'CustomDataset.md' file
```

## URL

https://github.com/Omid-Nejati/MedViT
