# Multi-Task Learning (MTL) - Hard and Soft Parameter Sharing

## Description

Multi-Task Learning (MTL) is a machine learning approach that trains a single model on multiple related tasks simultaneously. It acts as a form of inductive transfer and implicit regularization, forcing the model to learn a shared representation that generalizes better and reduces the risk of overfitting. The two main parameter-sharing methods are **Hard Parameter Sharing**, where the hidden layers are fully shared, and **Soft Parameter Sharing**, where each task has its own model but the distance between their parameters is regularized to encourage similarity. Hard sharing is the most common and effective for closely related tasks, while soft sharing offers greater flexibility for more loosely related tasks.

## Statistics

**Overfitting Reduction:** The risk of overfitting on the shared parameters is an order of magnitude $N$ (number of tasks) smaller than on the task-specific parameters. **Performance Gains:** Typical improvements of 1% to 5% or more on primary task metrics compared to single-task learning (STL) models. **Efficiency:** Hard sharing drastically reduces the total number of model parameters.

## Features

**Hard Parameter Sharing:** Shared hidden layers, task-specific output layers. Drastic reduction of overfitting risk ($\mathcal{O}(1/N)$). **Soft Parameter Sharing:** Separate per-task models with regularization on the distance between parameters (e.g., $L_2$ norm, trace norm). Greater flexibility and robustness against negative transfer. **General MTL Benefits:** Implicit data augmentation, attention focus on relevant features, inductive bias for better generalization.

## Use Cases

**Natural Language Processing (NLP):** Models that simultaneously perform part-of-speech (POS) tagging, named entity recognition (NER), and syntactic parsing. **Computer Vision:** Joint prediction of semantic segmentation and depth estimation from a single image. **Recommendation Systems:** Simultaneous prediction of click probability and the user's dwell time on an item. **Autonomous Cars:** Predicting steering direction using auxiliary tasks such as road feature prediction (e.g., lane markings). **Medicine and Bioinformatics:** Simultaneous prediction of multiple symptoms or the activity of multiple compounds in drug discovery.

## Integration

Integration is typically carried out in Deep Learning frameworks such as PyTorch or TensorFlow/Keras. **Hard Sharing** is implemented by defining common hidden layers followed by separate output heads for each task. Optimization is done by minimizing a combined total loss, usually a weighted sum of the individual losses of each task. **Soft Sharing** is implemented by adding a regularization term to the total loss function that penalizes the difference between the parameters of the task-specific models.

```python
import torch
import torch.nn as nn

# Hard Sharing Example (PyTorch)
class HardSharingMTLModel(nn.Module):
    def __init__(self, input_size, shared_hidden_size, task_a_output_size, task_b_output_size):
        super().__init__()
        # Shared Layers
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, shared_hidden_size),
            nn.ReLU(),
            nn.Linear(shared_hidden_size, shared_hidden_size),
            nn.ReLU()
        )
        # Task-Specific Heads
        self.task_a_head = nn.Linear(shared_hidden_size, task_a_output_size)
        self.task_b_head = nn.Linear(shared_hidden_size, task_b_output_size)

    def forward(self, x):
        shared_representation = self.shared_layers(x)
        output_a = self.task_a_head(shared_representation)
        output_b = self.task_b_head(shared_representation)
        return output_a, output_b

# Combined Loss Optimization:
# total_loss = weight_a * loss_a + weight_b * loss_b
```

## URL

https://www.ruder.io/multi-task/