# DARTS (Differentiable Architecture Search)

## Description

DARTS (Differentiable Architecture Search) is a Neural Architecture Search (NAS) method that addresses the scalability challenge by formulating architecture search in a differentiable way. It relaxes the discrete search into a continuous search space, enabling the use of gradient-based optimization (gradient descent) to find the best architecture. This drastically reduces the computational cost of the search compared to non-differentiable optimization methods.

## Statistics

Significant reduction in search cost (GPU hours). State-of-the-art results on tasks such as image classification (e.g., 94.36% accuracy on CIFAR-10). The original paper has been widely cited (more than 6,000 citations).

## Features

Continuous Relaxation of the Search Space; Gradient-Based Optimization (Gradient Descent); Weight Sharing within a supernetwork; Search for 'Cells' (building blocks) that are then stacked; Use of second-order optimization.

## Use Cases

Image Classification (CIFAR-10, ImageNet); Recurrent Models (RNNs); Starting point for more advanced NAS methods (e.g., second-order DARTS, P-DARTS, PC-DARTS) and applications in computer vision and natural language processing.

## Integration

Integration generally involves implementing the supernetwork and the bi-level optimization algorithm (weights and architecture). Code example (conceptual in PyTorch):\n```python\n# Installation (example of a popular implementation)\n# pip install darts-nas\n\n# Conceptual example of bi-level optimization\n# Optimizer for model weights (w)\noptimizer_w = torch.optim.SGD(model.weights(), lr=LR_W)\n# Optimizer for architecture parameters (alpha)\noptimizer_a = torch.optim.Adam(model.alphas(), lr=LR_A, betas=(0.5, 0.999), weight_decay=WEIGHT_DECAY)\n\n# Training loop\nfor step in range(num_steps):\n    # 1. Optimize weights (w) using the training set\n    loss_w = model(input_train, target_train)\n    optimizer_w.zero_grad()\n    loss_w.backward()\n    optimizer_w.step()\n\n    # 2. Optimize architecture (alpha) using the validation set\n    loss_a = model(input_valid, target_valid)\n    optimizer_a.zero_grad()\n    loss_a.backward()\n    optimizer_a.step()\n```

## URL

https://arxiv.org/abs/1806.09055
